"""
bgremover/pipeline.py
---------------------
Core inference loop shared by all model types.

Handles:
    - Frame reading via OpenCV
    - Tensor conversion + model inference
    - RGBA frame composition
    - Preview mode (N evenly-spaced frames → PNG)
    - Full encode via bgremover.encoder
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from bgremover import __version__
from bgremover.encoder import (
    PngEncoder,
    VideoEncoder,
    compute_processing_size,
    extract_audio,
    remux_audio,
    resolve_output_path,
)

console = Console()

# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def frame_to_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a (H, W, 3) uint8 BGR frame to a (1, 3, H, W) float32 RGB tensor
    normalised to [0, 1] on *device*.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # BGR → RGB
    t = torch.from_numpy(rgb)                           # (H, W, 3) uint8
    t = t.permute(2, 0, 1)                              # → (3, H, W)
    t = t.unsqueeze(0)                                  # → (1, 3, H, W)
    t = t.float() / 255.0                               # uint8 [0,255] → float [0,1]
    return t.to(device)


def compute_downsample_ratio(width: int, height: int, target: int = 512) -> float:
    """
    Compute the RVM downsample_ratio so that the shorter side of the
    downsampled resolution is ~*target* pixels (clamped to (0, 1]).

    Formula:
        ratio = target / min(width, height)
    RVM paper recommends 256-512px shorter side for the recurrent stage.
    """
    return min(1.0, target / min(width, height))


def compose_rgba(
    fgr: torch.Tensor,
    pha: torch.Tensor,
) -> np.ndarray:
    """
    Build a (H, W, 4) uint8 RGBA frame ready to be piped to FFmpeg.

    Math:
        fgr  : (1, 3, H, W) float [0, 1] — model foreground RGB
        pha  : (1, 1, H, W) float [0, 1] — alpha matte
        R, G, B = fgr * pha              — pre-multiplied (straight alpha)
        A       = pha

    Pre-multiplied alpha is correct because:
        - It avoids colour fringing at semi-transparent edges.
        - FFmpeg/ProRes expect straight alpha; NLEs de-premultiply.
    """
    alpha = pha.squeeze().cpu().numpy()          # (H, W) float [0,1]
    fg    = fgr.squeeze().cpu().numpy()          # (3, H, W) float [0,1]
    fg    = np.transpose(fg, (1, 2, 0))          # → (H, W, 3) RGB

    # Pre-multiply
    fg_pm = fg * alpha[:, :, np.newaxis]         # (H, W, 3)

    rgba  = np.concatenate(
        [fg_pm, alpha[:, :, np.newaxis]], axis=2 # (H, W, 4)
    )
    return np.clip(rgba * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# RVM inference with recurrent state
# ---------------------------------------------------------------------------

def _infer_rvm(
    model: torch.jit.ScriptModule,
    src: torch.Tensor,
    rec: List,
    downsample_ratio: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, List]:
    """Run one RVM frame. Returns (fgr, pha, new_rec)."""
    with torch.no_grad():
        out = model(src, *rec, downsample_ratio)
    fgr, pha = out[0], out[1]
    new_rec = list(out[2:])
    return fgr, pha, new_rec


def _infer_bgmv2(
    model: torch.jit.ScriptModule,
    src: torch.Tensor,
    bgr: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one BGMv2 frame. Returns (fgr, pha)."""
    with torch.no_grad():
        pha, fgr = model(src, bgr)[:2]
    return fgr, pha


# ---------------------------------------------------------------------------
# Rich UI helpers
# ---------------------------------------------------------------------------

def _make_header(input_path: Path, total_frames: int, fps: float) -> Panel:
    title = Text(f"🎬  bgremover v{__version__}", style="bold cyan")
    info = (
        f"[bold]{input_path.name}[/bold]  ·  "
        f"[yellow]{total_frames}[/yellow] frames  ·  "
        f"[yellow]{fps:.1f}[/yellow] fps"
    )
    return Panel(Text.assemble(title, "\n", Text.from_markup(info)), expand=False)


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[cyan]{task.fields[fps]:.1f} fps"),
        console=console,
    )


# ---------------------------------------------------------------------------
# Preview mode
# ---------------------------------------------------------------------------

def run_preview(
    input_path: Path,
    n: int,
    model: torch.jit.ScriptModule,
    model_type: str,
    device: torch.device,
    bg_image: Optional[np.ndarray],
    max_short: int = 1080,
) -> Path:
    """
    Process *n* evenly-spaced frames and save PNG previews.
    Returns the output folder path.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open: {input_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc_w, proc_h = compute_processing_size(orig_w, orig_h, max_short)

    indices = [int(i * total / n) for i in range(n)]
    out_folder = input_path.parent / f"{input_path.stem}_preview"
    out_folder.mkdir(parents=True, exist_ok=True)

    bg_tensor = None
    if model_type == "bgmv2" and bg_image is not None:
        bg_resized = cv2.resize(bg_image, (proc_w, proc_h))
        bg_tensor = frame_to_tensor(bg_resized, device)

    rec = [None] * 4
    dr = torch.tensor([compute_downsample_ratio(proc_w, proc_h)], dtype=torch.float32).to(device)

    console.print(f"\n🔍  Preview mode: processing {n} frames …")

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (proc_w, proc_h))
        src = frame_to_tensor(frame, device)

        if model_type == "rvm":
            fgr, pha, rec = _infer_rvm(model, src, rec, dr)
        else:
            fgr, pha = _infer_bgmv2(model, src, bg_tensor)

        rgba = compose_rgba(fgr, pha)
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        out_file = out_folder / f"preview_{i+1:02d}_frame{idx:05d}.png"
        cv2.imwrite(str(out_file), bgra)
        console.print(f"   [{i+1}/{n}] frame {idx:05d} → {out_file.name}")

    cap.release()
    return out_folder


# ---------------------------------------------------------------------------
# Full processing pipeline
# ---------------------------------------------------------------------------

def run(
    input_path: Path,
    output_path: Path,
    model: torch.jit.ScriptModule,
    model_type: str,
    model_name: str,
    device: torch.device,
    bg_image: Optional[np.ndarray],
    fmt: str = "prores",
    max_short: int = 1080,
) -> None:
    """
    Process all frames of *input_path* and write to *output_path*.
    """
    # ---- Open video ----
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open: {input_path}")

    orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    proc_w, proc_h = compute_processing_size(orig_w, orig_h, max_short)
    needs_resize   = (proc_w, proc_h) != (orig_w, orig_h)

    if needs_resize:
        console.print(f"📐  Downscaled: {orig_w}×{orig_h} → {proc_w}×{proc_h}")

    # ---- Prepare background tensor (BGMv2 only) ----
    bg_tensor = None
    if model_type == "bgmv2":
        if bg_image is None:
            sys.exit("❌  BGMv2 requires a background plate. Use --bg-image or omit --model-type.")
        bg_resized = cv2.resize(bg_image, (proc_w, proc_h))
        bg_tensor = frame_to_tensor(bg_resized, device)

    # ---- RVM state ----
    rec = [None] * 4
    dr  = torch.tensor(
        [compute_downsample_ratio(proc_w, proc_h)],
        dtype=torch.float32,
    ).to(device)

    # ---- Rich header ----
    console.print(_make_header(input_path, total, fps))
    console.print(
        f"  🖥  [bold]{device}[/bold]  ·  "
        f"Model [bold]{model_name}[/bold]  ·  "
        f"Format [bold]{fmt}[/bold]\n"
    )

    # ---- Audio + temp dir ----
    tmp_dir  = tempfile.mkdtemp(prefix="bgremover_")
    audio    = extract_audio(str(input_path), tmp_dir)
    tmp_vid  = str(Path(tmp_dir) / f"video_noaudio{Path(output_path).suffix or '.mov'}")

    # ---- Encoding context ----
    EncoderClass: type
    if fmt == "png":
        enc_ctx = PngEncoder(output_path)
    else:
        enc_ctx = VideoEncoder(fmt, proc_w, proc_h, fps, tmp_vid)

    # ---- Frame loop with Rich progress ----
    t_start   = time.time()
    frame_idx = 0

    with _make_progress() as progress:
        task = progress.add_task(
            "Matting", total=total, fps=0.0
        )
        with enc_ctx as enc:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if needs_resize:
                    frame = cv2.resize(frame, (proc_w, proc_h))

                src = frame_to_tensor(frame, device)

                if model_type == "rvm":
                    fgr, pha, rec = _infer_rvm(model, src, rec, dr)
                else:
                    fgr, pha = _infer_bgmv2(model, src, bg_tensor)

                rgba = compose_rgba(fgr, pha)
                enc.write(rgba)

                frame_idx += 1
                elapsed    = time.time() - t_start
                cur_fps    = frame_idx / max(elapsed, 1e-6)
                progress.update(task, advance=1, fps=cur_fps)

    cap.release()
    elapsed_total = time.time() - t_start
    console.print(f"\n✅  Encoded [bold]{frame_idx}[/bold] frames in {elapsed_total:.1f}s")

    # ---- Remux audio (video formats only) ----
    if fmt != "png":
        if audio:
            console.print("🔊  Remuxing audio …")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            remux_audio(tmp_vid, audio, str(output_path))
        else:
            shutil.move(tmp_vid, str(output_path))

    # ---- Cleanup ----
    for f in [tmp_vid, audio]:
        if f and Path(f).exists():
            try:
                Path(f).unlink()
            except OSError:
                pass
    try:
        Path(tmp_dir).rmdir()
    except OSError:
        pass

    # ---- Final summary ----
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1_048_576
        console.print(
            f"\n🎉  [bold green]Done![/bold green]  "
            f"→ [underline]{output_path}[/underline]\n"
            f"    Size [yellow]{size_mb:.0f} MB[/yellow]  ·  "
            f"Resolution [yellow]{proc_w}×{proc_h}[/yellow]  ·  "
            f"Time [yellow]{elapsed_total:.1f}s[/yellow]"
        )
