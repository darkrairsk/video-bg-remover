"""
bgremover/encoder.py
--------------------
FFmpeg pipe abstraction for writing RGBA frames to video/image outputs.

Supported formats:
    prores — ProRes 4444 with alpha (.mov)  [DEFAULT, CapCut-ready]
    webm   — VP9 with alpha (.webm)         [web / smaller file]
    png    — PNG frame sequence (folder)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

_VIDEOS_DIR = Path(__file__).resolve().parent.parent / "videos"
_FORMAT_EXT = {
    "prores": ".mov",
    "webm": ".webm",
    "png": "",  # folder, no extension
}


def resolve_output_path(
    input_path: Path,
    explicit_output: Optional[str],
    fmt: str,
    suffix: str = "_processed",
) -> Path:
    """
    Determine the final output path.

    Rules:
        - If *explicit_output* is provided: use it as-is.
        - Otherwise: <videos_dir>/<input_stem><suffix><ext>
    """
    if explicit_output:
        return Path(explicit_output).resolve()

    ext = _FORMAT_EXT.get(fmt, ".mov")
    out_name = f"{input_path.stem}{suffix}{ext}"
    return _VIDEOS_DIR / out_name


def compute_processing_size(width: int, height: int, max_short: int = 1080) -> Tuple[int, int]:
    """
    Downscale (width, height) so the shortest side ≤ *max_short*.
    Rounds to the nearest even number (codec requirement).
    Returns (new_w, new_h) — unchanged if already within limit.
    """
    short = min(width, height)
    if short <= max_short:
        return width, height
    scale = max_short / short
    new_w = int(round(width * scale / 2)) * 2
    new_h = int(round(height * scale / 2)) * 2
    return new_w, new_h


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, tmp_dir: str) -> Optional[str]:
    """
    Demux audio from *video_path* to a temp .aac file.
    Returns the path, or None if no audio stream is found.
    """
    audio_path = os.path.join(tmp_dir, "audio.aac")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "copy",
        audio_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if (
        result.returncode != 0
        or not os.path.exists(audio_path)
        or os.path.getsize(audio_path) == 0
    ):
        return None
    return audio_path


# ---------------------------------------------------------------------------
# FFmpeg pipe for video formats
# ---------------------------------------------------------------------------

def _ffmpeg_cmd_prores(width: int, height: int, fps: float, out_path: str) -> list:
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "prores_ks",
        "-profile:v", "4444",
        "-pix_fmt", "yuva444p10le",
        "-vendor", "apl0",
        out_path,
    ]


def _ffmpeg_cmd_webm(width: int, height: int, fps: float, out_path: str) -> list:
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-b:v", "0",
        "-crf", "30",
        "-auto-alt-ref", "0",  # required for VP9 alpha
        out_path,
    ]


_FORMAT_CMD = {
    "prores": _ffmpeg_cmd_prores,
    "webm": _ffmpeg_cmd_webm,
}


class VideoEncoder:
    """
    Context manager wrapping an FFmpeg stdin pipe for video encoding.

    Usage:
        with VideoEncoder("prores", w, h, fps, tmp_path) as enc:
            enc.write(rgba_uint8_frame)
    """

    def __init__(self, fmt: str, width: int, height: int, fps: float, out_path: str):
        if fmt not in _FORMAT_CMD:
            sys.exit(f"❌  Unknown video format: {fmt!r}. Expected: prores, webm")
        self.fmt = fmt
        self.out_path = out_path
        cmd = _FORMAT_CMD[fmt](width, height, fps, out_path)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, rgba_frame: np.ndarray) -> None:
        """Write one (H, W, 4) uint8 RGBA frame."""
        self._proc.stdin.write(rgba_frame.tobytes())

    def close(self) -> None:
        self._proc.stdin.close()
        self._proc.wait()
        if self._proc.returncode != 0:
            err = self._proc.stderr.read().decode(errors="replace")
            sys.exit(f"❌  FFmpeg encoding failed:\n{err}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# PNG sequence writer
# ---------------------------------------------------------------------------

class PngEncoder:
    """Write RGBA frames as numbered PNGs into an output folder."""

    def __init__(self, out_folder: Path):
        self.out_folder = out_folder
        out_folder.mkdir(parents=True, exist_ok=True)
        self._idx = 0

    def write(self, rgba_frame: np.ndarray) -> None:
        # OpenCV writes in BGRA order; convert RGBA→BGRA
        bgra = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGRA)
        path = self.out_folder / f"frame_{self._idx:05d}.png"
        cv2.imwrite(str(path), bgra)
        self._idx += 1

    def close(self) -> None:
        pass  # nothing to flush

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Audio remux
# ---------------------------------------------------------------------------

def remux_audio(video_path: str, audio_path: str, final_path: str) -> None:
    """Merge ProRes/WebM video with an extracted audio track."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        final_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")
        print(f"⚠  Audio remux failed:\n{err}")
        shutil.move(video_path, final_path)
