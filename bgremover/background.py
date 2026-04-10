"""
bgremover/background.py
-----------------------
Background plate resolution logic.

Priority order (highest → lowest):
    1. --bg-image CLI flag (explicit user choice)
    2. Saved image in assets/ (user prompted to confirm)
    3. Auto-extract via median of first N frames
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_BG_SAMPLE_FRAMES = 30
_SUPPORTED_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_saved_bg(assets_dir: Path = _ASSETS_DIR) -> Optional[Path]:
    """
    Search *assets_dir* for a supported image file.
    Returns the first match, or None if the folder is empty / missing.
    """
    if not assets_dir.exists():
        return None
    for pattern in _SUPPORTED_EXTS:
        matches = sorted(assets_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_image(path: str) -> np.ndarray:
    """Load a BGR image from *path*, exit with a clear error if it fails."""
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"❌  Cannot read image: {path}")
    return img


def extract_from_video(
    video_path: str,
    n_frames: int = _BG_SAMPLE_FRAMES,
) -> np.ndarray:
    """
    Generate a background plate from the first *n_frames* of the video.

    The per-pixel **median** is used instead of the mean: it naturally
    ignores the subject (who occupies a *minority* of any pixel's values
    across the frame stack) and keeps the static background.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open video: {video_path}")

    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        sys.exit("❌  Could not read any frames from the input video.")

    print(f"✅  Background plate from {len(frames)} frames (median).")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def resolve(
    cli_bg_image: Optional[str],
    video_path: str,
    saved_bg_path: Optional[Path] = None,
    interactive: bool = True,
) -> Optional[np.ndarray]:
    """
    Resolve the background plate following the documented priority order.

    Args:
        cli_bg_image:   Path from the --bg-image CLI flag, or None.
        video_path:     Input video path (used for auto-extract fallback).
        saved_bg_path:  Override assets/ search (used in tests).
        interactive:    If True, prompt the user when a saved bg is found.

    Returns:
        (H, W, 3) uint8 BGR image, or None if using RVM (no plate needed).
    """
    # 1 — Explicit CLI flag
    if cli_bg_image:
        print(f"🖼  Using provided background image: {cli_bg_image}")
        return load_image(cli_bg_image)

    # 2 — Check assets/ for a saved background
    found = saved_bg_path if saved_bg_path is not None else find_saved_bg()
    if found:
        print(f"🖼  Found saved background image: {found.name}")
        if interactive:
            answer = input("   Use this background? [Y/n]: ").strip().lower()
        else:
            answer = "y"

        if answer in ("", "y", "yes"):
            print(f"   ✅  Using: {found.name}")
            return load_image(str(found))
        else:
            print("   Skipped — extracting from video frames instead.")

    # 3 — Auto-extract from first N frames
    print(f"⚙  No bg image supplied. Extracting from first {_BG_SAMPLE_FRAMES} frames …")
    return extract_from_video(video_path)
