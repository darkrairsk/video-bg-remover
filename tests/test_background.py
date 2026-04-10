"""Tests for bgremover.background"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import bgremover.background as bg_module


# ---------------------------------------------------------------------------
# find_saved_bg()
# ---------------------------------------------------------------------------

class TestFindSavedBg:
    def test_returns_none_when_assets_missing(self, tmp_path):
        result = bg_module.find_saved_bg(assets_dir=tmp_path / "nonexistent")
        assert result is None

    def test_returns_none_when_assets_empty(self, tmp_path):
        (tmp_path / "assets").mkdir()
        result = bg_module.find_saved_bg(assets_dir=tmp_path / "assets")
        assert result is None

    def test_finds_jpg_file(self, tmp_path):
        assets = tmp_path / "assets"
        assets.mkdir()
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = assets / "bg.jpg"
        cv2.imwrite(str(out), img)
        result = bg_module.find_saved_bg(assets_dir=assets)
        assert result == out

    def test_finds_png_file(self, tmp_path):
        assets = tmp_path / "assets"
        assets.mkdir()
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = assets / "bg.png"
        cv2.imwrite(str(out), img)
        result = bg_module.find_saved_bg(assets_dir=assets)
        assert result == out

    def test_prefers_jpg_over_png_due_to_sort_order(self, tmp_path):
        """jpg comes before png in pattern iteration; sorted() is stable."""
        assets = tmp_path / "assets"
        assets.mkdir()
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        jpg_file = assets / "aaa.jpg"
        png_file = assets / "zzz.png"
        cv2.imwrite(str(jpg_file), img)
        cv2.imwrite(str(png_file), img)
        result = bg_module.find_saved_bg(assets_dir=assets)
        # First pattern checked is *.jpg so jpg wins
        assert result.suffix == ".jpg"


# ---------------------------------------------------------------------------
# extract_from_video()   (mocked – no real video file needed)
# ---------------------------------------------------------------------------

class TestExtractFromVideo:
    def _make_fake_cap(self, frames, opened=True):
        cap = MagicMock()
        cap.isOpened.return_value = opened
        # Each read() call returns (True, frame) until exhausted
        reads = [(True, f) for f in frames] + [(False, None)]
        cap.read.side_effect = reads
        return cap

    def test_returns_correct_shape(self, tmp_path):
        h, w = 8, 8
        frames = [np.full((h, w, 3), i * 10, dtype=np.uint8) for i in range(5)]
        cap = self._make_fake_cap(frames)
        with patch("cv2.VideoCapture", return_value=cap):
            result = bg_module.extract_from_video("fake.mov", n_frames=5)
        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8

    def test_median_removes_subject(self, tmp_path):
        """
        Background pixels are 100; subject pixels are 200 in 2/5 frames.
        Median of [100,100,100,200,200] = 100 → background wins.
        """
        h, w = 4, 4
        bg_val = 100
        frames = [np.full((h, w, 3), bg_val, dtype=np.uint8) for _ in range(5)]
        # Simulate subject in frames 3 and 4 at a specific pixel
        frames[3][0, 0] = 200
        frames[4][0, 0] = 200
        cap = self._make_fake_cap(frames)
        with patch("cv2.VideoCapture", return_value=cap):
            result = bg_module.extract_from_video("fake.mov", n_frames=5)
        assert result[0, 0, 0] == bg_val  # subject pixel resolved to bg


# ---------------------------------------------------------------------------
# resolve()  — priority chain
# ---------------------------------------------------------------------------

class TestResolve:
    def _dummy_image(self) -> np.ndarray:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def test_priority1_cli_flag_wins(self, tmp_path):
        """CLI --bg-image takes highest priority."""
        img = self._dummy_image()
        cv2.imwrite(str(tmp_path / "cli.jpg"), img)
        with patch.object(bg_module, "load_image", return_value=img) as mock_load:
            result = bg_module.resolve(
                cli_bg_image=str(tmp_path / "cli.jpg"),
                video_path="irrelevant.mov",
                saved_bg_path=None,
                interactive=False,
            )
        mock_load.assert_called_once()
        assert result is img

    def test_priority2_saved_bg_used_when_confirmed(self, tmp_path):
        """Saved bg in assets/ is used when user says Y."""
        img = self._dummy_image()
        saved = tmp_path / "bg.jpg"
        cv2.imwrite(str(saved), img)
        with patch.object(bg_module, "load_image", return_value=img):
            result = bg_module.resolve(
                cli_bg_image=None,
                video_path="irrelevant.mov",
                saved_bg_path=saved,
                interactive=False,  # non-interactive → auto-accepts
            )
        assert result is img

    def test_priority3_auto_extract_when_no_saved_bg(self, tmp_path):
        """Falls through to auto-extract when no CLI flag and no saved bg."""
        dummy = self._dummy_image()
        # Also mock find_saved_bg so the real assets/ folder doesn't interfere
        with (
            patch.object(bg_module, "find_saved_bg", return_value=None),
            patch.object(bg_module, "extract_from_video", return_value=dummy) as mock_ext,
        ):
            result = bg_module.resolve(
                cli_bg_image=None,
                video_path="fake.mov",
                saved_bg_path=None,
                interactive=False,
            )
        mock_ext.assert_called_once()
        assert result is dummy

    def test_saved_bg_declined_falls_to_auto_extract(self, tmp_path):
        """If user declines the saved bg, auto-extract is used."""
        img = self._dummy_image()
        saved = tmp_path / "bg.jpg"
        cv2.imwrite(str(saved), img)
        dummy = np.ones((10, 10, 3), dtype=np.uint8) * 5
        with (
            patch.object(bg_module, "load_image", return_value=img),
            patch("builtins.input", return_value="n"),
            patch.object(bg_module, "extract_from_video", return_value=dummy) as mock_ext,
        ):
            result = bg_module.resolve(
                cli_bg_image=None,
                video_path="fake.mov",
                saved_bg_path=saved,
                interactive=True,
            )
        mock_ext.assert_called_once()
        assert result is dummy
