"""Tests for bgremover.encoder (path helpers, size computation)"""

from pathlib import Path

import numpy as np
import pytest

from bgremover.encoder import compute_processing_size, resolve_output_path


# ---------------------------------------------------------------------------
# compute_processing_size()
# ---------------------------------------------------------------------------

class TestComputeProcessingSize:
    def test_no_change_when_within_limit(self):
        assert compute_processing_size(1080, 1920, 1080) == (1080, 1920)

    def test_no_change_when_equal_to_limit(self):
        assert compute_processing_size(1080, 1920, 1920) == (1080, 1920)

    def test_scales_portrait_video(self):
        # 1080×1920 portrait, max_short=540
        w, h = compute_processing_size(1080, 1920, 540)
        assert min(w, h) == 540
        assert w % 2 == 0
        assert h % 2 == 0

    def test_scales_landscape_video(self):
        # 3840×2160 (4K landscape), max_short=1080
        w, h = compute_processing_size(3840, 2160, 1080)
        assert min(w, h) == 1080

    def test_output_always_even(self):
        # Ensure no odd dimensions (codecs require even)
        w, h = compute_processing_size(1001, 1777, 540)
        assert w % 2 == 0
        assert h % 2 == 0

    def test_no_upscale(self):
        # Small video stays small — no upscaling
        w, h = compute_processing_size(480, 270, 1080)
        assert w == 480
        assert h == 270

    def test_square_video(self):
        w, h = compute_processing_size(2160, 2160, 1080)
        assert w == h == 1080


# ---------------------------------------------------------------------------
# resolve_output_path()
# ---------------------------------------------------------------------------

class TestResolveOutputPath:
    def _input(self, name="selfie.mov") -> Path:
        return Path(f"/fake/path/{name}")

    def test_explicit_output_used_as_is(self, tmp_path):
        out = resolve_output_path(
            self._input(), explicit_output=str(tmp_path / "custom.mov"), fmt="prores"
        )
        assert out == tmp_path / "custom.mov"

    def test_auto_name_prores(self):
        out = resolve_output_path(self._input("clip.mov"), explicit_output=None, fmt="prores")
        assert out.name == "clip_processed.mov"

    def test_auto_name_webm(self):
        out = resolve_output_path(self._input("clip.mov"), explicit_output=None, fmt="webm")
        assert out.name == "clip_processed.webm"

    def test_auto_name_png_no_extension(self):
        out = resolve_output_path(self._input("clip.mov"), explicit_output=None, fmt="png")
        assert out.name == "clip_processed"

    def test_custom_suffix(self):
        out = resolve_output_path(
            self._input("clip.mov"), explicit_output=None, fmt="prores", suffix="_alpha"
        )
        assert "_alpha" in out.name

    def test_output_lands_in_videos_dir(self):
        out = resolve_output_path(self._input("clip.mov"), explicit_output=None, fmt="prores")
        # Should be inside the project's videos/ folder
        assert "videos" in str(out)

    def test_stem_preserved(self):
        out = resolve_output_path(self._input("my_video.mov"), explicit_output=None, fmt="prores")
        assert out.stem.startswith("my_video")
