"""Tests for bgremover.pipeline (tensor helpers, RGBA math, downsample ratio)"""

import numpy as np
import pytest
import torch

from bgremover.pipeline import (
    compose_rgba,
    compute_downsample_ratio,
    frame_to_tensor,
)


# ---------------------------------------------------------------------------
# frame_to_tensor()
# ---------------------------------------------------------------------------

class TestFrameToTensor:
    def _bgr_frame(self, h=16, w=16):
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def test_output_shape(self):
        frame = self._bgr_frame(32, 64)
        t = frame_to_tensor(frame, torch.device("cpu"))
        assert t.shape == (1, 3, 32, 64)  # (B, C, H, W)

    def test_output_dtype_float32(self):
        t = frame_to_tensor(self._bgr_frame(), torch.device("cpu"))
        assert t.dtype == torch.float32

    def test_range_0_to_1(self):
        t = frame_to_tensor(self._bgr_frame(), torch.device("cpu"))
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_black_frame_is_zero(self):
        black = np.zeros((8, 8, 3), dtype=np.uint8)
        t = frame_to_tensor(black, torch.device("cpu"))
        assert t.sum().item() == 0.0

    def test_white_frame_is_one(self):
        white = np.full((8, 8, 3), 255, dtype=np.uint8)
        t = frame_to_tensor(white, torch.device("cpu"))
        assert torch.allclose(t, torch.ones_like(t))

    def test_bgr_to_rgb_conversion(self):
        """A pure-red BGR frame (0,0,255) should map to R=1 G=0 B=0 in the tensor."""
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :] = [0, 0, 255]  # BGR red
        t = frame_to_tensor(frame, torch.device("cpu"))
        assert torch.allclose(t[0, 0], torch.ones(4, 4))   # R channel = 1
        assert t[0, 1].sum() == 0.0                          # G channel = 0
        assert t[0, 2].sum() == 0.0                          # B channel = 0


# ---------------------------------------------------------------------------
# compose_rgba()
# ---------------------------------------------------------------------------

class TestComposeRgba:
    def _make_tensors(self, h=8, w=8, alpha_val=1.0, fg_val=0.5):
        """Create synthetic (1,3,H,W) fgr and (1,1,H,W) pha tensors."""
        fgr = torch.full((1, 3, h, w), fg_val)
        pha = torch.full((1, 1, h, w), alpha_val)
        return fgr, pha

    def test_output_shape(self):
        fgr, pha = self._make_tensors(16, 32)
        rgba = compose_rgba(fgr, pha)
        assert rgba.shape == (16, 32, 4)

    def test_output_dtype_uint8(self):
        fgr, pha = self._make_tensors()
        rgba = compose_rgba(fgr, pha)
        assert rgba.dtype == np.uint8

    def test_fully_opaque_alpha(self):
        fgr, pha = self._make_tensors(alpha_val=1.0, fg_val=0.5)
        rgba = compose_rgba(fgr, pha)
        # Alpha channel (index 3) should be 255
        assert rgba[:, :, 3].min() == 255

    def test_fully_transparent_alpha(self):
        fgr, pha = self._make_tensors(alpha_val=0.0, fg_val=0.8)
        rgba = compose_rgba(fgr, pha)
        # Alpha channel should be 0
        assert rgba[:, :, 3].max() == 0
        # RGB should also be 0 (pre-multiplied)
        assert rgba[:, :, :3].max() == 0

    def test_premultiply_math(self):
        """R = fg * alpha. With fg=0.8, alpha=0.5 → R ≈ 102 (0.4 * 255)."""
        fgr, pha = self._make_tensors(alpha_val=0.5, fg_val=0.8)
        rgba = compose_rgba(fgr, pha)
        expected = int(0.4 * 255)
        # Allow ±1 for rounding
        assert abs(int(rgba[0, 0, 0]) - expected) <= 1

    def test_clipped_to_255(self):
        """Values should never exceed 255."""
        fgr = torch.full((1, 3, 4, 4), 1.0)
        pha = torch.full((1, 1, 4, 4), 1.0)
        rgba = compose_rgba(fgr, pha)
        assert rgba.max() <= 255


# ---------------------------------------------------------------------------
# compute_downsample_ratio()
# ---------------------------------------------------------------------------

class TestComputeDownsampleRatio:
    def test_hd_portrait_gives_reasonable_ratio(self):
        # 1080×1920 portrait, target 512
        ratio = compute_downsample_ratio(1080, 1920, target=512)
        assert 0.0 < ratio <= 1.0
        # After downsampling, shorter side should be ≤ 512
        assert int(1080 * ratio) <= 512

    def test_small_video_clamped_to_1(self):
        # 320×240, target 512 → shorter side already < target → ratio=1.0
        ratio = compute_downsample_ratio(320, 240, target=512)
        assert ratio == 1.0

    def test_ratio_always_positive(self):
        assert compute_downsample_ratio(100, 100) > 0

    def test_4k_gives_ratio_below_half(self):
        # 4K landscape 3840×2160, target 512
        ratio = compute_downsample_ratio(3840, 2160, target=512)
        assert ratio < 0.5
