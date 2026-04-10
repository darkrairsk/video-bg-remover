"""
bgremover/models.py
-------------------
Download, load and configure AI matting models.

Supported model types:
    rvm   — Robust Video Matting (MobileNetV3, TorchScript)  [DEFAULT]
    bgmv2 — Background Matting V2 (MobileNetV2, TorchScript)
"""

import sys
import urllib.request
from pathlib import Path
from typing import Literal, Tuple

import torch

ModelType = Literal["rvm", "bgmv2"]

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "rvm": {
        "filename": "rvm_mobilenetv3_fp32.torchscript",
        "url": (
            "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/"
            "rvm_mobilenetv3_fp32.torchscript"
        ),
        "display_name": "RVM MobileNetV3",
    },
    "bgmv2": {
        "filename": "torchscript_mobilenetv2_fp32.pth",
        "url": (
            "https://github.com/PeterL1n/BackgroundMattingV2/releases/download/v1.0.0/"
            "torchscript_mobilenetv2_fp32.pth"
        ),
        "display_name": "BGMv2 MobileNetV2",
    },
}

# Default model storage: project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def select_device(preference: str = "auto") -> torch.device:
    """
    Resolve the best available torch device.

    Args:
        preference: 'auto' | 'mps' | 'cuda' | 'cpu'
    """
    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preference)


def get_model_path(model_type: ModelType, custom_path: str = None) -> Path:
    """Return resolved path to the model file (does not download)."""
    if custom_path:
        return Path(custom_path).resolve()
    info = MODELS[model_type]
    return _PROJECT_ROOT / info["filename"]


def download_if_needed(model_type: ModelType, model_path: Path) -> Path:
    """Download the model if not already present. Returns path."""
    if model_path.exists():
        return model_path

    info = MODELS[model_type]
    print(f"⏬  Downloading {info['display_name']} model → {model_path.name} …")
    try:
        urllib.request.urlretrieve(info["url"], str(model_path))
    except Exception as exc:
        sys.exit(f"❌  Download failed: {exc}\n    URL: {info['url']}")
    print("✅  Model downloaded.")
    return model_path


def load_rvm(model_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    """Load the RVM TorchScript model and move it to *device*."""
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model


def load_bgmv2(model_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    """Load the BGMv2 TorchScript model, configure and move it to *device*."""
    model = torch.jit.load(str(model_path), map_location=device)
    # Resolution-tuned settings for HD selfie footage (1080p)
    model.backbone_scale = 0.25
    model.refine_mode = "sampling"
    model.refine_sample_pixels = 80_000
    model.eval()
    return model


def load(
    model_type: ModelType,
    device: torch.device,
    custom_path: str = None,
) -> Tuple[torch.jit.ScriptModule, str]:
    """
    Download (if needed) and load a matting model.

    Returns:
        (model, display_name)
    """
    model_path = get_model_path(model_type, custom_path)
    download_if_needed(model_type, model_path)

    if model_type == "rvm":
        model = load_rvm(model_path, device)
    else:
        model = load_bgmv2(model_path, device)

    display_name = MODELS[model_type]["display_name"]
    return model, display_name
