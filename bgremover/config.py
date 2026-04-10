"""
bgremover/config.py
-------------------
Read, write and merge config.json preferences.

Priority (highest → lowest):
    1. Explicit CLI argument
    2. config.json saved value
    3. Built-in default
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

# Config lives in the project root alongside remove_bg.py
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

DEFAULTS: Dict[str, Any] = {
    "model_type": "rvm",            # rvm | bgmv2
    "format": "prores",             # prores | webm | png
    "max_short_side": 1080,         # downscale if video shorter side > this
    "output_suffix": "_processed",  # appended to input stem for auto-naming
    "default_bg_image": None,       # path string or null
    "device": "auto",               # auto | mps | cuda | cpu
}


def load() -> Dict[str, Any]:
    """Return merged config: defaults overridden by config.json values."""
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            # Only accept keys that exist in DEFAULTS to avoid pollution
            for key in DEFAULTS:
                if key in saved:
                    cfg[key] = saved[key]
        except (json.JSONDecodeError, OSError):
            pass  # silently fall back to defaults if file is corrupt
    return cfg


def save(overrides: Dict[str, Any]) -> None:
    """Persist *overrides* into config.json (merges with existing saved values)."""
    existing: Dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing.update({k: v for k, v in overrides.items() if k in DEFAULTS})
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)


def merge_args(cfg: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    Overlay parsed argparse namespace *args* on top of config dict *cfg*.
    Only non-None / explicitly set CLI values win over the config.
    Returns a new merged dict.
    """
    merged = dict(cfg)
    arg_map = {
        "model_type": "model_type",
        "format": "format",
        "bg_image": "default_bg_image",
        "device": "device",
    }
    for arg_attr, cfg_key in arg_map.items():
        val = getattr(args, arg_attr, None)
        if val is not None:
            merged[cfg_key] = val
    return merged


def show(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Return a human-readable string of the current config."""
    if cfg is None:
        cfg = load()
    lines = ["Current configuration (config.json + defaults):"]
    for key, val in cfg.items():
        source = "(default)" if val == DEFAULTS.get(key) else "(saved)"
        lines.append(f"  {key:<20} = {val!r:30}  {source}")
    return "\n".join(lines)
