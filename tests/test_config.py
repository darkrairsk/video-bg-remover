"""Tests for bgremover.config"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import bgremover.config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config_file(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

class TestLoad:
    def test_returns_all_defaults_when_no_file(self, tmp_path):
        with patch.object(cfg, "CONFIG_PATH", tmp_path / "config.json"):
            result = cfg.load()
        assert result == cfg.DEFAULTS

    def test_overrides_known_key_from_file(self, tmp_path):
        _make_config_file(tmp_path, {"model_type": "bgmv2"})
        with patch.object(cfg, "CONFIG_PATH", tmp_path / "config.json"):
            result = cfg.load()
        assert result["model_type"] == "bgmv2"
        # Other keys remain default
        assert result["format"] == cfg.DEFAULTS["format"]

    def test_ignores_unknown_keys_in_file(self, tmp_path):
        _make_config_file(tmp_path, {"unknown_key": "should_be_ignored"})
        with patch.object(cfg, "CONFIG_PATH", tmp_path / "config.json"):
            result = cfg.load()
        assert "unknown_key" not in result

    def test_falls_back_to_defaults_on_corrupt_json(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text("{invalid json")
        with patch.object(cfg, "CONFIG_PATH", p):
            result = cfg.load()
        assert result == cfg.DEFAULTS

    def test_all_defaults_present(self):
        assert set(cfg.DEFAULTS.keys()) == {
            "model_type", "format", "max_short_side",
            "output_suffix", "default_bg_image", "device",
        }


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

class TestSave:
    def test_saves_valid_key(self, tmp_path):
        with patch.object(cfg, "CONFIG_PATH", tmp_path / "config.json"):
            cfg.save({"model_type": "bgmv2"})
            with open(tmp_path / "config.json") as fh:
                data = json.load(fh)
        assert data["model_type"] == "bgmv2"

    def test_does_not_save_unknown_keys(self, tmp_path):
        with patch.object(cfg, "CONFIG_PATH", tmp_path / "config.json"):
            cfg.save({"unknown": "value"})
            with open(tmp_path / "config.json") as fh:
                data = json.load(fh)
        assert "unknown" not in data

    def test_merges_with_existing_file(self, tmp_path):
        _make_config_file(tmp_path, {"model_type": "bgmv2"})
        with patch.object(cfg, "CONFIG_PATH", tmp_path / "config.json"):
            cfg.save({"format": "webm"})
            with open(tmp_path / "config.json") as fh:
                data = json.load(fh)
        assert data["model_type"] == "bgmv2"
        assert data["format"] == "webm"

    def test_save_and_reload_roundtrip(self, tmp_path):
        with patch.object(cfg, "CONFIG_PATH", tmp_path / "config.json"):
            cfg.save({"model_type": "bgmv2", "format": "webm"})
            result = cfg.load()
        assert result["model_type"] == "bgmv2"
        assert result["format"] == "webm"


# ---------------------------------------------------------------------------
# merge_args()
# ---------------------------------------------------------------------------

class TestMergeArgs:
    def test_cli_arg_overrides_config(self):
        base = dict(cfg.DEFAULTS)
        base["model_type"] = "rvm"
        args = SimpleNamespace(model_type="bgmv2", format=None, bg_image=None, device=None)
        result = cfg.merge_args(base, args)
        assert result["model_type"] == "bgmv2"

    def test_none_cli_arg_does_not_override(self):
        base = dict(cfg.DEFAULTS)
        base["model_type"] = "bgmv2"
        args = SimpleNamespace(model_type=None, format=None, bg_image=None, device=None)
        result = cfg.merge_args(base, args)
        assert result["model_type"] == "bgmv2"

    def test_bg_image_maps_to_default_bg_image(self):
        base = dict(cfg.DEFAULTS)
        args = SimpleNamespace(model_type=None, format=None, bg_image="assets/wall.jpg", device=None)
        result = cfg.merge_args(base, args)
        assert result["default_bg_image"] == "assets/wall.jpg"

    def test_does_not_mutate_original_config(self):
        base = dict(cfg.DEFAULTS)
        original = dict(base)
        args = SimpleNamespace(model_type="bgmv2", format=None, bg_image=None, device=None)
        cfg.merge_args(base, args)
        assert base == original


# ---------------------------------------------------------------------------
# show()
# ---------------------------------------------------------------------------

class TestShow:
    def test_show_contains_all_keys(self):
        output = cfg.show(cfg.DEFAULTS)
        for key in cfg.DEFAULTS:
            assert key in output

    def test_show_marks_defaults(self):
        output = cfg.show(cfg.DEFAULTS)
        assert "(default)" in output
