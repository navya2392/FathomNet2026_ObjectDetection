"""Tests for predict_test_multiscale_tta CLI helpers."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_multiscale_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "predict_test_multiscale_tta.py"
    spec = importlib.util.spec_from_file_location("predict_test_multiscale_tta", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_predict_extra_args_combination():
    mod = _load_multiscale_module()
    ns = argparse.Namespace(
        half=True,
        tta=True,
        device="0",
        max_images=50,
        per_class_conf_json=None,
    )
    assert mod._predict_extra_args(ns) == [
        "--half", "--tta", "--device", "0", "--max-images", "50",
    ]


def test_predict_extra_args_empty():
    mod = _load_multiscale_module()
    ns = argparse.Namespace(
        half=False,
        tta=False,
        device="",
        max_images=None,
        per_class_conf_json=None,
    )
    assert mod._predict_extra_args(ns) == []


def test_predict_extra_args_with_per_class_conf(tmp_path):
    mod = _load_multiscale_module()
    js = tmp_path / "pc.json"
    js.write_text("{}", encoding="utf-8")
    ns = argparse.Namespace(
        half=False,
        tta=False,
        device="",
        max_images=None,
        per_class_conf_json=js,
    )
    assert mod._predict_extra_args(ns) == ["--per-class-conf-json", str(js)]
