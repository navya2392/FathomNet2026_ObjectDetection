"""Unit tests for src/inference_paths.py (Tier 1A deterministic image listing)."""
from __future__ import annotations

from pathlib import Path

from src.inference_paths import capped_sorted_test_paths, sorted_test_image_paths


def test_sorted_order(tmp_path: Path):
    (tmp_path / "z.jpg").write_bytes(b"a")
    (tmp_path / "a.png").write_bytes(b"b")
    (tmp_path / "skip.txt").write_text("no")
    names = [p.name for p in sorted_test_image_paths(tmp_path)]
    assert names == ["a.png", "z.jpg"]


def test_cap_none_returns_all(tmp_path: Path):
    for name in ("b.jpg", "a.jpg"):
        (tmp_path / name).write_bytes(b"x")
    out = capped_sorted_test_paths(tmp_path, None)
    assert [p.name for p in out] == ["a.jpg", "b.jpg"]


def test_cap_slices(tmp_path: Path):
    for i in range(5):
        (tmp_path / f"{i}.jpg").write_bytes(b"x")
    out = capped_sorted_test_paths(tmp_path, 3)
    assert len(out) == 3
    assert out[0].name == "0.jpg"
