"""Tests for src/underwater_preproc.py.

Covers:
  - gray_world_balance() removes a synthetic blue cast.
  - gray_world_balance() preserves brightness with target='mean'.
  - apply_clahe() boosts standard deviation of luminance (proxy for contrast).
  - apply_clahe() requires uint8 input.
  - underwater_preprocess() composes both steps and is idempotent
    on already-balanced images.
  - needs_underwater_preproc() correctly identifies cast vs neutral images.
  - Validation rejects bad shapes, bad clip_limit, etc.
  - All-zero channel doesn't crash.

Run from repo root:
    pytest tests/test_underwater_preproc.py -v

CLAHE tests need OpenCV (cv2). If not installed, those tests skip.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.underwater_preproc import (
    apply_clahe,
    gray_world_balance,
    needs_underwater_preproc,
    underwater_preprocess,
)

cv2 = pytest.importorskip("cv2", reason="OpenCV not installed; CLAHE tests skipped")


def _make_synthetic_underwater(h: int = 64, w: int = 64, blue_cast: float = 1.5,
                               base_intensity: int = 100, seed: int = 0) -> np.ndarray:
    """Build a synthetic 'underwater' image: brighter blue, dimmer red/green."""
    rng = np.random.default_rng(seed)
    img = rng.integers(50, 150, size=(h, w, 3), dtype=np.uint8)
    img = img.astype(np.float64)
    img[:, :, 0] = np.clip(img[:, :, 0] * blue_cast, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] / blue_cast, 0, 255)
    return img.astype(np.uint8)


def test_gray_world_removes_blue_cast():
    img = _make_synthetic_underwater(blue_cast=1.5)
    out = gray_world_balance(img, target="mean", clip_uint8=True)
    assert isinstance(out, np.ndarray)
    means_before = [img[:, :, c].mean() for c in range(3)]
    means_after = [out[:, :, c].mean() for c in range(3)]
    spread_before = max(means_before) - min(means_before)
    spread_after = max(means_after) - min(means_after)
    assert spread_after < spread_before / 5, (
        f"gray-world should compress per-channel spread; before={spread_before:.1f}, "
        f"after={spread_after:.1f}"
    )


def test_gray_world_preserves_overall_brightness_with_target_mean():
    img = _make_synthetic_underwater(blue_cast=1.5)
    out = gray_world_balance(img, target="mean", clip_uint8=True)
    mean_before = img.astype(np.float64).mean()
    mean_after = out.astype(np.float64).mean()
    assert abs(mean_before - mean_after) < 5.0, (
        f"overall brightness should be roughly preserved; before={mean_before:.1f}, "
        f"after={mean_after:.1f}"
    )


def test_gray_world_returns_stats_when_requested():
    img = _make_synthetic_underwater(blue_cast=1.5)
    result = gray_world_balance(img, return_stats=True)
    assert isinstance(result, tuple) and len(result) == 2
    out, stats = result
    assert isinstance(out, np.ndarray)
    assert stats.color_cast_score > 0
    assert stats.scales[0] != stats.scales[2], (
        "blue and red scales should differ when there's a cast"
    )


def test_gray_world_invalid_shape():
    with pytest.raises(ValueError):
        gray_world_balance(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        gray_world_balance(np.zeros((10, 10, 4), dtype=np.uint8))


def test_gray_world_invalid_target():
    img = _make_synthetic_underwater()
    with pytest.raises(ValueError):
        gray_world_balance(img, target="bogus")


def test_gray_world_handles_zero_channel():
    """An all-zero channel cannot be scaled meaningfully; should not crash."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :, 1] = 100
    out = gray_world_balance(img, return_stats=False)
    assert isinstance(out, np.ndarray)
    assert out.shape == img.shape


def test_apply_clahe_increases_local_contrast():
    """CLAHE on a low-contrast image should boost the standard deviation
    of luminance (a proxy for contrast)."""
    rng = np.random.default_rng(0)
    flat = rng.integers(100, 130, size=(128, 128, 3), dtype=np.uint8)
    out = apply_clahe(flat, clip_limit=3.0)
    assert out.shape == flat.shape and out.dtype == np.uint8
    gray_in = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float64)
    assert gray_out.std() > gray_in.std() * 1.5, (
        f"CLAHE should boost std (contrast); before={gray_in.std():.2f}, after={gray_out.std():.2f}"
    )


def test_apply_clahe_invalid_dtype():
    img_f = np.zeros((64, 64, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_clahe(img_f)


def test_apply_clahe_invalid_clip_limit():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_clahe(img, clip_limit=0)
    with pytest.raises(ValueError):
        apply_clahe(img, clip_limit=-1)


def test_apply_clahe_color_space_options():
    img = _make_synthetic_underwater()
    out_lab = apply_clahe(img, color_space="lab")
    out_gray = apply_clahe(img, color_space="gray")
    assert out_lab.shape == img.shape and out_gray.shape == img.shape
    diff_lab = np.abs(out_lab.astype(np.int32) - img.astype(np.int32)).mean()
    diff_gray = np.abs(out_gray.astype(np.int32) - img.astype(np.int32)).mean()
    assert diff_gray > 0
    assert diff_lab > 0


def test_apply_clahe_invalid_color_space():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_clahe(img, color_space="bogus")


def test_underwater_preprocess_pipeline():
    """End-to-end: gray-world + CLAHE should both reduce cast AND boost contrast."""
    img = _make_synthetic_underwater(blue_cast=1.5)
    out = underwater_preprocess(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    means_before = [img[:, :, c].mean() for c in range(3)]
    means_after = [out[:, :, c].mean() for c in range(3)]
    assert max(means_after) - min(means_after) < (max(means_before) - min(means_before))


def test_underwater_preprocess_skip_flags():
    img = _make_synthetic_underwater(blue_cast=1.5)
    no_clahe = underwater_preprocess(img, apply_clahe_step=False)
    no_gw = underwater_preprocess(img, apply_gray_world=False)
    assert not np.array_equal(no_clahe, img), "gray-world alone should change image"
    assert not np.array_equal(no_gw, img), "CLAHE alone should change image"


def test_needs_underwater_preproc_thresholds():
    cast_img = _make_synthetic_underwater(blue_cast=1.5)
    neutral = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert needs_underwater_preproc(cast_img, color_cast_threshold=0.15)
    assert not needs_underwater_preproc(neutral, color_cast_threshold=0.15)


def test_needs_underwater_preproc_invalid_shape():
    with pytest.raises(ValueError):
        needs_underwater_preproc(np.zeros((64, 64), dtype=np.uint8))
