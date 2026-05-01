"""Tests for src/rfs_sampler.py.

Covers:
  - compute_repeat_factors() math matches the LVIS paper formula.
  - Per-image r_i is the MAX of its categories' r_c (not sum, not mean).
  - Frequent categories (f_c > t) get r_c = 1 exactly.
  - Pickling round-trip preserves the result.
  - RepeatFactorSampler stochastic rounding has the right expected count.
  - RepeatFactorSampler shuffle preserves multiplicity.
  - set_epoch() changes the shuffle seed deterministically.
  - Constructor validation rejects bad threshold.

Run from repo root:
    pytest tests/test_rfs_sampler.py -v
"""
from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from pathlib import Path

import pytest
import torch

from src.rfs_sampler import (
    RFSResult,
    compute_repeat_factors,
    load_repeat_factors,
    write_repeat_factors,
    RepeatFactorSampler,
)


def _write_coco(tmp_path: Path, images_to_categories: dict[int, list[int]]) -> Path:
    """Helper: build a COCO-format JSON from an image_id -> [category_ids] dict."""
    images = [
        {"id": iid, "file_name": f"img_{iid}.png", "height": 100, "width": 100}
        for iid in images_to_categories
    ]
    annotations = []
    next_id = 1
    for iid, cats in images_to_categories.items():
        for cid in cats:
            annotations.append({
                "id": next_id,
                "image_id": iid,
                "category_id": cid,
                "bbox": [0, 0, 10, 10],
                "area": 100,
            })
            next_id += 1
    all_cats = sorted({c for cats in images_to_categories.values() for c in cats})
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": c, "name": f"cat_{c}"} for c in all_cats],
    }
    out = tmp_path / "train.json"
    out.write_text(json.dumps(coco))
    return out


def test_compute_basic_three_categories(tmp_path):
    """Setup: 10 images, category 1 in all 10, category 2 in 5, category 3 in 1.

    With t = 0.5:
      f_1 = 1.0  -> r_1 = max(1, sqrt(0.5/1.0))  = 1.0
      f_2 = 0.5  -> r_2 = max(1, sqrt(0.5/0.5))  = 1.0
      f_3 = 0.1  -> r_3 = max(1, sqrt(0.5/0.1))  = sqrt(5)
    """
    images = {i: [1] for i in range(1, 11)}
    for i in range(1, 6):
        images[i].append(2)
    images[1].append(3)
    json_path = _write_coco(tmp_path, images)

    result = compute_repeat_factors(json_path, threshold=0.5, verbose=False)
    assert result.n_images == 10
    assert result.category_image_counts == {1: 10, 2: 5, 3: 1}
    assert result.category_image_frequencies[1] == pytest.approx(1.0)
    assert result.category_image_frequencies[2] == pytest.approx(0.5)
    assert result.category_image_frequencies[3] == pytest.approx(0.1)
    assert result.category_repeat_factors[1] == pytest.approx(1.0)
    assert result.category_repeat_factors[2] == pytest.approx(1.0)
    assert result.category_repeat_factors[3] == pytest.approx(math.sqrt(5.0))


def test_per_image_repeat_factor_is_max_over_categories(tmp_path):
    """Image 1 contains both cat 1 (r_c=1.0) and cat 3 (r_c=sqrt(5)).
    Per-image factor should be the MAX = sqrt(5), not sum or mean."""
    images = {i: [1] for i in range(1, 11)}
    for i in range(1, 6):
        images[i].append(2)
    images[1].append(3)
    json_path = _write_coco(tmp_path, images)
    result = compute_repeat_factors(json_path, threshold=0.5, verbose=False)
    assert result.image_repeat_factors[1] == pytest.approx(math.sqrt(5.0))
    assert result.image_repeat_factors[2] == pytest.approx(1.0)


def test_frequent_categories_get_r_c_one(tmp_path):
    images = {i: [1, 2, 3] for i in range(1, 21)}
    json_path = _write_coco(tmp_path, images)
    result = compute_repeat_factors(json_path, threshold=0.001, verbose=False)
    for cid in [1, 2, 3]:
        assert result.category_repeat_factors[cid] == 1.0


def test_pickle_round_trip(tmp_path):
    images = {i: [1, 2, 3] for i in range(1, 11)}
    json_path = _write_coco(tmp_path, images)
    result = compute_repeat_factors(json_path, threshold=0.1, verbose=False)
    out = tmp_path / "rfs.pkl"
    write_repeat_factors(result, out)
    loaded = load_repeat_factors(out)
    assert loaded.n_images == result.n_images
    assert loaded.category_repeat_factors == result.category_repeat_factors
    assert loaded.image_repeat_factors == result.image_repeat_factors


def test_invalid_threshold(tmp_path):
    images = {i: [1] for i in range(1, 5)}
    json_path = _write_coco(tmp_path, images)
    with pytest.raises(ValueError):
        compute_repeat_factors(json_path, threshold=0.0, verbose=False)
    with pytest.raises(ValueError):
        compute_repeat_factors(json_path, threshold=1.5, verbose=False)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_repeat_factors(tmp_path / "nope.json", verbose=False)


def test_sampler_expected_length_matches_factor_sum(tmp_path):
    """For a sampler with stochastic rounding, the EXPECTED epoch size should
    equal the sum of repeat factors (rounded)."""
    images = {i: [1] for i in range(1, 11)}
    images[1].append(3)
    json_path = _write_coco(tmp_path, images)
    result = compute_repeat_factors(json_path, threshold=0.5, verbose=False)

    image_ids_in_dataset_order = list(range(1, 11))
    sampler = RepeatFactorSampler(image_ids_in_dataset_order, result, shuffle=False, seed=0)

    expected_sum = sum(result.image_repeat_factors[iid] for iid in image_ids_in_dataset_order)
    assert len(sampler) == max(1, int(round(expected_sum)))


def test_sampler_stochastic_rounding_gives_unbiased_count(tmp_path):
    """Average epoch yields, over many epochs, should approximate r_i for each image."""
    images = {1: [1], 2: [2], 3: [3]}
    json_path = _write_coco(tmp_path, images)
    result = RFSResult(
        threshold=0.0,
        n_images=3,
        image_repeat_factors={1: 1.7, 2: 2.3, 3: 1.0},
    )
    sampler = RepeatFactorSampler([1, 2, 3], result, shuffle=False, seed=42)

    counts = Counter()
    n_epochs = 1000
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        for idx in sampler:
            counts[idx] += 1

    avg_idx_0 = counts[0] / n_epochs
    avg_idx_1 = counts[1] / n_epochs
    avg_idx_2 = counts[2] / n_epochs
    assert abs(avg_idx_0 - 1.7) < 0.1, f"avg yields for image 1 = {avg_idx_0}"
    assert abs(avg_idx_1 - 2.3) < 0.1, f"avg yields for image 2 = {avg_idx_1}"
    assert avg_idx_2 == pytest.approx(1.0, abs=0.01)


def test_sampler_shuffle_preserves_multiplicity():
    """Shuffling must NOT add or remove instances; just reorder them."""
    result = RFSResult(
        threshold=0.0,
        n_images=4,
        image_repeat_factors={1: 1.0, 2: 2.0, 3: 1.0, 4: 3.0},
    )
    sampler = RepeatFactorSampler([1, 2, 3, 4], result, shuffle=True, seed=0)
    sampler.set_epoch(0)
    yielded = list(sampler)
    assert Counter(yielded) == Counter({0: 1, 1: 2, 2: 1, 3: 3})


def test_sampler_set_epoch_changes_order():
    """Different epochs must produce different shuffles."""
    result = RFSResult(
        threshold=0.0,
        n_images=4,
        image_repeat_factors={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
    )
    sampler = RepeatFactorSampler([1, 2, 3, 4], result, shuffle=True, seed=0)
    sampler.set_epoch(0)
    yielded_e0 = list(sampler)
    sampler.set_epoch(1)
    yielded_e1 = list(sampler)
    assert yielded_e0 != yielded_e1, "set_epoch should change the shuffle order"
    assert Counter(yielded_e0) == Counter(yielded_e1)


def test_sampler_load_pickled_result(tmp_path):
    """Sampler should accept an RFSResult that was pickled then loaded."""
    images = {i: [1, 2] for i in range(1, 6)}
    json_path = _write_coco(tmp_path, images)
    result = compute_repeat_factors(json_path, threshold=0.05, verbose=False)
    out = tmp_path / "rfs.pkl"
    write_repeat_factors(result, out)
    loaded = load_repeat_factors(out)

    sampler = RepeatFactorSampler([1, 2, 3, 4, 5], loaded, shuffle=False, seed=0)
    sampler.set_epoch(0)
    indices = list(sampler)
    assert all(0 <= i < 5 for i in indices)
