"""Tests for src/copy_paste.py.

Covers:
  - build_default_schedule() puts each class in the correct tier.
  - InstanceBank correctly groups crops by class.
  - class_aware_copy_paste() respects per-class probabilities deterministically
    when seeded.
  - Pasted boxes are added to labels and overlap policy is honored.
  - Pasting class with empty bank is silently skipped.
  - Pasting fails gracefully when image is too crowded.
  - Pickling round-trip preserves the bank.

Run from repo root:
    pytest tests/test_copy_paste.py -v
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from src.copy_paste import (
    CropEntry,
    DEFAULT_TIER_PROBABILITIES,
    InstanceBank,
    build_default_schedule,
    build_instance_bank,
    class_aware_copy_paste,
    load_bank,
    write_bank,
)


def test_default_schedule_tiers():
    counts = {0: 5, 1: 50, 2: 500, 3: 5000}
    sched = build_default_schedule(counts)
    assert sched[0] == DEFAULT_TIER_PROBABILITIES["critical"]
    assert sched[1] == DEFAULT_TIER_PROBABILITIES["rare"]
    assert sched[2] == DEFAULT_TIER_PROBABILITIES["mid"]
    assert sched[3] == DEFAULT_TIER_PROBABILITIES["common"]


def test_default_schedule_with_overrides():
    overrides = {"critical": 1.0, "rare": 0.5, "mid": 0.1, "common": 0.0}
    counts = {0: 10, 1: 75, 2: 500, 3: 9999}
    sched = build_default_schedule(counts, tier_probabilities=overrides)
    assert sched[0] == 1.0
    assert sched[1] == 0.5
    assert sched[2] == 0.1
    assert sched[3] == 0.0


def test_instance_bank_summary_and_n_crops():
    bank = InstanceBank()
    for _ in range(3):
        bank.add(CropEntry("/x.png", (0, 0, 10, 10), 0))
    bank.add(CropEntry("/y.png", (5, 5, 20, 20), 1))
    assert bank.n_crops_for(0) == 3
    assert bank.n_crops_for(1) == 1
    assert bank.n_crops_for(99) == 0
    assert bank.total() == 4
    summary = bank.summary()
    assert "Total crops:    4" in summary


def test_build_instance_bank_round_trip(tmp_path):
    """Verify build_instance_bank reads a synthetic COCO JSON correctly."""
    images = [
        {"id": 1, "file_name": "a.png", "height": 100, "width": 100},
        {"id": 2, "file_name": "b.png", "height": 100, "width": 100},
    ]
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 5, "bbox": [10, 10, 20, 20], "area": 400},
        {"id": 2, "image_id": 1, "category_id": 5, "bbox": [40, 40, 30, 30], "area": 900},
        {"id": 3, "image_id": 2, "category_id": 7, "bbox": [10, 10, 5, 5], "area": 25},
        {"id": 4, "image_id": 2, "category_id": 7, "bbox": [50, 50, 25, 25], "area": 625},
    ]
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 5, "name": "five"}, {"id": 7, "name": "seven"}],
    }
    json_path = tmp_path / "tiny.json"
    json_path.write_text(json.dumps(coco))

    bank = build_instance_bank(
        json_path,
        image_root=tmp_path,
        cat_id_to_idx={5: 0, 7: 1},
        min_box_pixels=16,
        verbose=False,
    )
    assert bank.n_crops_for(0) == 2
    assert bank.n_crops_for(1) == 1


def _make_image_loader(images: dict[str, np.ndarray]):
    def loader(path: str) -> np.ndarray:
        return images[path]
    return loader


def test_paste_actually_modifies_image_when_prob_one():
    rng = random.Random(0)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    crop_img = np.full((10, 10, 3), 255, dtype=np.uint8)

    bank = InstanceBank()
    bank.add(CropEntry("/fake.png", (0, 0, 10, 10), 0))

    src = np.zeros((10, 10, 3), dtype=np.uint8)
    src[:] = crop_img
    loader = _make_image_loader({"/fake.png": src})

    schedule = {0: 1.0}
    out_img, out_boxes, out_labels = class_aware_copy_paste(
        img, [], [], bank, schedule,
        rng=rng, image_loader=loader,
    )
    assert len(out_boxes) == 1
    assert out_labels == [0]
    assert (out_img == 255).any()


def test_paste_does_not_fire_when_prob_zero():
    rng = random.Random(0)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bank = InstanceBank()
    bank.add(CropEntry("/x.png", (0, 0, 10, 10), 0))
    schedule = {0: 0.0}
    loader = _make_image_loader({"/x.png": np.full((10, 10, 3), 255, dtype=np.uint8)})

    out_img, out_boxes, out_labels = class_aware_copy_paste(
        img, [], [], bank, schedule, rng=rng, image_loader=loader
    )
    assert len(out_boxes) == 0
    assert np.array_equal(out_img, img)


def test_paste_skipped_for_empty_class_bank():
    """A class with prob 1.0 but no crops in bank should silently skip."""
    rng = random.Random(0)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bank = InstanceBank()
    schedule = {0: 1.0, 1: 1.0}

    bank.add(CropEntry("/x.png", (0, 0, 10, 10), 1))
    loader = _make_image_loader({"/x.png": np.full((10, 10, 3), 200, dtype=np.uint8)})

    out_img, out_boxes, out_labels = class_aware_copy_paste(
        img, [], [], bank, schedule, rng=rng, image_loader=loader
    )
    assert sorted(out_labels) == [1]


def test_paste_respects_max_iou_with_existing():
    """If the entire image is already covered by an existing GT box, paste should fail."""
    rng = random.Random(0)
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    bank = InstanceBank()
    bank.add(CropEntry("/x.png", (0, 0, 30, 30), 0))
    loader = _make_image_loader({"/x.png": np.full((30, 30, 3), 200, dtype=np.uint8)})

    existing = [(0, 0, 50, 50)]
    schedule = {0: 1.0}
    out_img, out_boxes, out_labels = class_aware_copy_paste(
        img, existing, [99], bank, schedule,
        rng=rng, image_loader=loader, max_iou=0.05,
    )
    assert len(out_boxes) == 1, "should still keep original box"
    assert out_labels == [99]


def test_paste_pickle_round_trip(tmp_path):
    bank = InstanceBank(image_root=tmp_path)
    bank.add(CropEntry("/x.png", (0, 0, 10, 10), 0))
    bank.add(CropEntry("/y.png", (5, 5, 20, 20), 1))
    out = tmp_path / "bank.pkl"
    write_bank(bank, out)
    loaded = load_bank(out)
    assert loaded.total() == 2
    assert loaded.n_crops_for(0) == 1
    assert loaded.n_crops_for(1) == 1


def test_paste_input_validation():
    bank = InstanceBank()
    bank.add(CropEntry("/x.png", (0, 0, 10, 10), 0))
    schedule = {0: 1.0}
    img = np.zeros((50, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        class_aware_copy_paste(np.zeros((50, 50), dtype=np.uint8), [], [], bank, schedule)
    with pytest.raises(ValueError):
        class_aware_copy_paste(img, [(0, 0, 5, 5)], [], bank, schedule)


def test_multiple_paste_max_per_class():
    rng = random.Random(0)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    bank = InstanceBank()
    bank.add(CropEntry("/x.png", (0, 0, 10, 10), 0))
    loader = _make_image_loader({"/x.png": np.full((10, 10, 3), 100, dtype=np.uint8)})
    schedule = {0: 1.0}
    _, out_boxes, out_labels = class_aware_copy_paste(
        img, [], [], bank, schedule,
        rng=rng, image_loader=loader, max_paste_per_class=3, max_iou=0.0,
    )
    assert sum(1 for lbl in out_labels if lbl == 0) >= 1
