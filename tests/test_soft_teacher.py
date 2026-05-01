"""Tests for src/soft_teacher.py.

Covers:
  - apply_pseudo_label_filters score threshold + IoU + per-image cap.
  - Pseudo-labels for classes already GT-labeled in the image are dropped
    when suppress_class_already_gt_in_image=True.
  - Per-class global cap selects highest-score predictions.
  - merge_gt_and_pseudo() preserves original annotations and adds new
    annotations with unique ids and is_pseudo=True flag.
  - merge_gt_and_pseudo() applies idx_to_cat_id correctly.
  - update_ema_teacher() respects the decay rate exactly.
  - Stats tracking is correct across the pipeline.

Run from repo root:
    pytest tests/test_soft_teacher.py -v
"""
from __future__ import annotations

import copy

import pytest
import torch

from src.soft_teacher import (
    FilterStats,
    PseudoLabel,
    apply_pseudo_label_filters,
    merge_gt_and_pseudo,
    update_ema_teacher,
)


def _pl(image_id, class_idx, bbox, score):
    return PseudoLabel(image_id=image_id, class_idx=class_idx, bbox_xywh=bbox, score=score)


def test_score_threshold_filters_low_confidence():
    preds = [
        _pl(1, 0, (0, 0, 10, 10), 0.9),
        _pl(1, 1, (50, 50, 10, 10), 0.4),
        _pl(1, 2, (100, 100, 10, 10), 0.6),
    ]
    out, stats = apply_pseudo_label_filters(
        preds, gt_by_image={}, score_threshold=0.5, return_stats=True,
    )
    assert {p.score for p in out} == {0.9, 0.6}
    assert stats.n_predictions_in == 3
    assert stats.n_kept_after_score == 2


def test_suppress_class_already_gt():
    preds = [
        _pl(1, 0, (200, 200, 10, 10), 0.9),
        _pl(1, 1, (300, 300, 10, 10), 0.9),
    ]
    gt = {1: [(0, (0, 0, 50, 50))]}
    out = apply_pseudo_label_filters(
        preds, gt_by_image=gt, score_threshold=0.5,
        suppress_class_already_gt_in_image=True,
        iou_with_gt_threshold=1.1,
    )
    assert [p.class_idx for p in out] == [1]


def test_iou_with_gt_filter():
    preds = [
        _pl(1, 5, (5, 5, 50, 50), 0.9),
        _pl(1, 5, (200, 200, 50, 50), 0.9),
    ]
    gt = {1: [(0, (0, 0, 50, 50))]}
    out = apply_pseudo_label_filters(
        preds, gt_by_image=gt, score_threshold=0.5,
        suppress_class_already_gt_in_image=False,
        iou_with_gt_threshold=0.3,
    )
    assert len(out) == 1
    assert out[0].bbox_xywh == (200, 200, 50, 50)


def test_per_image_cap_keeps_top_k_by_score():
    preds = [_pl(1, 0, (i * 10, 0, 5, 5), 0.5 + i * 0.1) for i in range(5)]
    out = apply_pseudo_label_filters(
        preds, gt_by_image={}, score_threshold=0.5, max_per_image=2,
    )
    assert len(out) == 2
    assert {p.score for p in out} == {0.9, 0.8}


def test_per_class_global_cap():
    preds = (
        [_pl(i, 31, (i * 10, 0, 5, 5), 0.5 + i * 0.05) for i in range(1, 11)]
        + [_pl(100 + i, 22, (0, 0, 5, 5), 0.99 - i * 0.01) for i in range(3)]
    )
    out = apply_pseudo_label_filters(
        preds, gt_by_image={}, score_threshold=0.5,
        max_per_class_globally={31: 3},
        max_per_image=None,
    )
    urchin = [p for p in out if p.class_idx == 31]
    sea_slug = [p for p in out if p.class_idx == 22]
    assert len(urchin) == 3
    assert all(p.score >= 0.85 for p in urchin)
    assert len(sea_slug) == 3


def test_filter_stats_complete():
    preds = [
        _pl(1, 0, (0, 0, 5, 5), 0.9),
        _pl(1, 0, (0, 0, 5, 5), 0.4),
        _pl(2, 1, (50, 50, 5, 5), 0.7),
    ]
    gt = {1: [(0, (0, 0, 5, 5))]}
    out, stats = apply_pseudo_label_filters(
        preds, gt_by_image=gt, score_threshold=0.5,
        suppress_class_already_gt_in_image=True,
        return_stats=True,
    )
    assert stats.n_predictions_in == 3
    assert stats.n_kept_after_score == 2
    assert stats.n_kept_after_class_in_image_filter == 1
    assert stats.n_pseudo_per_class == {1: 1}


def test_merge_preserves_originals():
    gt_coco = {
        "images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10], "area": 100}],
        "categories": [{"id": 5, "name": "five"}],
    }
    pseudo = [_pl(1, 7, (50, 50, 20, 20), 0.85)]
    idx_to_cat = {7: 13}

    merged = merge_gt_and_pseudo(gt_coco, pseudo, idx_to_cat_id=idx_to_cat)
    original_ann = next(a for a in merged["annotations"] if a["id"] == 1)
    assert original_ann["category_id"] == 5

    pseudo_ann = next(a for a in merged["annotations"] if a.get("is_pseudo"))
    assert pseudo_ann["category_id"] == 13
    assert pseudo_ann["id"] == 2
    assert pseudo_ann["bbox"] == [50.0, 50.0, 20.0, 20.0]
    assert pseudo_ann["pseudo_score"] == 0.85


def test_merge_skips_unmapped_classes():
    """If a pseudo-label has a class not in idx_to_cat_id, skip it (would crash on submit)."""
    gt_coco = {
        "images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
        "annotations": [],
        "categories": [],
    }
    pseudo = [_pl(1, 99, (0, 0, 10, 10), 0.9)]
    merged = merge_gt_and_pseudo(gt_coco, pseudo, idx_to_cat_id={0: 1})
    assert merged["annotations"] == []


def test_merge_does_not_modify_input():
    gt_coco = {
        "images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10], "area": 100}],
        "categories": [{"id": 5, "name": "five"}],
    }
    snapshot = copy.deepcopy(gt_coco)
    pseudo = [_pl(1, 7, (50, 50, 20, 20), 0.85)]
    idx_to_cat = {7: 13}
    _ = merge_gt_and_pseudo(gt_coco, pseudo, idx_to_cat_id=idx_to_cat)
    assert gt_coco == snapshot


def test_update_ema_teacher_basic():
    student = torch.nn.Linear(2, 2)
    teacher = torch.nn.Linear(2, 2)

    with torch.no_grad():
        student.weight.fill_(1.0)
        teacher.weight.fill_(0.0)

    update_ema_teacher(teacher, student, decay=0.9)
    expected = 0.9 * 0.0 + 0.1 * 1.0
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, expected))


def test_update_ema_teacher_invalid_decay():
    student = torch.nn.Linear(2, 2)
    teacher = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError):
        update_ema_teacher(teacher, student, decay=0.0)
    with pytest.raises(ValueError):
        update_ema_teacher(teacher, student, decay=1.5)


def test_update_ema_teacher_mismatched_keys():
    student = torch.nn.Linear(2, 2)
    teacher = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError):
        update_ema_teacher(teacher, student, decay=0.999)
