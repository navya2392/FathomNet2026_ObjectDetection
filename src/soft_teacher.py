"""Soft Teacher pseudo-labeling for the Phase 3 EXP 3.5 / Phase 5 EXP 5.2 experiments.

Block reference: E.6 (Phase 3) + G.3 (Phase 5) in master_checklist.txt.

Reference
---------
Xu, Mengde et al. "End-to-End Semi-Supervised Object Detection with Soft Teacher."
ICCV 2021. https://arxiv.org/abs/2106.09018

What problem does Soft Teacher solve
------------------------------------
Phase 0 confirmed FathomNet 2026 has Positive-Unlabeled annotation:
62.5% of train images contain only ONE labeled class even though most
marine scenes contain several. There is HUGE unused supervisory signal
sitting in the train images themselves -- if we can recover it.

Soft Teacher recovers it by:
1. Training a strong "teacher" detector on the labeled GT only.
2. Running the teacher on the SAME train images to predict bboxes.
3. Filtering predictions to keep only confident, non-overlapping ones
   that don't conflict with existing GT.
4. Combining GT + pseudo-labels into a richer training set.
5. Re-training a "student" on the combined set.

The official Soft Teacher uses an online EMA-teacher + dual-view
(weak-aug for teacher inference, strong-aug for student training)
augmentation pattern. For our compressed 7-day timeline, we use the
"poor-man's" offline variant:
  Round 1: train on labeled GT only (this is just regular Phase 2-3 training).
  Round 2: predict on train images, filter, merge, retrain.
  Round 3 (Phase 5): repeat with the better Round 2 student as teacher.

This module provides the pseudo-label generation + filtering + merging
plumbing. The Round 1 / Round 2 / Round 3 model training itself uses
the standard `scripts/train_phase2_baseline.py` (or the Phase 3 variant
when E.6 is implemented).

Status (May 1, 2026)
--------------------
SKELETON. The filter + merge logic is implemented and unit-tested.
The orchestration script that runs the full round (predict -> filter
-> merge -> retrain) is `scripts/run_soft_teacher_round.py` (TBD).

Why ship the skeleton now
-------------------------
The merge logic has subtle correctness traps:
  - Pseudo-labels for a class that's ALREADY GT-labeled in this image
    must be SUPPRESSED (would create FPs against the labeled positive).
  - Pseudo-labels overlapping an existing GT box must be SUPPRESSED
    (a labeled instance shouldn't get a competing pseudo-instance).
  - The merged JSON must keep the original `annotation_id` space so
    Phase 6 evaluation doesn't break.
  - cat_id reverse mapping must be applied EXACTLY ONCE.

Tests catch all four of these without spending GPU time.
"""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class PseudoLabel:
    """One pseudo-label predicted by the teacher and accepted by the filter."""

    image_id: int
    class_idx: int
    bbox_xywh: tuple[float, float, float, float]
    score: float


@dataclass
class FilterStats:
    """Diagnostic stats from apply_pseudo_label_filters()."""

    n_predictions_in: int = 0
    n_kept_after_score: int = 0
    n_kept_after_class_in_image_filter: int = 0
    n_kept_after_iou_with_gt: int = 0
    n_kept_after_per_image_cap: int = 0
    n_pseudo_per_class: dict[int, int] = field(default_factory=dict)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def apply_pseudo_label_filters(
    predictions: Iterable[PseudoLabel],
    gt_by_image: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    *,
    score_threshold: float = 0.5,
    iou_with_gt_threshold: float = 0.3,
    suppress_class_already_gt_in_image: bool = True,
    max_per_image: Optional[int] = 50,
    max_per_class_globally: Optional[dict[int, int]] = None,
    return_stats: bool = False,
) -> list[PseudoLabel] | tuple[list[PseudoLabel], FilterStats]:
    """Filter teacher predictions to keep only high-quality pseudo-labels.

    Filter pipeline (applied IN ORDER -- each stage shrinks the set):
      1. Score >= score_threshold.
      2. (if suppress_class_already_gt_in_image) drop predictions whose
         class already has a labeled GT in the image -- those classes
         are KNOWN to be labeled, so a pseudo-label is either a duplicate
         or a hallucination, both bad.
      3. Drop predictions with IoU > iou_with_gt_threshold against ANY
         existing GT box, regardless of class -- a labeled object
         shouldn't get a competing pseudo-instance.
      4. (if max_per_image) cap to top-K per image by score.
      5. (if max_per_class_globally) cap each class to top-K globally
         by score (prevents the teacher from generating tens of thousands
         of pseudo-urchins which would distort the imbalance further).

    Parameters
    ----------
    predictions :
        Iterable of PseudoLabel from teacher inference.
    gt_by_image :
        {image_id: [(class_idx, bbox_xywh), ...]}. Source of truth for
        what's already labeled.
    score_threshold :
        Minimum confidence to keep. Default 0.5 (Soft Teacher paper uses
        0.7 but their teacher is much stronger; 0.5 is safer for ours).
    iou_with_gt_threshold :
        Drop pseudo-label if it overlaps an existing GT box by more than
        this. Default 0.3.
    suppress_class_already_gt_in_image :
        See pipeline step 2 above.
    max_per_image :
        Cap pseudo-labels per image. Default 50; None = no cap.
    max_per_class_globally :
        Per-class global cap (e.g. {31: 5000} to keep at most 5000 urchin
        pseudo-labels). None = no cap.
    return_stats :
        Also return FilterStats with per-stage counts.

    Returns
    -------
    list[PseudoLabel] (or tuple of (list, stats))
    """
    preds = list(predictions)
    stats = FilterStats(n_predictions_in=len(preds))

    by_score = [p for p in preds if p.score >= score_threshold]
    stats.n_kept_after_score = len(by_score)

    if suppress_class_already_gt_in_image:
        gt_classes_in_image: dict[int, set[int]] = {
            iid: {cls for cls, _ in items} for iid, items in gt_by_image.items()
        }
        by_class_in_image = [
            p for p in by_score
            if p.class_idx not in gt_classes_in_image.get(p.image_id, set())
        ]
    else:
        by_class_in_image = list(by_score)
    stats.n_kept_after_class_in_image_filter = len(by_class_in_image)

    by_iou: list[PseudoLabel] = []
    for p in by_class_in_image:
        existing = [bbox for _, bbox in gt_by_image.get(p.image_id, [])]
        if any(_box_iou(p.bbox_xywh, b) > iou_with_gt_threshold for b in existing):
            continue
        by_iou.append(p)
    stats.n_kept_after_iou_with_gt = len(by_iou)

    if max_per_image is not None:
        per_image: dict[int, list[PseudoLabel]] = defaultdict(list)
        for p in by_iou:
            per_image[p.image_id].append(p)
        capped: list[PseudoLabel] = []
        for iid, lst in per_image.items():
            lst.sort(key=lambda x: -x.score)
            capped.extend(lst[:max_per_image])
        by_image_cap = capped
    else:
        by_image_cap = by_iou
    stats.n_kept_after_per_image_cap = len(by_image_cap)

    if max_per_class_globally is not None:
        per_class: dict[int, list[PseudoLabel]] = defaultdict(list)
        for p in by_image_cap:
            per_class[p.class_idx].append(p)
        capped_global: list[PseudoLabel] = []
        for cid, lst in per_class.items():
            lst.sort(key=lambda x: -x.score)
            cap = max_per_class_globally.get(cid, len(lst))
            capped_global.extend(lst[:cap])
        result = capped_global
    else:
        result = by_image_cap

    stats.n_pseudo_per_class = dict(Counter(p.class_idx for p in result))

    if return_stats:
        return result, stats
    return result


def merge_gt_and_pseudo(
    gt_coco: dict,
    pseudo: list[PseudoLabel],
    *,
    idx_to_cat_id: Optional[dict[int, int]] = None,
    pseudo_score_field: str = "pseudo_score",
) -> dict:
    """Merge pseudo-labels into a copy of the GT COCO JSON.

    Parameters
    ----------
    gt_coco :
        The original COCO-format JSON (parsed). NOT modified in place;
        the function returns a deep copy with merged annotations.
    pseudo :
        Output of apply_pseudo_label_filters().
    idx_to_cat_id :
        Map from 0-31 class_idx back to original COCO category_id (1-41 with
        gaps). If None, uses configs.cat_id_mapping.idx_to_cat_id.
    pseudo_score_field :
        Each pseudo annotation gets this extra field set to its teacher
        score. Use it later for filtering or weighting the loss
        (e.g. weight = pseudo_score in EXP 3.5 final loss).

    Returns
    -------
    dict
        New COCO-format JSON with original GT + pseudo annotations
        appended. New annotation_ids start at max(existing) + 1.
    """
    if idx_to_cat_id is None:
        from configs.cat_id_mapping import idx_to_cat_id as _imported
        idx_to_cat_id = _imported

    out = copy.deepcopy(gt_coco)
    existing_ids = [int(a["id"]) for a in out.get("annotations", [])]
    next_id = max(existing_ids) + 1 if existing_ids else 1

    for p in pseudo:
        if p.class_idx not in idx_to_cat_id:
            continue
        cat_id = idx_to_cat_id[p.class_idx]
        x, y, w, h = p.bbox_xywh
        out["annotations"].append({
            "id": next_id,
            "image_id": int(p.image_id),
            "category_id": int(cat_id),
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(w * h),
            "iscrowd": 0,
            "is_pseudo": True,
            pseudo_score_field: float(p.score),
        })
        next_id += 1

    return out


@torch.no_grad()
def update_ema_teacher(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    *,
    decay: float = 0.999,
) -> None:
    """In-place EMA update of teacher params from student.

    teacher_param = decay * teacher_param + (1 - decay) * student_param

    This is the online "teacher" maintained alongside the student during
    Soft Teacher training. For the offline "poor-man's" variant we use
    in Phase 3 EXP 3.5, the teacher is a STATIC copy of the round-N
    model, so this function is only needed if you upgrade to the full
    online variant later.

    Parameters
    ----------
    teacher : nn.Module
    student : nn.Module
    decay : float
        EMA decay. Soft Teacher paper uses 0.999. Higher = teacher
        changes more slowly. Lower = teacher tracks student more closely.
    """
    if not 0 < decay < 1:
        raise ValueError(f"decay must be in (0, 1), got {decay}")

    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    if set(student_state) != set(teacher_state):
        raise ValueError(
            "Student and teacher state_dicts have different keys. "
            "Did you wrap one of them in a different module structure?"
        )
    for k, t_param in teacher_state.items():
        s_param = student_state[k]
        if not t_param.is_floating_point():
            t_param.copy_(s_param)
            continue
        t_param.mul_(decay).add_(s_param, alpha=1 - decay)


__all__ = [
    "PseudoLabel",
    "FilterStats",
    "apply_pseudo_label_filters",
    "merge_gt_and_pseudo",
    "update_ema_teacher",
]
