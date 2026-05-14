"""Kiryo non-negative PU correction for Ultralytics 8.4.x's `v8DetectionLoss`.

Status (May 2, 2026 ~09:30 PT)
------------------------------
v3 — METHOD-level monkey-patch on `v8DetectionLoss.get_assigned_targets_and_loss`.
This avoids the pickling issue that v2 hit (Ultralytics saves
`model.criterion` to checkpoints; a custom subclass defined inside a
factory function can't be pickled).

What problem this solves
------------------------
F-006 / F-008 in `notes/FINDINGS.md`: naive supervised baseline scored
0.0498 on Kaggle LB despite hitting 0.40 val mAP. The 8x val→LB gap is
caused by training on Positive-Unlabeled data (where unlabeled regions
are treated as background) and being scored against a full-coverage test
set. Kiryo PU loss is the principled fix.

The PU formulation we use (per-class)
-------------------------------------
For each class c, define:
- positive set P_c = anchors with target_score[i,c] > 0 (TAL-matched)
- unlabeled set U_c = anchors with target_score[i,c] == 0

Per-class quantities:
    sum_bce_pos_c        = sum over P_c of BCE(logit, target)        # original loss on matched (soft target)
    sum_bce_neg_at_pos_c = sum over P_c of BCE(logit, 0)              # what loss WOULD BE if pos treated as bg
    sum_bce_neg_at_neg_c = sum over U_c of BCE(logit, 0)              # standard "this is bg" loss
    n_pos_c, n_neg_c     = anchor counts in P_c, U_c

Kiryo non-negative PU loss (per class):
    PU_pos_c       = pi_c * sum_bce_pos_c
    PU_neg_c       = sum_bce_neg_at_neg_c - (n_neg_c / n_pos_c) * pi_c * sum_bce_neg_at_pos_c
    PU_neg_c_clip  = max(0, PU_neg_c)
    PU_per_class_c = PU_pos_c + PU_neg_c_clip

Total cls loss:
    loss_cls = sum_c PU_per_class_c / target_scores_sum

Matches Ultralytics' sum-then-divide normalizer exactly.

Integration approach (v3)
-------------------------
We monkey-patch `ultralytics.utils.loss.v8DetectionLoss.get_assigned_targets_and_loss`
at the CLASS level. ALL v8DetectionLoss instances in the process get our
PU-aware version. This is robust to:
- Trainer creating fresh DetectionModel instances (handled — patch is on
  the loss class, not the model)
- Ultralytics pickling `model.criterion` for checkpoint saving (handled —
  no custom class is created)
- Resuming training from a checkpoint (handled — patch lives in the
  Python process; re-install at start of resumed run)

Per-batch state (current epoch, pi_per_class) lives in module-level
mutable globals, accessed via the patched function's closure. The
`set_epoch()` callback (called by the trainer at the start of each
epoch) updates the global epoch counter.

Risks / edge cases
------------------
- Validator uses the same v8DetectionLoss class. After install, val
  loss numbers in `results.csv` reflect PU loss (not BCE). mAP is
  unaffected (computed separately via the validator's metrics path).
- Multiple training runs in the same process share the same patch.
  Call `uninstall()` between runs if needed.
- Weights saved during PU training are vanilla `.pt` files; loading
  them with stock Ultralytics (no patch) works fine for inference.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Module-level mutable state. Set via install_pu_loss_on_model() and
# updated by set_epoch() callback. Read by the patched
# get_assigned_targets_and_loss function.
_PU_STATE: dict[str, Any] = {
    "installed": False,
    "current_epoch": 0,
    "warmup_epochs": 3,
    "pi_min": 0.005,
    "pi_max": 0.5,
    "pi_scalar": None,           # if not None, broadcast to model.nc at runtime
    "pi_per_class": None,        # explicit per-class (must match model.nc)
    "orig_method": None,         # reference to original method for uninstall
    "call_count": 0,             # bumped each time the patched method is called
    "warmup_calls": 0,           # bumped during warmup epochs (BCE branch)
    "pu_calls": 0,               # bumped after warmup (PU branch)
}


def _patched_get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
    """PU-corrected replacement for v8DetectionLoss.get_assigned_targets_and_loss.

    Mirrors the original method byte-for-byte except for the cls loss
    block, which uses Kiryo non-negative PU correction (per-class).
    During warmup epochs, falls through to the original BCE behavior.
    """
    from ultralytics.utils.tal import make_anchors
    from ultralytics.utils.ops import xywh2xyxy

    _PU_STATE["call_count"] += 1

    loss = torch.zeros(3, device=self.device)
    pred_distri, pred_scores = (
        preds["boxes"].permute(0, 2, 1).contiguous(),
        preds["scores"].permute(0, 2, 1).contiguous(),
    )
    anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = (
        torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
        * self.stride[0]
    )

    targets = torch.cat(
        (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
        1,
    )
    targets = self.preprocess(
        targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
    )
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

    _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)
    target_scores_t = target_scores.to(dtype)

    # === CLS LOSS: PU-corrected (or plain BCE during warmup) ===
    is_warmup = _PU_STATE["current_epoch"] < _PU_STATE["warmup_epochs"]
    if is_warmup:
        _PU_STATE["warmup_calls"] += 1
        bce_loss = self.bce(pred_scores, target_scores_t)
        if self.class_weights is not None:
            bce_loss = bce_loss * self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum
    else:
        _PU_STATE["pu_calls"] += 1
        loss[1] = _compute_pu_cls_loss(
            pred_scores, target_scores_t, target_scores_sum, self.nc
        )

    # === BBOX & DFL LOSSES: unchanged ===
    if fg_mask.sum():
        loss[0], loss[2] = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )

    loss[0] *= self.hyp.box
    loss[1] *= self.hyp.cls
    loss[2] *= self.hyp.dfl

    return (
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
        loss,
        loss.detach(),
    )


def _compute_pu_cls_loss(
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    nc: int,
) -> torch.Tensor:
    """Per-class Kiryo PU correction on classification loss.

    Shapes:
      pred_scores:    (B, A, C) logits
      target_scores:  (B, A, C) soft targets in [0, 1]
      target_scores_sum: scalar
    """
    pi = _resolve_pi_for_nc(nc, device=pred_scores.device, dtype=pred_scores.dtype)

    pos_mask = target_scores > 0
    neg_mask = ~pos_mask

    n_pos_c = pos_mask.float().sum(dim=(0, 1)).clamp_min(1.0)
    n_neg_c = neg_mask.float().sum(dim=(0, 1)).clamp_min(1.0)

    bce_pos_loss = F.binary_cross_entropy_with_logits(
        pred_scores, target_scores, reduction="none"
    )
    bce_neg_loss = F.binary_cross_entropy_with_logits(
        pred_scores, torch.zeros_like(pred_scores), reduction="none"
    )

    sum_bce_pos_c = (bce_pos_loss * pos_mask.float()).sum(dim=(0, 1))
    sum_bce_neg_at_pos_c = (bce_neg_loss * pos_mask.float()).sum(dim=(0, 1))
    sum_bce_neg_at_neg_c = (bce_neg_loss * neg_mask.float()).sum(dim=(0, 1))

    pu_pos_c = pi * sum_bce_pos_c
    pu_neg_c = sum_bce_neg_at_neg_c - (n_neg_c / n_pos_c) * pi * sum_bce_neg_at_pos_c
    pu_neg_c_clipped = pu_neg_c.clamp_min(0.0)

    pu_per_class = pu_pos_c + pu_neg_c_clipped
    return pu_per_class.sum() / target_scores_sum


def _resolve_pi_for_nc(nc: int, *, device, dtype) -> torch.Tensor:
    """Materialize the per-class pi tensor for the model's actual `nc`.

    Caches the result keyed by (nc, device, dtype) for efficiency.
    """
    cached = _PU_STATE.get("_pi_cache", {})
    key = (nc, str(device), str(dtype))
    if key in cached:
        return cached[key]

    if _PU_STATE["pi_per_class"] is not None:
        explicit = _PU_STATE["pi_per_class"]
        if explicit.shape[0] != nc:
            raise ValueError(
                f"_compute_pu_cls_loss: explicit pi_per_class has {explicit.shape[0]} "
                f"entries but model.nc={nc}"
            )
        pi = explicit.to(device=device, dtype=dtype)
    else:
        pi_scalar = float(_PU_STATE["pi_scalar"])
        pi = torch.full((nc,), pi_scalar, device=device, dtype=dtype)

    pi = pi.clamp(_PU_STATE["pi_min"], _PU_STATE["pi_max"])
    cached[key] = pi
    _PU_STATE["_pi_cache"] = cached
    return pi


def install_pu_loss_on_model(
    yolo_model,
    *,
    num_classes: int,
    pi: float | torch.Tensor = 0.1401,
    pi_min: float = 0.005,
    pi_max: float = 0.5,
    warmup_epochs: int = 3,
):
    """Install the Kiryo PU loss patch on Ultralytics' v8DetectionLoss.

    Use this AFTER `model = YOLO(weights)` and BEFORE `model.train(...)`.

    Parameters
    ----------
    yolo_model :
        An `ultralytics.YOLO` instance. Used to clear any stale criterion
        cached on the underlying model.
    num_classes :
        Hint for the number of classes (informational; runtime resolves
        the actual nc from `self.nc` during the patched method call).
    pi :
        Either a scalar (broadcast to all classes at runtime) or a
        Tensor of length num_classes (per-class pi).
    pi_min, pi_max :
        Clamp range for per-class pi values.
    warmup_epochs :
        Use plain BCE for first N epochs (let the model lock onto easy
        positives before PU starts protecting unlabeled potentials).

    Returns
    -------
    PUInstallHandle
        Object exposing `set_epoch(epoch)`, `uninstall()`, and
        `state()` introspection. Keep a reference and call set_epoch
        from your training callback.
    """
    if isinstance(pi, (int, float)):
        _PU_STATE["pi_scalar"] = float(pi)
        _PU_STATE["pi_per_class"] = None
    else:
        _PU_STATE["pi_scalar"] = None
        explicit = pi.float().clone()
        if explicit.shape != (num_classes,):
            raise ValueError(
                f"install_pu_loss_on_model: pi must be scalar or shape "
                f"({num_classes},), got {tuple(explicit.shape)}"
            )
        _PU_STATE["pi_per_class"] = explicit

    _PU_STATE["pi_min"] = pi_min
    _PU_STATE["pi_max"] = pi_max
    _PU_STATE["warmup_epochs"] = warmup_epochs
    _PU_STATE["current_epoch"] = 0
    _PU_STATE["call_count"] = 0
    _PU_STATE["warmup_calls"] = 0
    _PU_STATE["pu_calls"] = 0
    _PU_STATE["_pi_cache"] = {}

    from ultralytics.utils.loss import v8DetectionLoss

    if not _PU_STATE["installed"]:
        _PU_STATE["orig_method"] = v8DetectionLoss.get_assigned_targets_and_loss
        v8DetectionLoss.get_assigned_targets_and_loss = _patched_get_assigned_targets_and_loss
        _PU_STATE["installed"] = True
        print(
            f"[install_pu_loss_on_model] METHOD-level patch installed on "
            f"v8DetectionLoss.get_assigned_targets_and_loss "
            f"(num_classes_hint={num_classes}, pi={pi if isinstance(pi, (int, float)) else 'per-class'}, "
            f"pi_min={pi_min}, pi_max={pi_max}, warmup_epochs={warmup_epochs}). "
            f"Actual nc resolved at training time."
        )
    else:
        print(
            f"[install_pu_loss_on_model] re-using existing patch with new state "
            f"(num_classes_hint={num_classes}, pi={pi if isinstance(pi, (int, float)) else 'per-class'}, "
            f"pi_min={pi_min}, pi_max={pi_max}, warmup_epochs={warmup_epochs})."
        )

    underlying = yolo_model.model
    if hasattr(underlying, "criterion"):
        underlying.criterion = None

    return PUInstallHandle()


class PUInstallHandle:
    """Lightweight handle for the user to drive the installed PU patch."""

    @staticmethod
    def set_epoch(epoch: int) -> None:
        _PU_STATE["current_epoch"] = int(epoch)

    @staticmethod
    def uninstall() -> None:
        if not _PU_STATE["installed"]:
            return
        from ultralytics.utils.loss import v8DetectionLoss
        v8DetectionLoss.get_assigned_targets_and_loss = _PU_STATE["orig_method"]
        _PU_STATE["installed"] = False
        _PU_STATE["orig_method"] = None
        print("[install_pu_loss_on_model] restored original v8DetectionLoss.get_assigned_targets_and_loss")

    @staticmethod
    def state() -> dict[str, Any]:
        return {
            "installed": _PU_STATE["installed"],
            "current_epoch": _PU_STATE["current_epoch"],
            "warmup_epochs": _PU_STATE["warmup_epochs"],
            "call_count": _PU_STATE["call_count"],
            "warmup_calls": _PU_STATE["warmup_calls"],
            "pu_calls": _PU_STATE["pu_calls"],
        }


__all__ = ["install_pu_loss_on_model", "PUInstallHandle"]
