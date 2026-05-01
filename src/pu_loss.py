"""Kiryo non-negative PU loss for the Phase 3 EXP 3.2 experiment.

Block reference: E.3 in master_checklist.txt.

Reference
---------
Kiryo, Ryuichi et al. "Positive-Unlabeled Learning with Non-Negative Risk
Estimator." NeurIPS 2017. https://arxiv.org/abs/1703.00593

What problem does PU loss solve
-------------------------------
FathomNet 2026 has Positive-Unlabeled (PU) annotation: each train image
is labeled for *some* of the categories present, not necessarily all.
Phase 0 confirmed 62.5% of train images contain only one labeled class,
even though most marine scenes contain multiple species. So when the
detector encounters an anchor over an UNLABELED region, that region
might actually contain an object the labelers missed.

If we naively treat every unlabeled region as "background" (the standard
BCE objectness loss does this), the model is being TOLD that a real
object is background, and it eventually learns to suppress detections
on those objects -- which is the OPPOSITE of what we want.

PU loss reformulates the problem: assume only labeled regions are
genuine positives; treat unlabeled regions as a mixture of true
negatives + missed positives, with mixing ratio pi (the class prior).

The Kiryo formulation
---------------------
For binary classification with sigmoid output g(x) and loss L:

    R_pu(g) = pi * E_p[L(g(x), 1)]                          (pos. risk)
            + max(0, E_u[L(g(x), 0)] - pi * E_p[L(g(x), 0)]) (non-neg trick)

The non-negative trick (the max with 0) prevents the negative-class
risk from going negative when pi is mis-estimated -- a well-known
failure mode of the original "uPU" estimator.

For object detection, we apply this to OBJECTNESS specifically (the
"is there any object here at all" head). Per-class classification still
uses standard cross-entropy or EFL (see src/efl_loss.py).

How pi is estimated for FathomNet 2026
--------------------------------------
pi = P(positive anchor) is approximated by:

    pi ~ avg fraction of pixel area occupied by GT boxes per image

E.g., if on average 8% of pixels per image are inside a GT box, pi = 0.08.
This is computed by `estimate_pu_prior()` from train_dataset.json.

Phase 0 expectation: pi ~ 0.05-0.15 for FathomNet (scenes are mostly
background; GT boxes are small relative to image area).

Status (May 1, 2026)
--------------------
SKELETON + TESTS. NOT yet integrated with Ultralytics' detection loss
path. Phase 3 EXP 3.2 (E.3) is the integration task: subclass
`v8DetectionLoss` and replace its objectness BCE with this Kiryo loss.

Why ship the skeleton now
-------------------------
Tests run on CPU, no GPU needed. Validates the math BEFORE Phase 3 GPU
spend. The non-negative trick has a subtle correctness bug if you use
the wrong sign convention; tests catch that here, not after a 4-hr
training run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class KiryoPULoss(nn.Module):
    """Kiryo non-negative PU loss for binary objectness detection.

    Use as a drop-in replacement for `BCEWithLogitsLoss` when the targets
    represent "this anchor overlaps a labeled GT" (positive) vs "no
    labeled GT here" (unlabeled, NOT necessarily true negative).

    Parameters
    ----------
    pi : float
        Class prior P(positive). Estimate via `estimate_pu_prior()` once
        from train_dataset.json. Typical FathomNet value: 0.05 - 0.15.
    surrogate_loss : str
        'sigmoid' (Kiryo's default) or 'logistic'. The loss applied to
        each individual prediction before risk aggregation.
        - 'sigmoid':  L(z, y) = sigmoid(-z)  if y=1  else  sigmoid(z)
        - 'logistic': L(z, y) = log(1 + exp(-z))  if y=1  else  log(1 + exp(z))
        'sigmoid' is bounded in [0, 1] and is more stable when pi is
        mis-estimated. 'logistic' matches BCE loss.
    beta : float
        Threshold for the non-negative correction (Kiryo et al. eq. 7).
        Default 0 (the original "nnPU" formulation). Setting beta > 0
        applies a slightly stronger correction. Rarely useful.
    gamma : float
        Discount factor in the gradient when correction is active.
        Default 1.0 (no discount). Set to 0.0 to fully zero out the
        gradient when the negative risk estimate goes below -beta.
    reduction : str
        'mean', 'sum', or 'none'.
    eps : float
        Numerical-stability term to avoid log(0).

    Notes
    -----
    Aggregation note: the reduction='none' return shape matches the
    input shape (one loss per anchor), but the loss VALUE depends on
    the BATCH (because pi-weighted positive risk and unlabeled risk
    are computed per-batch). So 'none' returns the per-anchor loss
    AFTER batch-level mixing, not a clean per-anchor decomposition.
    Use 'none' for debugging only.

    The non-negative trick is applied at the BATCH level, not the
    per-anchor level. This is intentional and matches the paper.
    """

    def __init__(
        self,
        pi: float,
        *,
        surrogate_loss: str = "sigmoid",
        beta: float = 0.0,
        gamma: float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if not 0 < pi < 1:
            raise ValueError(f"pi must be in (0, 1), got {pi}")
        if surrogate_loss not in ("sigmoid", "logistic"):
            raise ValueError(f"surrogate_loss must be 'sigmoid' or 'logistic', got {surrogate_loss!r}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean'/'sum'/'none', got {reduction!r}")

        self.pi = float(pi)
        self.surrogate_loss = surrogate_loss
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.reduction = reduction
        self.eps = float(eps)

    def _surrogate(self, logits: torch.Tensor, target_value: float) -> torch.Tensor:
        """Per-element surrogate loss; assumes target is constant across the tensor."""
        if self.surrogate_loss == "sigmoid":
            return torch.sigmoid(-logits) if target_value == 1.0 else torch.sigmoid(logits)
        z = -logits if target_value == 1.0 else logits
        return F.softplus(z)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the PU loss.

        Parameters
        ----------
        logits :
            Raw objectness logits, shape (...,). Sigmoid is applied
            internally inside the surrogate; do NOT pre-apply.
        targets :
            Binary labels in {0, 1}: 1 = anchor overlaps a labeled GT
            (positive), 0 = anchor does NOT overlap any labeled GT
            (unlabeled, NOT a true negative). Shape must match logits.

        Returns
        -------
        torch.Tensor
            Scalar (mean/sum) or per-anchor tensor (none).
        """
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits {tuple(logits.shape)} vs targets {tuple(targets.shape)} shape mismatch"
            )
        if logits.numel() == 0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        flat_logits = logits.reshape(-1)
        flat_targets = targets.reshape(-1)

        pos_mask = flat_targets > 0.5
        unl_mask = ~pos_mask
        n_pos = pos_mask.sum().clamp_min(1)
        n_unl = unl_mask.sum().clamp_min(1)

        loss_pos_treated_as_positive = self._surrogate(flat_logits, 1.0)
        loss_pos_treated_as_negative = self._surrogate(flat_logits, 0.0)

        loss_unl_treated_as_negative = self._surrogate(flat_logits, 0.0)

        risk_pos_pos = (loss_pos_treated_as_positive * pos_mask.float()).sum() / n_pos
        risk_pos_neg = (loss_pos_treated_as_negative * pos_mask.float()).sum() / n_pos
        risk_unl_neg = (loss_unl_treated_as_negative * unl_mask.float()).sum() / n_unl

        positive_risk = self.pi * risk_pos_pos
        negative_risk_uncorrected = risk_unl_neg - self.pi * risk_pos_neg

        if negative_risk_uncorrected.item() < -self.beta:
            if self.gamma == 0.0:
                total = positive_risk.detach()
            else:
                total = positive_risk - self.gamma * negative_risk_uncorrected
        else:
            total = positive_risk + negative_risk_uncorrected

        if self.reduction == "mean":
            return total
        if self.reduction == "sum":
            n_total = (n_pos + n_unl).float()
            return total * n_total

        per_anchor = torch.zeros_like(flat_logits)
        per_anchor[pos_mask] = self.pi * loss_pos_treated_as_positive[pos_mask]
        per_anchor[unl_mask] = loss_unl_treated_as_negative[unl_mask]
        return per_anchor.reshape_as(logits)

    def extra_repr(self) -> str:
        return (
            f"pi={self.pi}, surrogate={self.surrogate_loss!r}, "
            f"beta={self.beta}, gamma={self.gamma}, reduction={self.reduction!r}"
        )


def estimate_pu_prior(
    train_json_path: str | Path,
    *,
    method: str = "area_fraction",
    verbose: bool = True,
) -> float:
    """Estimate the PU class prior pi from train_dataset.json.

    Parameters
    ----------
    train_json_path :
        Path to the COCO-format train_dataset.json shipped by Kaggle.
    method : str
        - 'area_fraction':   pi = mean over images of (sum of GT box areas / image area).
                             Best when "positive" means "anchor overlaps a GT box."
        - 'annotation_rate': pi = (n_annotations) / (n_images * 100).
                             Approximates the fraction of typical anchors
                             (~100 per image) that are positives.
    verbose :
        If True, print a per-image area-distribution summary.

    Returns
    -------
    float
        Estimated pi in (0, 1).

    Notes
    -----
    The estimate is intentionally rough -- the Kiryo loss is robust to
    pi being mis-estimated by 50% in either direction, thanks to the
    non-negative correction. So we do NOT need a sophisticated estimator
    (e.g., Elkan-Noto). For Phase 3 EXP 3.1 (E.2), this one-liner
    suffices.
    """
    train_json_path = Path(train_json_path)
    with train_json_path.open() as f:
        coco = json.load(f)

    if method not in ("area_fraction", "annotation_rate"):
        raise ValueError(f"method must be 'area_fraction' or 'annotation_rate', got {method!r}")

    n_images = len(coco["images"])
    n_anns = len(coco["annotations"])

    if method == "annotation_rate":
        anchors_per_image = 100
        pi = n_anns / (n_images * anchors_per_image)
        if verbose:
            print(f"[estimate_pu_prior] annotation_rate: {n_anns:,} anns / "
                  f"({n_images:,} imgs * {anchors_per_image} anchors) = {pi:.4f}")
        return float(pi)

    image_id_to_area: dict[int, float] = {}
    for img in coco["images"]:
        h = float(img.get("height", 0)) or 1.0
        w = float(img.get("width", 0)) or 1.0
        image_id_to_area[int(img["id"])] = h * w

    image_id_to_box_area_sum: dict[int, float] = {iid: 0.0 for iid in image_id_to_area}
    for ann in coco["annotations"]:
        iid = int(ann["image_id"])
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            continue
        _, _, w_box, h_box = bbox
        if w_box <= 0 or h_box <= 0:
            continue
        image_id_to_box_area_sum[iid] = image_id_to_box_area_sum.get(iid, 0.0) + float(w_box) * float(h_box)

    fractions = []
    for iid, total_box_area in image_id_to_box_area_sum.items():
        img_area = image_id_to_area[iid]
        fractions.append(min(total_box_area / img_area, 1.0))

    if not fractions:
        raise RuntimeError(f"No annotations found in {train_json_path}; check file integrity")

    fractions_tensor = torch.tensor(fractions)
    pi = float(fractions_tensor.mean().item())

    if verbose:
        print(f"[estimate_pu_prior] area_fraction:")
        print(f"  images:       {n_images:,}")
        print(f"  annotations:  {n_anns:,}")
        print(f"  fraction stats:")
        print(f"    mean:    {fractions_tensor.mean().item():.4f}  <-- pi estimate")
        print(f"    median:  {fractions_tensor.median().item():.4f}")
        print(f"    p5:      {fractions_tensor.quantile(0.05).item():.4f}")
        print(f"    p95:     {fractions_tensor.quantile(0.95).item():.4f}")

    if pi < 0.005 or pi > 0.5:
        print(f"[estimate_pu_prior] WARNING: pi={pi:.4f} is outside the expected "
              f"FathomNet range [0.005, 0.5]. Sanity-check the JSON file.")

    return pi


__all__ = ["KiryoPULoss", "estimate_pu_prior"]
