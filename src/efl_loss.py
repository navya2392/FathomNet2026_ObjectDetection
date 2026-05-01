"""Equalized Focal Loss (EFL) for the Phase 3 EXP 3.3 experiment.

Block reference: E.4 in master_checklist.txt.

Reference
---------
Li, Bo et al. "Equalized Focal Loss for Dense Long-Tailed Object Detection."
CVPR 2022. https://arxiv.org/abs/2201.02593

What problem does EFL solve
---------------------------
FathomNet 2026 has an extreme class imbalance (817:1 between urchin at
5,723 instances and sea slug at 7 instances). Standard focal loss applies
the same gamma exponent to all classes, so frequent classes dominate the
gradient signal and rare classes (sea slug, isopod, octopus) are
under-represented in updates. EFL fixes this by **adapting gamma per class
based on its gradient ratio**: rare classes get a HIGHER effective gamma,
focusing the loss on their hard examples and pushing the model to actually
learn them.

Phase 0 confirmed the trigger condition: imbalance ratio 817:1 >> our
"include EFL" threshold of 50:1, plus 14 classes with <100 instances and
9 classes with <50. EFL is mandatory for this competition.

Algorithm summary
-----------------
For a binary classification target (one-hot or sigmoid output):

    standard FL:   FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    EFL augments gamma per-class:

        g_j = pos_grad_norm[j] / (pos_grad_norm[j] + neg_grad_norm[j])
        gamma_j = gamma + (1 - g_j) * gamma_b

    where g_j is the class's "gradient ratio" (estimated via EMA across
    training steps). Frequent classes have g_j -> 1 (gamma_j -> gamma),
    rare classes have g_j -> 0 (gamma_j -> gamma + gamma_b, which is
    typically 4-6 vs the standard 2 -- much harsher focusing).

    gamma_b is the "balance gamma," a hyperparameter (paper uses 4).
    The pos_grad_norm and neg_grad_norm are tracked as state inside
    the loss module and updated each backward pass via EMA with
    momentum 0.95-0.99.

Status (May 1, 2026)
--------------------
SKELETON + TESTS. The `EqualizedFocalLoss` class is a drop-in nn.Module
with the right API and EMA accumulator; it is unit-tested for correctness
on synthetic data. NOT yet integrated with Ultralytics' detection loss
in `ultralytics.utils.loss.v8DetectionLoss` -- that integration is the
Phase 3 EXP 3.3 implementation task and lives in a follow-up PR.

The integration plan for Phase 3 (E.4):
  1. Subclass `v8DetectionLoss` and replace its `BCEWithLogitsLoss` for
     classification with `EqualizedFocalLoss`
  2. Create a custom Trainer subclass that uses the new loss
  3. Run on fold 0 for 50 epochs, compare per-class AP vs the Phase 2
     baseline. Success signal: rare-class AP +2-5 pts, frequent-class AP
     within 1 pt of baseline.

Why ship the skeleton now
-------------------------
Tests can run on CPU, no GPU needed. Validates the math BEFORE Phase 3
GPU time is spent. If the EMA accumulator or the gamma-adjustment formula
has a bug, we catch it cheaply tonight rather than during Phase 3 GPU rental.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualizedFocalLoss(nn.Module):
    """Equalized Focal Loss for multi-class detection-style classification.

    Designed for the SIGMOID multi-label head used by YOLO-family detectors
    (one Bernoulli per class per anchor; classes are not exclusive). For
    softmax/cross-entropy heads, the formulas need adjustment -- see paper.

    Parameters
    ----------
    num_classes : int
        Number of classes (32 for FathomNet 2026).
    gamma : float
        Base focal exponent. Standard focal loss uses gamma=2.
    gamma_b : float
        Balance gamma -- max additional gamma applied to the rarest class.
        Paper recommends 4. Total gamma for the rarest class is
        gamma + gamma_b (= 2 + 4 = 6 by default).
    alpha : float | None
        Optional class-balancing weight applied to all positives. If None,
        no alpha weighting (matches the paper's default for EFL since EFL
        replaces alpha-balancing with per-class gamma).
    momentum : float
        EMA momentum for the gradient-ratio accumulator. Higher values
        (closer to 1) make g_j change more slowly; lower values respond
        faster to recent batches. Paper uses 0.95-0.99.
    reduction : str
        'mean', 'sum', or 'none'. Matches PyTorch convention.
    eps : float
        Numerical-stability term added inside log() to avoid log(0).

    Notes
    -----
    The g_j (gradient ratio) accumulator is tracked as a non-persistent
    buffer (`pos_grad_ema`, `neg_grad_ema`) so it is NOT saved to the
    checkpoint -- it's just optimization state, like an Adam moment, and
    can be re-warmed up at the start of resumed training.

    For determinism in tests, you can call `set_grad_ratios()` to pin the
    g_j values to specific numbers (e.g., 0.5 for all classes, recovering
    standard focal loss).
    """

    def __init__(
        self,
        num_classes: int,
        *,
        gamma: float = 2.0,
        gamma_b: float = 4.0,
        alpha: Optional[float] = None,
        momentum: float = 0.95,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be one of mean/sum/none, got {reduction!r}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")

        self.num_classes = num_classes
        self.gamma = float(gamma)
        self.gamma_b = float(gamma_b)
        self.alpha = float(alpha) if alpha is not None else None
        self.momentum = float(momentum)
        self.reduction = reduction
        self.eps = float(eps)

        self.register_buffer(
            "pos_grad_ema", torch.zeros(num_classes), persistent=False,
        )
        self.register_buffer(
            "neg_grad_ema", torch.zeros(num_classes), persistent=False,
        )
        self.register_buffer(
            "_initialized", torch.zeros(1, dtype=torch.bool), persistent=False,
        )

    @torch.no_grad()
    def set_grad_ratios(self, g: torch.Tensor) -> None:
        """Pin the gradient-ratio buffer to a specific value (test helper).

        After calling this, get_grad_ratios() will return exactly `g`
        until the next forward pass updates it (or until the user calls
        set_grad_ratios() again).
        """
        if g.shape != (self.num_classes,):
            raise ValueError(
                f"set_grad_ratios expects shape ({self.num_classes},), got {tuple(g.shape)}"
            )
        self.pos_grad_ema = g.clone().to(self.pos_grad_ema)
        self.neg_grad_ema = (1.0 - g).clamp_min(0.0).to(self.neg_grad_ema)
        self._initialized.fill_(True)

    @torch.no_grad()
    def get_grad_ratios(self) -> torch.Tensor:
        """Return current per-class g_j (the EMA-smoothed gradient ratio)."""
        denom = self.pos_grad_ema + self.neg_grad_ema
        denom = denom.clamp_min(self.eps)
        return self.pos_grad_ema / denom

    def per_class_gamma(self) -> torch.Tensor:
        """Return current per-class effective gamma (gamma + (1 - g_j) * gamma_b)."""
        g = self.get_grad_ratios()
        return self.gamma + (1.0 - g) * self.gamma_b

    @torch.no_grad()
    def _update_ema(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Update pos_grad_ema and neg_grad_ema after a forward pass.

        For the binary-cross-entropy gradient w.r.t. logits, the magnitude is
        |sigmoid(logit) - target|. Summing this magnitude separately over
        positive (target=1) and negative (target=0) anchors gives the per-class
        positive and negative gradient norms used in EFL's g_j formula.

        Shape:
            logits  : (..., C)  -- any leading shape; we flatten to (N, C)
            targets : (..., C)  -- same shape as logits, values in {0, 1}
                                   (multi-label, sigmoid-style)
        """
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits {tuple(logits.shape)} and targets {tuple(targets.shape)} must match"
            )
        flat_l = logits.reshape(-1, self.num_classes)
        flat_t = targets.reshape(-1, self.num_classes)

        p = torch.sigmoid(flat_l)
        grad_mag = (p - flat_t).abs()

        pos_mask = flat_t > 0.5
        neg_mask = ~pos_mask

        per_class_pos = (grad_mag * pos_mask.float()).sum(dim=0)
        per_class_neg = (grad_mag * neg_mask.float()).sum(dim=0)

        if not bool(self._initialized.item()):
            self.pos_grad_ema = per_class_pos.detach()
            self.neg_grad_ema = per_class_neg.detach()
            self._initialized.fill_(True)
        else:
            m = self.momentum
            self.pos_grad_ema = m * self.pos_grad_ema + (1 - m) * per_class_pos.detach()
            self.neg_grad_ema = m * self.neg_grad_ema + (1 - m) * per_class_neg.detach()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        update_ema: bool = True,
    ) -> torch.Tensor:
        """Compute EFL.

        Parameters
        ----------
        logits :
            Raw classification logits, shape (..., num_classes). Sigmoid is
            applied internally; do NOT pre-apply.
        targets :
            Multi-label binary targets, same shape as logits. Values in
            {0, 1}; soft targets are also accepted but EMA assumes a
            (1 - target) "negative" mass, so soft targets bias toward
            the negative class.
        update_ema :
            If True (default), update the gradient-ratio EMA before
            computing the loss (the order is important: EFL uses
            CURRENT g_j to weight the loss; updating after means we
            track gradients of the loss we just computed).

        Returns
        -------
        torch.Tensor
            Scalar (if reduction in mean/sum) or tensor (if reduction='none').
        """
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits {tuple(logits.shape)} and targets {tuple(targets.shape)} must match"
            )
        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"logits last dim ({logits.shape[-1]}) must equal num_classes ({self.num_classes})"
            )

        if update_ema:
            self._update_ema(logits, targets)

        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)

        gamma_j = self.per_class_gamma().to(logits.dtype).to(logits.device)
        broadcast_shape = (1,) * (logits.dim() - 1) + (self.num_classes,)
        gamma_per_pos = gamma_j.view(broadcast_shape).expand_as(logits)

        focal_weight = (1 - p_t).clamp_min(self.eps).pow(gamma_per_pos)

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, gamma={self.gamma}, "
            f"gamma_b={self.gamma_b}, alpha={self.alpha}, "
            f"momentum={self.momentum}, reduction={self.reduction!r}"
        )


__all__ = ["EqualizedFocalLoss"]
