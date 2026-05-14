"""Federated Loss for long-tail object detection.

Reference: Zhou et al., "Probabilistic Two-Stage Detection" (CenterNet2),
2021 — Section 3.4 introduces "Federated Loss". Used by all major LVIS
challenge winners since 2021.

Idea
----
Standard BCE classification loss in YOLO computes loss for ALL classes on
every anchor:

    L_cls = sum_c BCE(p_c, y_c)        for c in {0..C-1}

For very rare classes, the negative-sample gradient (y_c=0, p_c~=0) is
applied billions of times more than positive-sample gradients. The model
learns "always predict not-this-class" for rare categories.

Federated Loss restricts the per-anchor classification sum to:

    F = positive_classes_in_image  union  Sample(N from negative_classes)

i.e., we randomly sample N "negative" classes per image (typically N=50
out of 1000 in LVIS, 16 of 32 here) and only compute BCE for those. The
positive-class gradients are unchanged; the negative-sample gradient
budget is now spread thinner across rare classes, so they're not drowned.

Combine vs. EQLv2
-----------------
EQLv2 reweights gradients by class frequency.
Federated Loss subsamples the class set per image.

Both are valid attacks on the SAME problem. They CAN compose (use both
simultaneously). For this competition we run them as two independent
single-variable experiments first; if both help, we stack them in E13.

Integration with Ultralytics YOLOv8
-----------------------------------
Same monkey-patch pattern as src/eqlv2_loss.py: replace
`model.model.criterion.bce` with our FederatedBCELoss.

The patch is per-call: for every BCE invocation, build a per-image active
mask and zero out the loss for inactive classes. Per-image because
positive classes vary per image; per-batch active mask is the union.

In Ultralytics' YOLOv8 v8DetectionLoss, the BCE call signature is:
    self.bce(pred_scores, target_scores).sum() / target_scores_sum

`pred_scores` shape: (B, num_anchors, num_classes)
`target_scores` shape: (B, num_anchors, num_classes), one-hot per anchor
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FederatedBCELoss(nn.Module):
    """BCE-with-logits restricted to a per-image federated subset of classes.

    Active classes per image = positive_classes_present_in_image ∪
                              Sample(num_negatives, from negative_classes)

    All classes outside this set get BCE weight = 0.

    Parameters
    ----------
    num_classes : int
        Total class count (32 for FathomNet).
    num_negatives : int
        Negative classes to sample per image. CenterNet2/LVIS uses 50/1000.
        For our 32 classes, 16 is a reasonable default (50% sampling).
        Set to num_classes-1 to disable subsampling (recovers standard BCE).
    class_freq : torch.Tensor or None
        If provided (shape (num_classes,)), negative classes are sampled
        with probability proportional to freq^0.5 (frequency-aware sampling
        from CenterNet2). If None, uniform sampling over negatives.
    reduction : str
        'mean' (default) or 'sum'. Note: mean is averaged over the ACTIVE
        loss elements, not the full tensor — same convention as BCE.
    """

    def __init__(
        self,
        num_classes: int = 32,
        num_negatives: int = 16,
        class_freq: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if not 0 < num_negatives <= num_classes:
            raise ValueError(
                f"num_negatives must be in (0, num_classes], got {num_negatives} (num_classes={num_classes})"
            )
        self.num_classes = num_classes
        self.num_negatives = num_negatives
        self.reduction = reduction
        if class_freq is not None:
            assert class_freq.shape == (num_classes,)
            sampling_weight = (class_freq + 1e-6).sqrt()
            sampling_weight = sampling_weight / sampling_weight.sum()
            self.register_buffer("class_sampling_weight", sampling_weight.float())
        else:
            self.class_sampling_weight = None

    def _build_active_mask(self, target: torch.Tensor) -> torch.Tensor:
        """Return per-image (B, C) bool mask of "active" classes.

        target shape: (B, num_anchors, C). Positive = any anchor has target_c > 0.
        """
        if target.dim() == 3:
            pos_per_image = (target.sum(dim=1) > 0)
        elif target.dim() == 2:
            pos_per_image = (target > 0)
            if pos_per_image.dim() == 1:
                pos_per_image = pos_per_image.unsqueeze(0)
        else:
            raise ValueError(
                f"Cannot infer per-image positive set from target shape {tuple(target.shape)}"
            )

        B, C = pos_per_image.shape
        device = target.device
        active = pos_per_image.clone()

        for b in range(B):
            n_neg_to_sample = min(self.num_negatives,
                                  int((~pos_per_image[b]).sum().item()))
            if n_neg_to_sample <= 0:
                continue
            neg_idx_pool = (~pos_per_image[b]).nonzero(as_tuple=True)[0]
            if self.class_sampling_weight is not None:
                w = self.class_sampling_weight.to(device).index_select(0, neg_idx_pool)
                w = w / w.sum().clamp_min(1e-12)
                sampled_pool_pos = torch.multinomial(w, n_neg_to_sample, replacement=False)
                sampled = neg_idx_pool[sampled_pool_pos]
            else:
                perm = torch.randperm(neg_idx_pool.numel(), device=device)
                sampled = neg_idx_pool[perm[:n_neg_to_sample]]
            active[b, sampled] = True

        return active

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute federated BCE.

        pred shape:   (B, num_anchors, C)  or  (N, C)
        target shape: same as pred.
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
        if pred.dim() == 3:
            B, A, C = pred.shape
            assert C == self.num_classes, f"expected {self.num_classes} classes, got {C}"
            active_per_image = self._build_active_mask(target)
            mask = active_per_image.unsqueeze(1).expand(B, A, C).to(pred.dtype)
        elif pred.dim() == 2:
            N, C = pred.shape
            assert C == self.num_classes
            pos_per_batch = (target.sum(dim=0) > 0).unsqueeze(0)
            active = pos_per_batch.clone()
            n_neg_to_sample = min(self.num_negatives,
                                  int((~pos_per_batch[0]).sum().item()))
            if n_neg_to_sample > 0:
                neg_pool = (~pos_per_batch[0]).nonzero(as_tuple=True)[0]
                if self.class_sampling_weight is not None:
                    w = self.class_sampling_weight.to(pred.device).index_select(0, neg_pool)
                    w = w / w.sum().clamp_min(1e-12)
                    sampled_pool_pos = torch.multinomial(w, n_neg_to_sample, replacement=False)
                    sampled = neg_pool[sampled_pool_pos]
                else:
                    perm = torch.randperm(neg_pool.numel(), device=pred.device)
                    sampled = neg_pool[perm[:n_neg_to_sample]]
                active[0, sampled] = True
            mask = active.expand(N, C).to(pred.dtype)
        else:
            raise ValueError(f"pred dim must be 2 or 3, got {pred.dim()}")

        bce_full = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        masked = bce_full * mask
        if self.reduction == "mean":
            denom = mask.sum().clamp_min(1.0)
            return masked.sum() / denom
        if self.reduction == "sum":
            return masked.sum()
        if self.reduction == "none":
            return masked
        raise ValueError(f"Unknown reduction {self.reduction!r}")


@dataclass
class FederatedLossHandle:
    original_cls_loss: nn.Module
    federated_loss: FederatedBCELoss
    model: object

    def state(self) -> dict:
        return {
            "num_classes": self.federated_loss.num_classes,
            "num_negatives": self.federated_loss.num_negatives,
            "reduction": self.federated_loss.reduction,
            "freq_aware_sampling": self.federated_loss.class_sampling_weight is not None,
        }

    def uninstall(self):
        try:
            loss_fn = self.model.model.criterion
            if hasattr(loss_fn, "bce"):
                loss_fn.bce = self.original_cls_loss
        except Exception:
            pass


def install_federated_loss_on_model(
    model,
    num_classes: int = 32,
    num_negatives: int = 16,
    class_freq: Optional[torch.Tensor] = None,
) -> FederatedLossHandle:
    """Monkey-patch federated BCE into an Ultralytics YOLO model's criterion.

    Same pattern as src/eqlv2_loss.install_eqlv2_on_model.
    """
    loss_fn = model.model.criterion
    if not hasattr(loss_fn, "bce"):
        raise AttributeError(
            "Cannot find model.model.criterion.bce — "
            "is this a valid Ultralytics detection model?"
        )

    original_bce = loss_fn.bce
    fed_loss = FederatedBCELoss(
        num_classes=num_classes,
        num_negatives=num_negatives,
        class_freq=class_freq,
    )
    loss_fn.bce = fed_loss

    return FederatedLossHandle(
        original_cls_loss=original_bce,
        federated_loss=fed_loss,
        model=model,
    )


__all__ = [
    "FederatedBCELoss",
    "FederatedLossHandle",
    "install_federated_loss_on_model",
]
