"""EQLv2 (Equalization Loss v2) for long-tail object detection.

Reference: Tan et al., "Equalization Loss v2: A New Gradient Balance Approach
for Long-tailed Object Detection", CVPR 2021.
https://arxiv.org/abs/2012.08548

The key idea: instead of reweighting the loss values, EQLv2 reweights the
GRADIENTS. For rare classes, the negative gradients (from negative samples)
are suppressed so the model doesn't learn to always predict "not this class"
for a rare category.

Gradient weight for class j at sample i:
    w_j = 1 - beta * (1 - freq_j) * (1 - y_ij)

Where:
    freq_j = frequency of class j in the training set (normalized)
    y_ij = 1 if sample i is positive for class j, 0 otherwise
    beta = control parameter (0 = standard BCE, 1 = full equalization)

Effect: for very rare classes (freq_j ≈ 0), negative sample gradients are
multiplied by (1 - beta), heavily suppressing them. Positive gradients are
always unmodified (w=1 when y=1).

Integration with Ultralytics YOLOv8:
    We monkey-patch the classification head's BCEWithLogitsLoss to use our
    gradient-reweighted version. Same pattern as src/pu_v8_detection_loss.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_frequencies(
    labels_dir: Path,
    num_classes: int = 32,
    smoothing: float = 1e-3,
) -> torch.Tensor:
    """Scan YOLO label files and compute per-class frequency.

    Returns a tensor of shape (num_classes,) where freq[c] = count[c] / total.
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    n_files = 0
    for label_file in sorted(labels_dir.glob("*.txt")):
        n_files += 1
        with label_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if 0 <= cls_id < num_classes:
                        counts[cls_id] += 1

    total = counts.sum()
    if total < 1:
        return torch.ones(num_classes, dtype=torch.float32) / num_classes

    freq = (counts + smoothing) / (total + smoothing * num_classes)
    return freq.float()


class EQLv2Loss(nn.Module):
    """BCE classification loss with EQLv2 gradient reweighting.

    Drop-in replacement for nn.BCEWithLogitsLoss in YOLO's detection head.
    """

    def __init__(
        self,
        class_freq: torch.Tensor,
        beta: float = 0.5,
        reduction: str = "none",
    ):
        super().__init__()
        self.register_buffer("class_freq", class_freq)
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target shape: (batch*anchors, num_classes) or similar
        # Gradient reweighting via the surrogate: modify the target gradient
        # contribution. EQLv2 keeps positive gradients intact and suppresses
        # negative gradients for rare classes.

        freq = self.class_freq.to(pred.device)

        # For negative samples (target=0), weight = 1 - beta * (1 - freq_j)
        # For positive samples (target=1), weight = 1
        neg_weight = 1.0 - self.beta * (1.0 - freq)  # shape (num_classes,)

        # Expand to match pred shape
        while neg_weight.dim() < pred.dim():
            neg_weight = neg_weight.unsqueeze(0)

        # weight = 1 for positives, neg_weight for negatives
        weight = target + (1.0 - target) * neg_weight

        # Standard BCE with logits, then apply gradient weight
        loss = F.binary_cross_entropy_with_logits(
            pred, target, weight=weight, reduction=self.reduction
        )
        return loss


@dataclass
class EQLv2Handle:
    """Handle returned by install_eqlv2_on_model for state tracking."""
    original_cls_loss: nn.Module
    eqlv2_loss: EQLv2Loss
    model: object

    def state(self) -> dict:
        return {
            "beta": self.eqlv2_loss.beta,
            "class_freq_min": float(self.eqlv2_loss.class_freq.min()),
            "class_freq_max": float(self.eqlv2_loss.class_freq.max()),
        }

    def uninstall(self):
        """Restore original loss."""
        try:
            loss_fn = self.model.model.criterion
            if hasattr(loss_fn, "bce"):
                loss_fn.bce = self.original_cls_loss
        except Exception:
            pass


def install_eqlv2_on_model(
    model,
    class_freq: torch.Tensor,
    beta: float = 0.5,
) -> EQLv2Handle:
    """Monkey-patch EQLv2 into an Ultralytics YOLO model's detection loss.

    Replaces the BCE classification loss (model.model.criterion.bce) with
    our EQLv2Loss. Same integration pattern as PU loss.
    """
    inner = model.model
    loss_fn = getattr(inner, "criterion", None)
    if loss_fn is None:
        if hasattr(inner, "init_criterion"):
            inner.criterion = inner.init_criterion()
            loss_fn = inner.criterion
        else:
            raise AttributeError(
                "Cannot find model.model.criterion or init_criterion()."
            )
    if not hasattr(loss_fn, "bce"):
        raise AttributeError(
            "Cannot find model.model.criterion.bce — "
            "is this a valid Ultralytics detection model?"
        )

    original_bce = loss_fn.bce
    orig_reduction = getattr(original_bce, "reduction", "none")
    eqlv2 = EQLv2Loss(class_freq=class_freq, beta=beta, reduction=orig_reduction)
    loss_fn.bce = eqlv2

    return EQLv2Handle(
        original_cls_loss=original_bce,
        eqlv2_loss=eqlv2,
        model=model,
    )
