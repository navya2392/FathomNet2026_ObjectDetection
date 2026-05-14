"""Hierarchical auxiliary classification head for FathomNet 2026.

E11 in the v7 master plan: a small classification head that predicts the
super-class (taxonomic group) of organisms present in each image, attached
to a mid-backbone feature map. Acts as a regularizer that biases the
backbone toward learning features that capture coarse semantic structure
(cnidarian vs echinoderm vs fish vs ...), which helps both class imbalance
(rare classes share gradient signal with their group siblings) and domain
shift (group-level features are typically more domain-invariant than
species-level features).

This is distinct from `src/hierarchical_loss.py`, which replaces the
classification loss with a hierarchy-aware version. The aux-head approach
here is ADDITIVE (a separate head + loss term), keeping the standard
detection loss untouched. Last year's FathomNet 2025 winners reported
+0.005-0.020 mAP from a hierarchical aux head.

Design choice: image-level multi-label
--------------------------------------
For each training image, we compute a 9-dim multi-hot vector over
taxonomic groups (1 if any annotation in the image belongs to that group,
0 otherwise) and use BCE-with-logits. This is simpler than per-instance
group prediction (which would need to be routed through the detection
head's matched anchors) while still providing a strong regularization
signal — and it elegantly handles images with multiple species across
groups. Per-image dominant-class labels (single CE) lose multi-species
information; per-instance group prediction is much more invasive.

Architecture
------------
    Conv 3x3 -> BN -> ReLU
    Conv 3x3 -> BN -> ReLU
    Global avg pool
    Linear -> ReLU -> Dropout
    Linear -> num_groups logits

Same shape as DANN's DomainClassifier, but:
* Output dim is num_groups (9 for FathomNet) not 2.
* No GRL on input (we WANT this gradient flowing back into the backbone).
* Trained with BCE-with-logits (multi-label) not CE (single-class).

Usage
-----
    from src.hier_aux_head import HierAuxHead, image_group_multihot
    from src.dann import install_feature_hook  # reused

    head = HierAuxHead(in_channels=640, num_groups=9).to(device)
    feat_hook = install_feature_hook(yolo.model, layer_idx=9)

    # in your training loop, after running the model on a batch:
    feat = feat_hook.feature                               # (B, 640, H/32, W/32)
    group_logits = head(feat)                              # (B, 9)
    group_targets = image_group_multihot(batch, class_to_group, num_groups=9)
    aux_loss = F.binary_cross_entropy_with_logits(group_logits, group_targets)
    total = det_loss + group_weight * aux_loss
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierAuxHead(nn.Module):
    """Small CNN that predicts taxonomic-group multi-hot from a feature map.

    Parameters
    ----------
    in_channels : int
        Channel count of the input feature map (640 for YOLOv8x SPPF output).
    num_groups : int
        Number of taxonomic groups. 9 for FathomNet 2026 per
        `configs/taxonomy.py`.
    hidden : int
        Width of the intermediate conv + linear layers.
    dropout : float
        Dropout applied between FC layers.
    """

    def __init__(
        self,
        in_channels: int,
        num_groups: int,
        hidden: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_groups)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool(x).flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Label aggregation
# ---------------------------------------------------------------------------

def image_group_multihot(
    cls: torch.Tensor,
    batch_idx: torch.Tensor,
    class_to_group: torch.Tensor,
    batch_size: int,
    num_groups: int = 9,
) -> torch.Tensor:
    """Aggregate per-instance class labels into a per-image multi-hot vector.

    Parameters
    ----------
    cls : (N_total,) or (N_total, 1) class indices (float or int).
        N_total = sum of object counts across all images in the batch.
    batch_idx : (N_total,) which image each annotation belongs to.
        Values in [0, batch_size).
    class_to_group : (num_classes,) long tensor mapping class_idx -> group_idx.
        Build via `src.hierarchical_loss.build_class_to_group_tensor`.
    batch_size : int
        Number of images in the batch (so empty-image rows are still emitted).
    num_groups : int
        Total number of groups (9 for FathomNet).

    Returns
    -------
    multihot : (batch_size, num_groups) float tensor in {0., 1.}.
        multihot[i, g] = 1 iff image i contains at least one annotation
        whose class maps to group g.

    Notes
    -----
    Empty images get all-zero rows. If you train with `bce_with_logits`,
    these rows contribute pure-negative gradient (model penalized for
    predicting any group on an empty image), which is the right signal.
    Empty-image rate in FathomNet train is ~0%, but val/inference sees
    them, so this is just for safety.
    """
    cls = cls.flatten().long()
    batch_idx = batch_idx.flatten().long()
    out = torch.zeros(batch_size, num_groups, device=cls.device, dtype=torch.float32)
    if cls.numel() == 0:
        return out

    # Map class -> group
    group_idx = class_to_group.to(cls.device)[cls]  # (N_total,)
    # Linear index: row*num_groups + col, then unique-set bits to 1.
    flat = batch_idx * num_groups + group_idx
    out.view(-1).scatter_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    return out


__all__ = ["HierAuxHead", "image_group_multihot"]
