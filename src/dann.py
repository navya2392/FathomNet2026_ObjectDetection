"""Domain-Adversarial Neural Network (DANN) primitives for FathomNet 2026.

Reference: Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation",
ICML 2015 (https://arxiv.org/abs/1409.7495). Detection-domain extensions:
Saito et al., "Strong-Weak Distribution Alignment for Adaptive Object
Detection" (CVPR 2019).

The competition has a clear domain shift: training images come from SOI/NOAA
ROVs while the test set is MBARI VARS (different lighting, water column,
camera, fauna distribution). DANN attacks this directly: a Gradient Reversal
Layer (GRL) sits between the backbone and a small "domain classifier"; the
classifier is trained to tell source from target images, while GRL forces the
backbone to fool it. The result: backbone features become domain-invariant.

We provide three reusable pieces:

  * GradReverseFunction / grad_reverse() — the canonical autograd op
    that is identity on the forward pass and multiplies gradients by
    -lambda on the backward pass.

  * DomainClassifier — a small CNN head (3x3 conv -> GAP -> 2 FC) that
    predicts source(0) vs target(1) from intermediate backbone feature
    maps.

  * dann_lambda_schedule — the original ICML 2015 schedule
    lambda(p) = 2 / (1 + exp(-gamma * p)) - 1, where p in [0,1] is the
    relative training progress. Lambda ramps from 0 -> 1 across training,
    keeping early epochs DANN-free so the detector can learn the source
    task before domain adaptation kicks in.

The pieces here are framework-agnostic: integration with the Ultralytics
trainer happens in scripts/train_dann.py (which mixes labeled SOI/NOAA
batches with unlabeled MBARI test batches and adds a domain term to the
total loss).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradReverseFunction(torch.autograd.Function):
    """Identity forward, gradient-reversed (and scaled by lambda) backward.

    The standard DANN trick: place this between a feature extractor and a
    domain classifier. The classifier minimizes its loss as usual; gradients
    flowing back into the extractor are negated and scaled by lambda, which
    makes the extractor maximize the classifier's loss — i.e. produce
    features that are indistinguishable across domains.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to `x` with scaling factor `lambda_`."""
    return GradReverseFunction.apply(x, lambda_)


class GRL(nn.Module):
    """Gradient-reversal as a `nn.Module` (handy for `nn.Sequential`).

    The lambda value is read from `self.lambda_` at every forward call;
    callers update it from a schedule in their training loop.
    """

    def __init__(self, lambda_: float = 0.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambda_)


# ---------------------------------------------------------------------------
# Domain Classifier
# ---------------------------------------------------------------------------

class DomainClassifier(nn.Module):
    """Small CNN that predicts source(0)/target(1) from a feature map.

    Designed to attach to a mid-backbone feature map (e.g. SPPF output of
    YOLOv8x: shape (B, 640, H/32, W/32)). Architecture:

        Conv 3x3 -> BN -> ReLU
        Conv 3x3 -> BN -> ReLU
        Global avg pool
        FC -> ReLU -> Dropout
        FC -> 2 logits

    The classifier is intentionally small: we want the *features* to do
    the heavy lifting via gradient reversal, not the classifier itself.
    Bigger heads dominate the adversarial game and stop the backbone from
    receiving useful gradients.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 2)

        # Initialize all conv/linear with kaiming-normal; BN with default
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
# DANN Head (GRL + domain classifier glued together)
# ---------------------------------------------------------------------------

class DANNHead(nn.Module):
    """GRL + DomainClassifier as a single attachable module.

    Usage:
        head = DANNHead(in_channels=640)
        # in your training loop:
        head.grl.set_lambda(lambda_now)
        feat = backbone(x)         # (B, C, H, W) — source+target mixed
        logits = head(feat)        # (B, 2) — source(0) vs target(1)
        d_loss = F.cross_entropy(logits, domain_labels)
        total_loss = det_loss + d_loss
        total_loss.backward()
    """

    def __init__(self, in_channels: int, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.grl = GRL(lambda_=0.0)
        self.classifier = DomainClassifier(
            in_channels=in_channels, hidden=hidden, dropout=dropout
        )

    def set_lambda(self, lambda_: float) -> None:
        self.grl.set_lambda(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.grl(x))


# ---------------------------------------------------------------------------
# Lambda schedule
# ---------------------------------------------------------------------------

@dataclass
class LambdaSchedule:
    """ICML 2015 sigmoid schedule: lambda(p) = 2/(1+exp(-gamma*p)) - 1.

    `p` is the relative training progress in [0, 1] (e.g. epoch / total_epochs
    or step / total_steps). With gamma=10 (canonical), lambda ramps from
    0.0 at p=0 to ~0.99 at p=0.7 and saturates at 1.0.

    `max_lambda` lets you cap the final value (often 0.1 for detection,
    where the detector loss should dominate).
    """

    gamma: float = 10.0
    max_lambda: float = 1.0

    def __call__(self, progress: float) -> float:
        progress = max(0.0, min(1.0, float(progress)))
        v = 2.0 / (1.0 + math.exp(-self.gamma * progress)) - 1.0
        return self.max_lambda * v


def dann_lambda_schedule(progress: float, gamma: float = 10.0, max_lambda: float = 1.0) -> float:
    """Functional form of the ICML 2015 sigmoid schedule. See LambdaSchedule."""
    return LambdaSchedule(gamma=gamma, max_lambda=max_lambda)(progress)


# ---------------------------------------------------------------------------
# Backbone hook installation (for Ultralytics YOLOv8/v11)
# ---------------------------------------------------------------------------

@dataclass
class FeatureHookHandle:
    """Holds the captured feature map and a torch hook handle (for cleanup)."""

    layer_idx: int
    in_channels: int
    feature: Optional[torch.Tensor] = None
    _torch_handle: Optional[object] = None

    def remove(self) -> None:
        if self._torch_handle is not None:
            self._torch_handle.remove()
            self._torch_handle = None

    def get(self) -> torch.Tensor:
        if self.feature is None:
            raise RuntimeError(
                "FeatureHookHandle: no feature captured yet — call this after a "
                "forward pass through the model."
            )
        return self.feature


def install_feature_hook(model: nn.Module, layer_idx: int = 9) -> FeatureHookHandle:
    """Register a forward hook that captures the output of `model.model[layer_idx]`.

    For YOLOv8x, layer 9 is the SPPF block (output channels = 640) at
    the end of the backbone — the canonical place to attach a domain
    classifier (the literature attaches at the deepest backbone feature).

    `model` here is the inner Ultralytics `DetectionModel` (i.e.
    `yolo_instance.model`). `model.model` is the `nn.Sequential` of
    layers; we hook one of them.

    Returns a handle. After every forward pass, `handle.feature` holds
    the captured tensor (B, C, H, W). Call `handle.remove()` to detach
    the hook (e.g. before saving the checkpoint).
    """
    seq = getattr(model, "model", None)
    if seq is None or not hasattr(seq, "__getitem__"):
        raise AttributeError(
            "install_feature_hook: expected `model.model` to be a Sequential. "
            "Pass the inner DetectionModel (yolo.model), not the YOLO wrapper."
        )

    layer = seq[layer_idx]
    handle = FeatureHookHandle(layer_idx=layer_idx, in_channels=0)

    def _hook(_module, _inputs, output):
        handle.feature = output
        if handle.in_channels == 0 and output.dim() == 4:
            handle.in_channels = output.shape[1]

    handle._torch_handle = layer.register_forward_hook(_hook)
    return handle


__all__ = [
    "GradReverseFunction",
    "grad_reverse",
    "GRL",
    "DomainClassifier",
    "DANNHead",
    "LambdaSchedule",
    "dann_lambda_schedule",
    "FeatureHookHandle",
    "install_feature_hook",
]
