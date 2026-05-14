"""E26 — Multi-scale context rescorer.

A box-level classifier that re-ranks the predicted class probability of each
detection by examining crops at multiple context scales (1.0x, 2.0x, 4.0x).
The detector tells us "there's a box here at xywh; my best guess is class C
with score s"; this module *separately* asks "given the same box plus its
surrounding pixels, what does a 32-way classifier say?". The two scores are
blended at apply time.

Why this is high-EV given our findings
--------------------------------------
- Every train-time intervention we've tried hurt LB (E5/E6/E9/T1).
- E27 (cross-arch ensemble) is our other top bet but it requires T2/T3 to
  finish.
- E26 is INDEPENDENT of all that. It only needs the train fold's GT labels
  (already on pod) and an existing submission CSV. It re-scores boxes
  without ever touching the detector. Worst case: alpha=1.0 → no change.
- FathomNet 2025 winning teams used context-aware re-rankers.

Module contents
---------------
- ``CropDataset``: yields (image_tensor, class_idx) tuples by reading YOLO
  labels from a fold's train.txt. Each label is one box; we crop it at
  ``crop_scale * box`` and resize to ``image_size``.
- ``build_model(...)``: wraps a `timm` backbone with a 32-class head.
- ``crop_box_with_context(...)``: pure utility, used by both training and
  apply scripts to extract a single (potentially padded) crop. Identical
  geometry at train and apply time keeps the train/inference distributions
  matched (avoids the F-017 fiasco from inference-only preproc).
- ``blend_scores(...)``: convex blend of detector score and classifier prob.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CROP_SCALE = 2.0
DEFAULT_NUM_CLASSES = 32

# ImageNet stats (timm/torchvision default; the timm model itself sets
# default_cfg['mean']/['std'] but we standardize once here for portability).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_box_with_context(
    bgr: np.ndarray,
    x_topleft: float,
    y_topleft: float,
    box_w: float,
    box_h: float,
    crop_scale: float = DEFAULT_CROP_SCALE,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Crop a box's neighborhood with reflection padding; return 224x224 BGR.

    ``crop_scale=1.0`` returns the tight box; ``crop_scale=2.0`` doubles
    each side so the box's surroundings are visible (helpful for benthic
    organisms whose context = substrate texture). Coordinates are in pixel
    units of the source image.

    Out-of-frame regions are ``cv2.copyMakeBorder``-padded with REFLECT_101
    so the resized crop never contains "frame artifact" zero pixels (the
    classifier hates those).
    """
    H, W = bgr.shape[:2]
    cx = x_topleft + box_w / 2.0
    cy = y_topleft + box_h / 2.0
    half_w = box_w / 2.0 * crop_scale
    half_h = box_h / 2.0 * crop_scale
    x1 = int(round(cx - half_w))
    y1 = int(round(cy - half_h))
    x2 = int(round(cx + half_w))
    y2 = int(round(cy + half_h))

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)
    if (pad_left + pad_top + pad_right + pad_bottom) > 0:
        bgr_padded = cv2.copyMakeBorder(
            bgr, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
        x1 += pad_left; x2 += pad_left
        y1 += pad_top;  y2 += pad_top
    else:
        bgr_padded = bgr

    crop = bgr_padded[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)


def to_tensor(bgr_crop: np.ndarray) -> torch.Tensor:
    """Convert a BGR uint8 crop to a normalized RGB float tensor (CHW)."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


class CropDataset(Dataset):
    """Yields (image_tensor, class_idx) per GT box from a YOLO fold list.

    Parameters
    ----------
    fold_train_txt :
        Path to a file with one image path per line (e.g. data/folds/fold0_train.txt).
        Labels are auto-discovered by replacing '/images/' -> '/labels/'.
    crop_scale :
        Crop multiplier (1.0 = tight box, 2.0 = double, etc.). Use a single
        scale for training; multi-scale is applied at inference time only
        (averaging classifier probs across scales).
    image_size :
        Output square crop size (default 224).
    augment :
        If True, apply mild train-time augmentation (random horizontal flip
        + colorjitter). Marine organisms often have natural orientation, so
        we keep flips MILD (p=0.3) — and only horizontal, never vertical.
    label_xy_in_pixels :
        Standard YOLO labels are normalized [0,1] xc/yc/w/h. If your fold
        labels are pixels instead, set to True.
    """

    def __init__(
        self,
        fold_train_txt: Path,
        crop_scale: float = DEFAULT_CROP_SCALE,
        image_size: int = DEFAULT_IMAGE_SIZE,
        augment: bool = True,
        label_xy_in_pixels: bool = False,
    ) -> None:
        super().__init__()
        self.crop_scale = float(crop_scale)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.label_xy_in_pixels = bool(label_xy_in_pixels)
        self._index = self._build_index(fold_train_txt)

    @staticmethod
    def _label_path_for_image(img_path: Path) -> Path:
        s = str(img_path)
        if "/images/" in s:
            s2 = s.replace("/images/", "/labels/")
        elif "\\images\\" in s:
            s2 = s.replace("\\images\\", "\\labels\\")
        else:
            s2 = s
        return Path(s2).with_suffix(".txt")

    def _build_index(self, fold_train_txt: Path) -> list[tuple[Path, Path, int, list[float]]]:
        if not fold_train_txt.exists():
            raise FileNotFoundError(f"fold list not found: {fold_train_txt}")
        index: list[tuple[Path, Path, int, list[float]]] = []
        for raw_line in fold_train_txt.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            img_path = Path(line)
            label_path = self._label_path_for_image(img_path)
            if not label_path.exists():
                continue
            for label_line in label_path.read_text(encoding="utf-8").splitlines():
                parts = label_line.strip().split()
                if len(parts) < 5:
                    continue
                cls_idx = int(float(parts[0]))
                xc, yc, bw, bh = (float(x) for x in parts[1:5])
                if bw <= 0 or bh <= 0:
                    continue
                index.append((img_path, label_path, cls_idx, [xc, yc, bw, bh]))
        if not index:
            raise RuntimeError(f"No (image, label) pairs found from {fold_train_txt}")
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, _label_path, cls_idx, (xc, yc, bw, bh) = self._index[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            return torch.zeros(3, self.image_size, self.image_size), cls_idx
        H, W = bgr.shape[:2]
        if self.label_xy_in_pixels:
            cx_px, cy_px, bw_px, bh_px = xc, yc, bw, bh
        else:
            cx_px = xc * W; cy_px = yc * H
            bw_px = bw * W; bh_px = bh * H
        x_topleft = cx_px - bw_px / 2.0
        y_topleft = cy_px - bh_px / 2.0
        crop = crop_box_with_context(
            bgr, x_topleft, y_topleft, bw_px, bh_px,
            crop_scale=self.crop_scale, image_size=self.image_size,
        )
        if self.augment:
            if np.random.rand() < 0.3:
                crop = cv2.flip(crop, 1)
            if np.random.rand() < 0.3:
                # Mild HSV jitter — marine images vary heavily in lighting.
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[..., 1] *= np.random.uniform(0.85, 1.15)
                hsv[..., 2] *= np.random.uniform(0.85, 1.15)
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return to_tensor(crop), cls_idx


def build_model(
    backbone: str = "convnext_tiny",
    num_classes: int = DEFAULT_NUM_CLASSES,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Build a timm classification backbone with a num_classes head.

    Default: ConvNeXt-Tiny (~28M params, ~5 ms / img on a 5090). Other
    sensible choices: 'resnet50' (slightly worse but ~2x cheaper) or
    'convnext_small' (~50M params, +0.5-1 acc typically).
    """
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "timm required for the context rescorer. "
            "`pip install timm` (already pinned in requirements.txt)."
        ) from exc
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


def blend_scores(
    detector_score: float,
    classifier_prob: float,
    alpha: float = 0.5,
    method: str = "geo",
) -> float:
    """Blend the detector's box score and the classifier's class prob.

    method='geo' (default):
        ``s_new = (s_det)^alpha * (p_cls)^(1-alpha)``
        Geometric blend treats both as probabilities and is symmetric in
        log-space. Use alpha=1.0 to recover the original detector score.

    method='arith':
        ``s_new = alpha * s_det + (1 - alpha) * p_cls``
        Linear blend; less sensitive to small probabilities.

    method='multiply':
        ``s_new = s_det * p_cls``
        Strict multiplicative — equivalent to alpha=0.5 with method='geo'
        when (s_det, p_cls) are probabilities. Most aggressive re-ranking.
    """
    s = max(0.0, min(1.0, float(detector_score)))
    p = max(0.0, min(1.0, float(classifier_prob)))
    a = max(0.0, min(1.0, float(alpha)))
    if method == "geo":
        if s <= 0.0 or p <= 0.0:
            return 0.0
        return (s ** a) * (p ** (1.0 - a))
    if method == "arith":
        return a * s + (1.0 - a) * p
    if method == "multiply":
        return s * p
    raise ValueError(f"unknown blend method: {method}")
