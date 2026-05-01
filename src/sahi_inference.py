"""SAHI (Slicing Aided Hyper Inference) for the Phase 7 EXP 7.2 experiment.

Block reference: I.2 in master_checklist.txt.

Reference
---------
Akyon, Fatih Cagatay et al. "Slicing Aided Hyper Inference and Fine-tuning
for Small Object Detection." 2022. https://arxiv.org/abs/2202.06934
Library: https://github.com/obss/sahi

What problem does SAHI solve
----------------------------
Object detectors trained at imgsz=640 see small objects (e.g., a single
amphipod in a 1920x1080 deep-sea photo) as 10-30 pixels wide -- below
their effective resolution. SAHI fixes this by tiling the high-res
image into overlapping 640x640 patches, running inference on each
patch (the small object is now 30-90 pixels, easily detectable), and
merging the patch predictions back into image coordinates.

The "Slicing Aided" part is the standard tiling. The "Hyper" is the
score recalibration that prevents the per-patch detections from being
overconfident at the merge step.

The resolution-conditional strategy (per v5 master plan)
---------------------------------------------------------
NOT every test image needs SAHI. Test set has two regimes:

- LOW RES (~14% of test, ~720x486): The image is already smaller than
  most training crops. Tiling would actively HURT (each tile would be
  context-starved). Use plain Ultralytics inference at imgsz=640.
- HIGH RES (~86% of test, >= 1024x1024): Plain inference under-detects
  small objects. Use SAHI with 640x640 patches, 0.2 overlap.

This module's `predict_resolution_conditional()` makes the plain-vs-SAHI
choice per-image automatically based on image dimensions.

Cost
----
SAHI tiling is ~Nx slower than plain inference where N is the number
of patches per image. For a 1920x1080 image with 640x640 patches and
0.2 overlap, N ~= 3x3 = 9. So SAHI is ~9x slower.

Test set: 1,425 images. Plain inference on RTX 4090: ~30 sec total.
SAHI inference: ~5 min total. Cheap.

Status (May 1, 2026)
--------------------
WRITTEN. Has NOT been run end-to-end -- needs `pip install sahi` and
a Phase 6 model checkpoint. Phase 7 EXP 7.2 (I.4) is the first
invocation.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class SahiConfig:
    """SAHI inference configuration knobs.

    Defaults match the v5 plan recommendations + SAHI paper's COCO settings.
    """

    slice_height: int = 640
    slice_width: int = 640
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2
    perform_standard_pred: bool = True
    """If True, also run plain (non-tiled) inference and merge with sliced.
    Recommended -- the plain pass catches LARGE objects that span multiple tiles
    poorly. The merge keeps best-of-both-worlds detections."""

    postprocess_type: str = "GREEDYNMM"
    """Merge strategy. 'GREEDYNMM' (default in SAHI) is greedy
    non-maximum-merging; 'NMS' is plain NMS; 'NMM' is more aggressive."""

    postprocess_match_metric: str = "IOS"
    """Match metric for the merge. 'IOS' = intersection over smaller box
    (tolerant for partial overlaps); 'IOU' = intersection over union."""

    postprocess_match_threshold: float = 0.5
    """Match threshold for the merge."""

    high_res_threshold_px: int = 1024
    """Image is "HIGH RES" if max(h, w) >= this. Plain inference for
    images smaller than this (avoids context-starvation in patches)."""

    confidence_threshold: float = 0.05
    """Drop SAHI predictions below this conf BEFORE merge."""


def load_sahi_model(weights_path: str | Path, *, image_size: int = 640, device: str = "cuda:0"):
    """Lazy-import SAHI and load an Ultralytics YOLO checkpoint as a SAHI model."""
    try:
        from sahi import AutoDetectionModel
    except ImportError as exc:
        raise RuntimeError(
            "sahi package required. Install via `pip install sahi`."
        ) from exc

    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights_path),
        confidence_threshold=0.0,
        image_size=image_size,
        device=device,
    )


def _to_xywh_score_label(prediction_list) -> list[tuple[tuple[float, float, float, float], float, int]]:
    """Convert SAHI's PredictionResult.object_prediction_list to our (xywh, score, label) tuples."""
    out = []
    for op in prediction_list:
        bbox = op.bbox
        x = float(bbox.minx)
        y = float(bbox.miny)
        w = float(bbox.maxx - bbox.minx)
        h = float(bbox.maxy - bbox.miny)
        score = float(op.score.value)
        label = int(op.category.id)
        out.append(((x, y, w, h), score, label))
    return out


def predict_one_sahi(
    image_path: str | Path,
    sahi_model,
    cfg: SahiConfig,
) -> list[tuple[tuple[float, float, float, float], float, int]]:
    """Run SAHI sliced prediction on one image. Returns (xywh, score, label) list."""
    try:
        from sahi.predict import get_sliced_prediction
    except ImportError as exc:
        raise RuntimeError("sahi package required for predict_one_sahi") from exc

    result = get_sliced_prediction(
        image=str(image_path),
        detection_model=sahi_model,
        slice_height=cfg.slice_height,
        slice_width=cfg.slice_width,
        overlap_height_ratio=cfg.overlap_height_ratio,
        overlap_width_ratio=cfg.overlap_width_ratio,
        perform_standard_pred=cfg.perform_standard_pred,
        postprocess_type=cfg.postprocess_type,
        postprocess_match_metric=cfg.postprocess_match_metric,
        postprocess_match_threshold=cfg.postprocess_match_threshold,
    )
    return [
        (xywh, score, label)
        for xywh, score, label in _to_xywh_score_label(result.object_prediction_list)
        if score >= cfg.confidence_threshold
    ]


def predict_one_plain(
    image_path: str | Path,
    sahi_model,
    cfg: SahiConfig,
) -> list[tuple[tuple[float, float, float, float], float, int]]:
    """Run a plain (non-tiled) Ultralytics-via-SAHI prediction on one image."""
    try:
        from sahi.predict import get_prediction
    except ImportError as exc:
        raise RuntimeError("sahi package required for predict_one_plain") from exc

    result = get_prediction(image=str(image_path), detection_model=sahi_model)
    return [
        (xywh, score, label)
        for xywh, score, label in _to_xywh_score_label(result.object_prediction_list)
        if score >= cfg.confidence_threshold
    ]


def predict_resolution_conditional(
    image_path: str | Path,
    image_h: int,
    image_w: int,
    sahi_model,
    cfg: Optional[SahiConfig] = None,
) -> tuple[list[tuple[tuple[float, float, float, float], float, int]], str]:
    """Choose SAHI (tiled) for high-res or plain for low-res.

    Returns (predictions, mode_used) where mode_used is 'sahi' or 'plain'.
    """
    if cfg is None:
        cfg = SahiConfig()
    if max(image_h, image_w) >= cfg.high_res_threshold_px:
        return predict_one_sahi(image_path, sahi_model, cfg), "sahi"
    return predict_one_plain(image_path, sahi_model, cfg), "plain"


__all__ = [
    "SahiConfig",
    "load_sahi_model",
    "predict_one_sahi",
    "predict_one_plain",
    "predict_resolution_conditional",
]
