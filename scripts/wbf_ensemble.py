"""Weighted Box Fusion (WBF) ensemble for the Phase 7 EXP 7.1 experiment.

Block reference: I.2 in master_checklist.txt.

Reference
---------
Solovyev, Roman et al. "Weighted boxes fusion: Ensembling boxes from
different object detection models." Image and Vision Computing, 2021.
https://arxiv.org/abs/1910.13302

What problem does WBF solve
---------------------------
Phase 6 produces 15 model checkpoints (5 folds x 3 architectures:
yolo11l, RT-DETR-X, best-marine-init). Each predicts on the test set;
we get 15 candidate bounding boxes per object instance. Standard NMS
picks ONE of the 15 and discards the rest. WBF AVERAGES them, which:

1. Reduces localization noise (15 noisy estimates of the same edge
   position average out).
2. Improves mAP at higher IoU thresholds (where fine box accuracy
   matters most -- exactly where COCO mAP@[.50:.95] punishes us).

WBF is the de-facto standard for COCO ensembles and is used by every
recent winning submission to detection competitions.

Per the v5.5 strategy, we expect WBF to add +0.05 - +0.10 mAP@[.50:.95]
on top of the best single model. This single block of code is plausibly
the second-largest mAP contributor in the entire pipeline (after the
multi-init bake-off).

Inputs
------
This script reads multiple submission CSVs in our standard 8-column
format and fuses them into one output CSV. Each input CSV is one
"model" in the ensemble; you can weight them with --weights.

Format reminder (`docs/sample_submission.template.csv`):

    annotation_id, image_id, category_id, x_min, y_min, w, h, score

The 8th column (`score`) is the per-box confidence, used as the WBF
input score. The output preserves the same 8-column schema.

Why use the `ensemble_boxes` pip package
----------------------------------------
WBF has a few subtle correctness traps (cluster matching tie-breaks,
score thresholding semantics, per-class vs cross-class fusion). The
`ensemble_boxes` package is the reference implementation maintained
by Solovyev (the paper's author). We use it directly to avoid silently
diverging from the paper.

Status (May 1, 2026)
--------------------
SCRIPT WRITTEN. Has NOT yet been run end-to-end -- requires Phase 6
ensemble checkpoints. Phase 7 EXP 7.1 (I.2) is the first invocation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.submit import validate_submission


def _load_submission_csv(path: Path) -> pd.DataFrame:
    """Read an 8-column submission CSV and validate the columns.

    Accepts EITHER schema:
      * Internal: annotation_id, image_id, category_id, x_min, y_min, w, h, score
      * Kaggle:   annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_width, bbox_height, score
                  (or without annotation_id, in which case it's synthesized)
    Returns the internal schema (renamed) for downstream code.
    """
    df = pd.read_csv(path)
    rename_map = {
        "bbox_x": "x_min",
        "bbox_y": "y_min",
        "bbox_width": "w",
        "bbox_height": "h",
    }
    df = df.rename(columns=rename_map)
    if "annotation_id" not in df.columns:
        df = df.copy()
        df["annotation_id"] = range(1, len(df) + 1)
    required = ["annotation_id", "image_id", "category_id", "x_min", "y_min", "w", "h", "score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing} (after rename); "
                         f"present: {list(df.columns)}")
    return df[required].copy()


def _normalize_boxes_for_image(
    df: pd.DataFrame, image_w: int, image_h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a per-image DataFrame to WBF inputs (boxes in xyxy, normalized to [0, 1])."""
    boxes_xyxy = np.stack([
        df["x_min"].values / image_w,
        df["y_min"].values / image_h,
        (df["x_min"].values + df["w"].values) / image_w,
        (df["y_min"].values + df["h"].values) / image_h,
    ], axis=1).clip(0.0, 1.0)
    scores = df["score"].values.astype(np.float32)
    labels = df["category_id"].values.astype(np.int64)
    return boxes_xyxy, scores, labels


def _denormalize_boxes(boxes_xyxy_norm: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    """Invert normalization: WBF output (xyxy, normalized) -> our submission (xywh, pixels)."""
    abs_xyxy = boxes_xyxy_norm.copy()
    abs_xyxy[:, 0] *= image_w
    abs_xyxy[:, 2] *= image_w
    abs_xyxy[:, 1] *= image_h
    abs_xyxy[:, 3] *= image_h
    out = np.stack([
        abs_xyxy[:, 0],
        abs_xyxy[:, 1],
        abs_xyxy[:, 2] - abs_xyxy[:, 0],
        abs_xyxy[:, 3] - abs_xyxy[:, 1],
    ], axis=1)
    return out


def _build_image_size_lookup(test_json_path: Path) -> dict[int, tuple[int, int]]:
    """image_id -> (width, height) from test_dataset.json."""
    with test_json_path.open() as f:
        coco = json.load(f)
    out: dict[int, tuple[int, int]] = {}
    for img in coco["images"]:
        out[int(img["id"])] = (int(img["width"]), int(img["height"]))
    return out


def fuse_submissions(
    submission_paths: list[Path],
    test_json_path: Path,
    *,
    weights: Optional[list[float]] = None,
    iou_threshold: float = 0.55,
    skip_box_threshold: float = 0.001,
    conf_type: str = "avg",
    verbose: bool = True,
) -> pd.DataFrame:
    """Fuse multiple submission CSVs into one DataFrame using WBF.

    Parameters
    ----------
    submission_paths :
        2 or more paths to 8-column submission CSVs.
    test_json_path :
        Path to test_dataset.json (needed for image dimensions).
    weights :
        Per-submission weights. Default: equal weights.
    iou_threshold :
        Minimum IoU to add a prediction to an existing cluster. Default
        0.55 (paper recommendation for COCO-style 0.5-0.95 evaluation).
    skip_box_threshold :
        Drop predictions with score < this BEFORE fusion (speedup). Default 0.001.
    conf_type :
        How to compute fused cluster confidence. 'avg' (default) is the
        paper recommendation; 'max' makes the ensemble overconfident.
    verbose :
        Print per-image fusion summary.

    Returns
    -------
    pd.DataFrame
        Fused predictions in the 8-column submission format. annotation_id
        is renumbered starting at 1.
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError as exc:
        raise RuntimeError(
            "ensemble_boxes package required. Install via "
            "`pip install ensemble-boxes`."
        ) from exc

    if len(submission_paths) < 2:
        raise ValueError("Need at least 2 submissions to fuse")
    if weights is None:
        weights = [1.0] * len(submission_paths)
    if len(weights) != len(submission_paths):
        raise ValueError(f"weights length ({len(weights)}) != submissions ({len(submission_paths)})")

    image_size = _build_image_size_lookup(test_json_path)

    submissions = [_load_submission_csv(p) for p in submission_paths]
    all_image_ids = set()
    for s in submissions:
        all_image_ids.update(int(i) for i in s["image_id"].unique())
    all_image_ids = sorted(all_image_ids)
    if verbose:
        print(f"Fusing {len(submissions)} submissions with weights {weights} "
              f"across {len(all_image_ids):,} images")

    rows_out: list[dict] = []
    next_ann_id = 1
    skipped_no_size = 0

    for image_id in all_image_ids:
        if image_id not in image_size:
            skipped_no_size += 1
            continue
        w_img, h_img = image_size[image_id]

        boxes_per_model = []
        scores_per_model = []
        labels_per_model = []
        for s in submissions:
            sub_img = s[s["image_id"] == image_id]
            if len(sub_img) == 0:
                boxes_per_model.append(np.zeros((0, 4), dtype=np.float32))
                scores_per_model.append(np.zeros((0,), dtype=np.float32))
                labels_per_model.append(np.zeros((0,), dtype=np.int64))
                continue
            b, sc, lb = _normalize_boxes_for_image(sub_img, w_img, h_img)
            boxes_per_model.append(b)
            scores_per_model.append(sc)
            labels_per_model.append(lb)

        if all(len(b) == 0 for b in boxes_per_model):
            continue

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_per_model,
            scores_per_model,
            labels_per_model,
            weights=weights,
            iou_thr=iou_threshold,
            skip_box_thr=skip_box_threshold,
            conf_type=conf_type,
        )

        if len(fused_boxes) == 0:
            continue

        denorm = _denormalize_boxes(fused_boxes, w_img, h_img)
        for k in range(len(fused_boxes)):
            x, y, ww, hh = denorm[k]
            rows_out.append({
                "annotation_id": next_ann_id,
                "image_id": image_id,
                "category_id": int(fused_labels[k]),
                "x_min": float(x),
                "y_min": float(y),
                "w": float(ww),
                "h": float(hh),
                "score": float(fused_scores[k]),
            })
            next_ann_id += 1

    if verbose and skipped_no_size > 0:
        print(f"WARNING: skipped {skipped_no_size} image_ids missing from {test_json_path}")

    return pd.DataFrame(rows_out)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Weighted Box Fusion ensemble of multiple submission CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--submissions",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more 8-column submission CSVs to fuse.",
    )
    p.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Per-submission weights (defaults to equal).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    p.add_argument(
        "--test-json",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "test_dataset.json",
        help="Path to test_dataset.json (for image dimensions).",
    )
    p.add_argument("--iou-threshold", type=float, default=0.55)
    p.add_argument("--skip-box-threshold", type=float, default=0.001)
    p.add_argument(
        "--conf-type",
        choices=("avg", "max", "box_and_model_avg", "absent_model_aware_avg"),
        default="avg",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the post-write validate_submission() call.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fused = fuse_submissions(
        submission_paths=args.submissions,
        test_json_path=args.test_json,
        weights=args.weights,
        iou_threshold=args.iou_threshold,
        skip_box_threshold=args.skip_box_threshold,
        conf_type=args.conf_type,
        verbose=True,
    )

    # Convert internal schema -> Kaggle schema for the saved CSV so that
    # downstream submission tools (and Kaggle's grader) can ingest it
    # without further renaming. Also filter degenerate boxes (same defense
    # as src.submit.yolo_preds_to_submission_csv) since WBF can produce
    # zero-area clusters when the pre-fusion conf is very low.
    fused_kaggle = fused.rename(columns={
        "x_min": "bbox_x",
        "y_min": "bbox_y",
        "w": "bbox_width",
        "h": "bbox_height",
    })
    n_before = len(fused_kaggle)
    fused_kaggle = fused_kaggle[
        (fused_kaggle["bbox_width"] > 0) & (fused_kaggle["bbox_height"] > 0)
    ].copy()
    n_dropped = n_before - len(fused_kaggle)
    if n_dropped > 0:
        print(f"[wbf_ensemble] dropped {n_dropped} degenerate fused boxes "
              f"(width<=0 or height<=0) of {n_before} total "
              f"({100 * n_dropped / max(n_before, 1):.3f}%)")

    fused_kaggle.to_csv(args.out, index=False)
    print(f"\nWrote {len(fused_kaggle):,} fused predictions to {args.out}")

    if not args.no_validate:
        try:
            validate_submission(args.out, test_json_path=args.test_json)
            print("validate_submission OK")
        except Exception as exc:
            print(f"validate_submission FAILED: {exc.__class__.__name__}: {exc}")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
