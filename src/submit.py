"""Submission helper: the one place that owns category_id reverse mapping
and submission validation for FathomNet 2026.

THREE PUBLIC FUNCTIONS:

  yolo_preds_to_submission_csv(pred_df, out_path)
      Convert model predictions (in 0-31 class indices) into the 8-column
      Kaggle submission CSV (with original COCO category_ids 1-41 with gaps).
      THIS IS THE SINGLE POINT WHERE class_idx -> cat_id MAPPING HAPPENS.
      Every inference script in Phase 2-7 MUST use this -- never hand-roll.

  validate_submission(csv_path)
      Run every check the official Kaggle scorer runs. Raises ValidationError
      with a precise message if anything would cause Kaggle rejection. Returns
      silently if the CSV is submission-ready.

  local_score(pred_df, fold_idx)
      Compute Kaggle's EXACT mAP@[.50:.95] offline on a CV fold. Wraps the
      official pycocotools-based scorer (vendored from
      docs/eval_notebook/map50-95.ipynb). Use this for every "is my run any
      good?" check so daily Kaggle submission slots are reserved for genuine
      leaderboard probes (per master plan rules R8 + R10).

SUPPLY HELPERS (Tier 1C-i / tooling):

  local_per_category_ap5095_from_solution, build_cv_val_solution_dataframe,
  filter_pred_df_by_per_class_conf, load_per_class_conf_thresholds

DESIGN DECISIONS:

- pred_df schema is the lingua franca: every Phase 2-7 inference script
  produces this DataFrame; submit helpers consume it. Never circumvent.
- The official scorer's source is COPIED inline below (with attribution)
  rather than imported from the .ipynb. Reason: the notebook isn't a Python
  module, and we want zero risk of import path / kernel issues at
  submission time. If the official notebook updates we should re-vendor.
- Validation errors are FAIL-LOUD. A rejected Kaggle submission still
  consumes a daily slot, so we want to catch problems in the local
  validate_submission() call, not at upload time.

MAPPING REFERENCE:

    pred_df["class_idx"]  ->  configs/cat_id_mapping.py::idx_to_cat_id
                          ->  CSV "category_id" column
    (0-31)                                              (1-41 with gaps)

SUBMISSION CSV SCHEMA (per master plan v5 "Submission Format" subsection):

    annotation_id    int    unique row identifier (we generate sequentially)
    image_id         int    must match dataset_test.json image_id
    category_id      int    must be in COCO_CAT_IDS (FLAG 1 / FLAG 9)
    bbox_x           float  top-left x in pixels (COCO format, NOT YOLO)
    bbox_y           float  top-left y in pixels
    bbox_width       float  must be > 0 strict
    bbox_height      float  must be > 0 strict
    score            float  must be in [0, 1] inclusive
"""
from __future__ import annotations

import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Allow `from src.submit import ...` from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.cat_id_mapping import (  # noqa: E402  (after sys.path bump)
    COCO_CAT_IDS,
    cat_id_to_idx,
    idx_to_cat_id,
)

VALID_COCO_CAT_IDS: set[int] = set(COCO_CAT_IDS)
SUBMISSION_COLUMNS: list[str] = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "score",
]
PRED_DF_REQUIRED_COLUMNS: list[str] = [
    "image_id",
    "class_idx",
    "x_topleft",
    "y_topleft",
    "width",
    "height",
    "score",
]
NUMERIC_BBOX_AND_SCORE_COLUMNS: list[str] = [
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "score",
]
COCO_MAX_DETS_PER_IMAGE: int = 100  # pycocotools default; warning threshold

DEFAULT_CV_FOLDS_PATH = _REPO_ROOT / "data" / "cv_folds.pkl"
DEFAULT_TRAIN_JSON_PATH = _REPO_ROOT / "data" / "raw" / "train_dataset.json"
DEFAULT_TEST_JSON_PATH = _REPO_ROOT / "data" / "raw" / "test_dataset.json"


def val_image_ids_for_fold(
    fold_idx: int,
    *,
    cv_folds_path: str | Path | None = None,
) -> set[int]:
    """CV fold validation image ids (same definition as ``local_score``)."""
    cv_folds_path = Path(cv_folds_path) if cv_folds_path else DEFAULT_CV_FOLDS_PATH
    with cv_folds_path.open("rb") as f:
        cv_data = pickle.load(f)
    n_folds = cv_data["n_folds"]
    if not 0 <= fold_idx < n_folds:
        raise ValueError(f"fold_idx must be in [0, {n_folds - 1}], got {fold_idx}")
    return set(cv_data["folds"][fold_idx]["val_image_ids"])


def build_cv_val_solution_dataframe(
    fold_idx: int,
    *,
    cv_folds_path: str | Path | None = None,
    train_json_path: str | Path | None = None,
    val_image_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Ground-truth rows for ``local_score`` / tuning (val annotations only)."""
    cv_folds_path = Path(cv_folds_path) if cv_folds_path else DEFAULT_CV_FOLDS_PATH
    train_json_path = Path(train_json_path) if train_json_path else DEFAULT_TRAIN_JSON_PATH
    if val_image_ids is None:
        val_image_ids = val_image_ids_for_fold(fold_idx, cv_folds_path=cv_folds_path)

    with train_json_path.open() as f:
        train_data = json.load(f)
    val_annotations = [a for a in train_data["annotations"] if a["image_id"] in val_image_ids]

    solution_rows = []
    for ann_idx, ann in enumerate(val_annotations, start=1):
        x, y, w, h = ann["bbox"]
        solution_rows.append(
            {
                "annotation_id": ann_idx,
                "image_id": int(ann["image_id"]),
                "category_id": int(ann["category_id"]),
                "bbox_x": float(x),
                "bbox_y": float(y),
                "bbox_width": float(w),
                "bbox_height": float(h),
            }
        )
    return pd.DataFrame(
        solution_rows,
        columns=[
            "annotation_id",
            "image_id",
            "category_id",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
        ],
    )


def _pred_subset_to_submission_df(pred_subset: pd.DataFrame) -> pd.DataFrame:
    """pred_df rows -> Kaggle submission-frame columns for COCOeval."""
    return pd.DataFrame(
        {
            "annotation_id": np.arange(1, len(pred_subset) + 1, dtype=int),
            "image_id": pred_subset["image_id"].astype(int).values,
            "category_id": pred_subset["class_idx"].map(idx_to_cat_id).astype(int).values,
            "bbox_x": pred_subset["x_topleft"].astype(float).values,
            "bbox_y": pred_subset["y_topleft"].astype(float).values,
            "bbox_width": pred_subset["width"].astype(float).values,
            "bbox_height": pred_subset["height"].astype(float).values,
            "score": pred_subset["score"].astype(float).values,
        }
    )


def filter_pred_df_by_per_class_conf(
    pred_df: pd.DataFrame,
    thresholds_by_cat_id: dict[int, float],
    *,
    default_threshold: float,
) -> pd.DataFrame:
    """Drop rows whose ``score`` is below the threshold for that row's COCO category."""
    if pred_df.empty:
        return pred_df
    out = pred_df.copy()
    mapped_cat = out["class_idx"].map(idx_to_cat_id).astype(int)
    thr_series = mapped_cat.map(lambda c: thresholds_by_cat_id.get(int(c), default_threshold)).astype(
        np.float64
    )
    keep = out["score"].to_numpy(dtype=np.float64) >= thr_series.to_numpy()
    return out.loc[keep].reset_index(drop=True)


def load_per_class_conf_thresholds(path: str | Path) -> tuple[dict[int, float], float]:
    """Load Tier 1C-i JSON written by ``scripts/tune_per_class_conf.py``."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    default_thr = float(payload.get("default_conf", 0.001))
    by_cat: dict[int, float] = {}
    by_idx = payload.get("thresholds_by_class_idx")
    if isinstance(by_idx, list) and len(by_idx) == len(idx_to_cat_id):
        for i, thr in enumerate(by_idx):
            by_cat[int(idx_to_cat_id[i])] = float(thr)
    for k, v in (payload.get("thresholds_by_cat_id") or {}).items():
        by_cat[int(k)] = float(v)
    return by_cat, default_thr


class ValidationError(ValueError):
    """Raised by validate_submission when a check fails.

    The message is human-readable and identifies the exact violation, so the
    error can be acted on without needing to inspect the dataframe.
    """


def yolo_preds_to_submission_csv(
    pred_df: pd.DataFrame,
    out_path: str | Path,
    *,
    sort_by_image: bool = True,
) -> Path:
    """Write a Kaggle submission CSV from a predictions DataFrame.

    Maps 0-31 class indices to the original (non-contiguous) COCO category_ids
    via configs/cat_id_mapping.py::idx_to_cat_id. This is the ONLY place that
    mapping should happen in the codebase.

    Parameters
    ----------
    pred_df :
        Must have columns: image_id (int), class_idx (int 0-31),
        x_topleft (float pixels), y_topleft (float pixels),
        width (float pixels), height (float pixels), score (float [0, 1]).
        Bounding boxes are in COCO pixel format (top-left + width/height),
        NOT YOLO normalized format.
    out_path :
        Where to write the CSV. Parent dir is created if missing.
    sort_by_image :
        If True (default), output rows are grouped by image_id then by
        descending score. Improves human readability and is friendly for
        any downstream tool that expects per-image grouping.

    Returns
    -------
    Path
        Resolved path of the written CSV.

    Raises
    ------
    KeyError
        If pred_df is missing any required column.
    ValueError
        If pred_df contains class_idx values outside 0-31.

    Notes
    -----
    annotation_id is generated as a 1-based sequential integer. Kaggle says
    it must be unique per row but is otherwise arbitrary; we use the row
    number for traceability.
    """
    missing = [c for c in PRED_DF_REQUIRED_COLUMNS if c not in pred_df.columns]
    if missing:
        raise KeyError(
            f"pred_df is missing required columns: {missing}. "
            f"Required schema: {PRED_DF_REQUIRED_COLUMNS}"
        )

    bad_class_idx = pred_df.loc[~pred_df["class_idx"].isin(idx_to_cat_id.keys()), "class_idx"]
    if not bad_class_idx.empty:
        raise ValueError(
            f"pred_df contains {len(bad_class_idx)} rows with class_idx outside 0-31. "
            f"Sample bad values: {bad_class_idx.unique()[:5].tolist()}"
        )

    out_df = pd.DataFrame(
        {
            "image_id": pred_df["image_id"].astype(int),
            "category_id": pred_df["class_idx"].map(idx_to_cat_id).astype(int),
            "bbox_x": pred_df["x_topleft"].astype(float),
            "bbox_y": pred_df["y_topleft"].astype(float),
            "bbox_width": pred_df["width"].astype(float),
            "bbox_height": pred_df["height"].astype(float),
            "score": pred_df["score"].astype(float),
        }
    )

    # Models occasionally produce degenerate boxes (width or height <= 0) at
    # image boundaries, after NMS coordinate rounding, or on overlapping
    # cluster collapse. Kaggle validates STRICT positive dimensions
    # (validate_submission() below raises ValidationError otherwise), so we
    # filter them here. Keeping the count visible for monitoring.
    n_before = len(out_df)
    out_df = out_df[(out_df["bbox_width"] > 0) & (out_df["bbox_height"] > 0)].copy()
    n_dropped = n_before - len(out_df)
    if n_dropped > 0:
        print(f"[yolo_preds_to_submission_csv] dropped {n_dropped} degenerate "
              f"boxes (width<=0 or height<=0) of {n_before} total rows "
              f"({100 * n_dropped / max(n_before, 1):.3f}%)")

    if sort_by_image:
        out_df = out_df.sort_values(
            by=["image_id", "score"],
            ascending=[True, False],
            kind="stable",
        ).reset_index(drop=True)

    out_df.insert(0, "annotation_id", np.arange(1, len(out_df) + 1, dtype=int))
    out_df = out_df[SUBMISSION_COLUMNS]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_path.resolve()


def validate_submission(
    csv_path: str | Path,
    *,
    test_json_path: str | Path | None = None,
    warn_callback=None,
) -> dict:
    """Run every check the official Kaggle scorer runs against a CSV.

    Mirrors the validation logic in docs/eval_notebook/map50-95.ipynb.
    Use this before EVERY upload -- a rejected submission still consumes
    a daily slot (per master plan rule R10).

    Parameters
    ----------
    csv_path :
        Path to the submission CSV to validate.
    test_json_path :
        Optional path to dataset_test.json. If provided, image_ids not
        in the test set are reported as warnings (Kaggle silently drops
        these rows but they're usually a sign of a pipeline bug).
    warn_callback :
        Callable that takes a single str argument. Used for non-fatal
        warnings (e.g., >100 detections per image). Defaults to print.

    Returns
    -------
    dict
        Summary statistics: {"row_count", "image_count", "category_counts",
        "max_dets_per_image", "image_ids_not_in_test"}.

    Raises
    ------
    ValidationError
        If any check that would cause Kaggle rejection fails.
    """
    if warn_callback is None:
        warn_callback = lambda msg: print(f"WARN: {msg}", file=sys.stderr)

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise ValidationError(f"Submission CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    missing_cols = [c for c in SUBMISSION_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"CSV missing required columns: {missing_cols}. "
            f"Required (in any order): {SUBMISSION_COLUMNS}"
        )
    extra_cols = [c for c in df.columns if c not in SUBMISSION_COLUMNS]
    if extra_cols:
        raise ValidationError(
            f"CSV has unexpected extra columns: {extra_cols}. "
            f"Allowed: {SUBMISSION_COLUMNS}"
        )

    if df["annotation_id"].duplicated().any():
        dup_count = df["annotation_id"].duplicated().sum()
        raise ValidationError(
            f"annotation_id contains {dup_count} duplicate values. Must be unique."
        )

    for col in NUMERIC_BBOX_AND_SCORE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValidationError(
                f'Column "{col}" must be numeric, got dtype: {df[col].dtype}'
            )
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            raise ValidationError(
                f'Column "{col}" contains {n} NaN/null values. All values must be finite numbers.'
            )
        if np.isinf(df[col]).any():
            n = int(np.isinf(df[col]).sum())
            raise ValidationError(
                f'Column "{col}" contains {n} infinity values. All values must be finite numbers.'
            )

    bad_score_mask = (df["score"] < 0) | (df["score"] > 1)
    if bad_score_mask.any():
        bad = df.loc[bad_score_mask, "score"]
        raise ValidationError(
            f'Column "score" must be in [0, 1] inclusive. Found {len(bad)} invalid '
            f"values (min={bad.min():.4f}, max={bad.max():.4f})."
        )

    if (df["bbox_width"] <= 0).any():
        n = int((df["bbox_width"] <= 0).sum())
        raise ValidationError(f'Column "bbox_width" must be > 0 (strict). Found {n} bad rows.')
    if (df["bbox_height"] <= 0).any():
        n = int((df["bbox_height"] <= 0).sum())
        raise ValidationError(f'Column "bbox_height" must be > 0 (strict). Found {n} bad rows.')

    submitted_cats = set(df["category_id"].astype(int).unique())
    bad_cats = sorted(submitted_cats - VALID_COCO_CAT_IDS)
    if bad_cats:
        raise ValidationError(
            f"category_id contains values not in the valid set: {bad_cats}. "
            f"Valid IDs (32 values): {sorted(VALID_COCO_CAT_IDS)}. "
            f"Common bug: submitting 0-31 class indices instead of COCO category_ids "
            f"-- use yolo_preds_to_submission_csv() to avoid this."
        )

    dets_per_image = df.groupby("image_id").size()
    max_dets = int(dets_per_image.max()) if not dets_per_image.empty else 0
    over_limit = dets_per_image[dets_per_image > COCO_MAX_DETS_PER_IMAGE]
    if not over_limit.empty:
        warn_callback(
            f"{len(over_limit)} image(s) have >{COCO_MAX_DETS_PER_IMAGE} detections "
            f"(max observed: {max_dets}). pycocotools maxDets={COCO_MAX_DETS_PER_IMAGE} "
            f"will silently drop the lowest-scoring extras during scoring."
        )

    image_ids_not_in_test: list[int] = []
    if test_json_path is not None:
        test_path = Path(test_json_path)
        with test_path.open() as f:
            test_data = json.load(f)
        test_ids = {int(img["id"]) for img in test_data["images"]}
        unknown = sorted(set(df["image_id"].astype(int).unique()) - test_ids)
        if unknown:
            warn_callback(
                f"Submission contains {len(unknown)} image_ids not in the test set "
                f"(Kaggle will silently drop these rows). Sample: {unknown[:5]}"
            )
            image_ids_not_in_test = unknown

    return {
        "row_count": len(df),
        "image_count": int(df["image_id"].nunique()),
        "category_counts": df["category_id"].value_counts().to_dict(),
        "max_dets_per_image": max_dets,
        "image_ids_not_in_test": image_ids_not_in_test,
    }


def local_score(
    pred_df: pd.DataFrame,
    fold_idx: int,
    *,
    cv_folds_path: str | Path | None = None,
    train_json_path: str | Path | None = None,
) -> float:
    """Compute Kaggle's EXACT mAP@[.50:.95] on a held-out CV fold.

    Builds the "solution" DataFrame from cv_folds.pkl + dataset_train.json,
    converts pred_df via the same logic as yolo_preds_to_submission_csv,
    then runs the vendored official scorer. The returned float IS what
    Kaggle's leaderboard would show.

    Parameters
    ----------
    pred_df :
        Predictions on the val images of fold_idx. Same schema as
        yolo_preds_to_submission_csv input. May contain predictions on
        train images of the fold; they are silently ignored (they don't
        appear in the solution DataFrame so don't affect scoring).
    fold_idx :
        Which CV fold (0-4) to score against.
    cv_folds_path, train_json_path :
        Override defaults if you keep these files outside the repo.

    Returns
    -------
    float
        mAP@[.50:.95] in [0.0, 1.0]. Higher is better. Returns 0.0 if the
        prediction set is empty after filtering.

    Notes
    -----
    The scorer caps at 100 detections per image (pycocotools maxDets=100).
    If your model produces >100 dets per image, only the top-100-by-score
    are considered.
    """
    cv_folds_path = Path(cv_folds_path) if cv_folds_path else DEFAULT_CV_FOLDS_PATH
    train_json_path = Path(train_json_path) if train_json_path else DEFAULT_TRAIN_JSON_PATH

    val_image_ids = val_image_ids_for_fold(fold_idx, cv_folds_path=cv_folds_path)
    solution_df = build_cv_val_solution_dataframe(
        fold_idx,
        cv_folds_path=cv_folds_path,
        train_json_path=train_json_path,
        val_image_ids=val_image_ids,
    )

    pred_subset = pred_df[pred_df["image_id"].isin(val_image_ids)].copy()
    if pred_subset.empty:
        return 0.0

    submission_df = _pred_subset_to_submission_df(pred_subset)

    return _coco_score(solution_df, submission_df, "annotation_id")


def local_per_category_ap5095_from_solution(
    solution_df: pd.DataFrame,
    pred_subset: pd.DataFrame,
) -> dict[int, float]:
    """Per-COCO-category AP@[.50:.95] using the same eval setup as ``local_score``.

    Typically ``solution_df`` is fixed (val GT) while ``pred_subset`` varies during a
    hyper-parameter sweep (Tier 1C-i per-class confidence grid).

    Requires ``pycocotools``. Returns an empty dict if ``pred_subset`` is empty.
    """
    if pred_subset.empty:
        return {}
    submission_df = _pred_subset_to_submission_df(pred_subset)
    coco_eval = _coco_bbox_accumulate(solution_df, submission_df)
    if coco_eval is None:
        return {}
    return _per_category_ap5095_from_eval(coco_eval)


def _per_category_ap5095_from_eval(coco_eval: Any) -> dict[int, float]:
    """Mean precision tensor aggregation (Detectron2-style nanmean over IoU × recall)."""
    prec = coco_eval.eval["precision"]
    params = coco_eval.params
    try:
        a_idx = params.areaRngLbl.index("all")
    except (AttributeError, ValueError):
        a_idx = 0
    try:
        m_idx = list(params.maxDets).index(100)
    except (AttributeError, ValueError):
        m_idx = min(len(params.maxDets) - 1, 2)
    prec_k = prec[:, :, :, a_idx, m_idx].astype(np.float64)
    prec_k[prec_k < 0] = np.nan
    ap_per_k = np.nanmean(prec_k, axis=(0, 1))
    out: dict[int, float] = {}
    for cid, ap in zip(params.catIds, ap_per_k):
        v = float(ap)
        out[int(cid)] = 0.0 if np.isnan(v) else v
    return out


def _coco_bbox_accumulate(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
) -> Any | None:
    """Validate frames, build COCO structs, ``evaluate`` + ``accumulate``. No ``summarize``."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_df = solution.copy()
    pred_df = submission.copy()

    required_solution_cols = ["image_id", "category_id", "bbox_x", "bbox_y", "bbox_width", "bbox_height"]
    for col in required_solution_cols:
        if col not in gt_df.columns:
            raise ValidationError(f"Solution missing column: {col}")

    required_submission_cols = required_solution_cols + ["score"]
    for col in required_submission_cols:
        if col not in pred_df.columns:
            raise ValidationError(f"Submission missing column: {col}")

    numeric_cols = ["bbox_x", "bbox_y", "bbox_width", "bbox_height", "score"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(pred_df[col]):
            raise ValidationError(f'Column "{col}" must be numeric, got dtype: {pred_df[col].dtype}')
        if pred_df[col].isna().any():
            raise ValidationError(f'Column "{col}" contains NaN values.')
        if np.isinf(pred_df[col]).any():
            raise ValidationError(f'Column "{col}" contains infinity values.')

    if (pred_df["score"] < 0).any() or (pred_df["score"] > 1).any():
        raise ValidationError('Column "score" must be in [0, 1] inclusive.')
    if (pred_df["bbox_width"] <= 0).any() or (pred_df["bbox_height"] <= 0).any():
        raise ValidationError('Columns "bbox_width" and "bbox_height" must be > 0 strictly.')

    gt_image_ids = set(gt_df["image_id"].unique())
    pred_df = pred_df[pred_df["image_id"].isin(gt_image_ids)].copy()
    if len(pred_df) == 0:
        return None

    valid_category_ids = set(gt_df["category_id"].unique())
    invalid = sorted(set(pred_df["category_id"].unique()) - valid_category_ids)
    if invalid:
        raise ValidationError(f"Submission has invalid category_ids: {invalid}")

    unique_images = sorted(gt_image_ids)
    unique_categories = sorted(valid_category_ids)

    coco_gt = {
        "info": {"description": "Local CV scoring"},
        "licenses": [],
        "images": [{"id": int(i)} for i in unique_images],
        "categories": [{"id": int(c), "name": str(c)} for c in unique_categories],
        "annotations": [],
    }
    for ann_idx, (_, row) in enumerate(gt_df.iterrows(), start=1):
        coco_gt["annotations"].append(
            {
                "id": int(ann_idx),
                "image_id": int(row["image_id"]),
                "category_id": int(row["category_id"]),
                "bbox": [
                    float(row["bbox_x"]),
                    float(row["bbox_y"]),
                    float(row["bbox_width"]),
                    float(row["bbox_height"]),
                ],
                "area": float(row["bbox_width"] * row["bbox_height"]),
                "iscrowd": 0,
            }
        )

    coco_dt = []
    for _, row in pred_df.iterrows():
        coco_dt.append(
            {
                "image_id": int(row["image_id"]),
                "category_id": int(row["category_id"]),
                "bbox": [
                    float(row["bbox_x"]),
                    float(row["bbox_y"]),
                    float(row["bbox_width"]),
                    float(row["bbox_height"]),
                ],
                "score": float(row["score"]),
            }
        )
    if not coco_dt:
        return None

    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_gt_obj = COCO()
        coco_gt_obj.dataset = coco_gt
        coco_gt_obj.createIndex()
        coco_dt_obj = coco_gt_obj.loadRes(coco_dt)
        coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        return coco_eval
    finally:
        sys.stdout = original_stdout


def _coco_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
) -> float:
    """Compute COCO mAP@[.50:.95] via pycocotools.

    VENDORED FROM docs/eval_notebook/map50-95.ipynb (Kaggle's official
    scorer for FathomNet 2026, written by Laura Chrobak / MBARI).
    DO NOT EDIT independently -- if Kaggle updates the notebook, re-vendor
    this function from there to keep local scores aligned with leaderboard.

    Original docstring trimmed for brevity; behavior identical.
    """
    coco_eval = _coco_bbox_accumulate(solution, submission)
    if coco_eval is None:
        return 0.0

    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_eval.summarize()
        m_ap = coco_eval.stats[0]
    finally:
        sys.stdout = original_stdout

    return float(m_ap) if not np.isnan(m_ap) else 0.0


def unmirror_horizontal_xywh(pred_df: pd.DataFrame, image_width: float) -> pd.DataFrame:
    """Map COCO xywh boxes from a horizontally flipped image back to original coords.

    Flipped-input inference sees x increasing rightward on the mirrored raster.
    Original top-left x is: W - x_flip - w.
    """
    if pred_df.empty:
        return pred_df
    out = pred_df.copy()
    out["x_topleft"] = float(image_width) - out["x_topleft"] - out["width"]
    return out


def from_ultralytics_results(
    results: Iterable,
    image_id_lookup: dict[str, int],
    *,
    override_fnames: list[str] | None = None,
) -> pd.DataFrame:
    """Convenience: convert Ultralytics YOLO predict() output -> pred_df schema.

    Parameters
    ----------
    results :
        Iterable of ultralytics.engine.results.Results objects (one per image).
        Each must have .boxes.xyxy, .boxes.cls, .boxes.conf, and .path.
    image_id_lookup :
        Map from image filename (basename) to int image_id from
        dataset_test.json. Build once via:
            with open("data/raw/test_dataset.json") as f:
                test = json.load(f)
            image_id_lookup = {img["file_name"]: img["id"] for img in test["images"]}
    override_fnames :
        If set, must align 1:1 with ``results`` (same length / order). Use when
        ``predict()`` was run on in-memory arrays so ``Result.path`` is not the
        real test filename.

    Returns
    -------
    pd.DataFrame
        Schema matching PRED_DF_REQUIRED_COLUMNS, ready to feed into
        yolo_preds_to_submission_csv() or local_score().

    Notes
    -----
    Ultralytics returns xyxy in absolute pixels. We convert to xywh (top-left
    + width/height) which is COCO's bbox convention.
    """
    rows = []
    results_list = list(results)
    if override_fnames is not None and len(override_fnames) != len(results_list):
        raise ValueError(
            f"override_fnames length ({len(override_fnames)}) != "
            f"len(results) ({len(results_list)})"
        )
    for i, res in enumerate(results_list):
        if override_fnames is not None:
            fname = override_fnames[i]
        else:
            path = Path(res.path)
            fname = path.name
        if fname not in image_id_lookup:
            raise KeyError(f"Filename {fname} not in image_id_lookup; build lookup from test_dataset.json")
        image_id = image_id_lookup[fname]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
            rows.append(
                {
                    "image_id": int(image_id),
                    "class_idx": int(c),
                    "x_topleft": float(x1),
                    "y_topleft": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                    "score": float(s),
                }
            )
    return pd.DataFrame(rows, columns=PRED_DF_REQUIRED_COLUMNS)


__all__ = [
    "ValidationError",
    "yolo_preds_to_submission_csv",
    "validate_submission",
    "local_score",
    "local_per_category_ap5095_from_solution",
    "build_cv_val_solution_dataframe",
    "val_image_ids_for_fold",
    "filter_pred_df_by_per_class_conf",
    "load_per_class_conf_thresholds",
    "from_ultralytics_results",
    "unmirror_horizontal_xywh",
    "SUBMISSION_COLUMNS",
    "PRED_DF_REQUIRED_COLUMNS",
    "VALID_COCO_CAT_IDS",
]
