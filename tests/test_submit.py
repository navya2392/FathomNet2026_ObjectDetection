"""Tests for src/submit.py.

Covers:
  - yolo_preds_to_submission_csv() conversion correctness, including the
    cat_id reverse mapping at the gap boundaries (0->1, 11->13, 31->41).
  - validate_submission() catches every rejection case the official
    Kaggle scorer would.
  - local_score() smoke-test: feeds ground-truth-as-predictions and
    expects mAP=1.0 (perfect score on its own labels).

Run from repo root:
    pytest tests/test_submit.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.submit import (
    PRED_DF_REQUIRED_COLUMNS,
    SUBMISSION_COLUMNS,
    VALID_COCO_CAT_IDS,
    ValidationError,
    local_score,
    validate_submission,
    yolo_preds_to_submission_csv,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_CSV = REPO_ROOT / "docs" / "sample_submission.template.csv"
CV_FOLDS_PKL = REPO_ROOT / "data" / "cv_folds.pkl"
TRAIN_JSON = REPO_ROOT / "data" / "raw" / "train_dataset.json"
TEST_JSON = REPO_ROOT / "data" / "raw" / "test_dataset.json"


def _good_pred_df() -> pd.DataFrame:
    """Minimal valid pred_df with 3 rows exercising boundaries."""
    return pd.DataFrame(
        {
            "image_id": [1, 2, 3],
            "class_idx": [0, 11, 31],
            "x_topleft": [10.0, 50.0, 100.0],
            "y_topleft": [10.0, 50.0, 100.0],
            "width": [30.0, 40.0, 50.0],
            "height": [30.0, 40.0, 50.0],
            "score": [0.9, 0.8, 0.7],
        }
    )


def _good_csv_df() -> pd.DataFrame:
    """A valid submission DataFrame matching the template."""
    return pd.read_csv(TEMPLATE_CSV)


def _write_csv(df: pd.DataFrame, tmp_path: Path) -> Path:
    out = tmp_path / "submission.csv"
    df.to_csv(out, index=False)
    return out


class TestYoloPredsToSubmissionCsv:
    def test_template_exists(self):
        assert TEMPLATE_CSV.exists(), f"Missing fixture: {TEMPLATE_CSV}"

    def test_happy_path_writes_correct_columns(self, tmp_path):
        out = tmp_path / "sub.csv"
        yolo_preds_to_submission_csv(_good_pred_df(), out)
        df = pd.read_csv(out)
        assert list(df.columns) == SUBMISSION_COLUMNS

    def test_class_idx_to_cat_id_boundary_mapping(self, tmp_path):
        """The critical FLAG 1 protection: class_idx 0/11/31 -> cat_id 1/13/41."""
        out = tmp_path / "sub.csv"
        yolo_preds_to_submission_csv(_good_pred_df(), out)
        df = pd.read_csv(out)
        cat_id_by_image = dict(zip(df["image_id"], df["category_id"]))
        assert cat_id_by_image[1] == 1, "class_idx=0 must map to cat_id=1 (amphipod)"
        assert cat_id_by_image[2] == 13, "class_idx=11 must map to cat_id=13 (isopod) -- skips the 12 gap"
        assert cat_id_by_image[3] == 41, "class_idx=31 must map to cat_id=41 (urchin) -- the top boundary"

    def test_annotation_id_is_sequential_starting_at_1(self, tmp_path):
        out = tmp_path / "sub.csv"
        yolo_preds_to_submission_csv(_good_pred_df(), out)
        df = pd.read_csv(out)
        assert df["annotation_id"].tolist() == [1, 2, 3]

    def test_sort_by_image_groups_by_image_then_descending_score(self, tmp_path):
        df_in = pd.DataFrame(
            {
                "image_id": [2, 1, 1, 2],
                "class_idx": [0, 0, 1, 1],
                "x_topleft": [0.0, 0.0, 0.0, 0.0],
                "y_topleft": [0.0, 0.0, 0.0, 0.0],
                "width": [10.0, 10.0, 10.0, 10.0],
                "height": [10.0, 10.0, 10.0, 10.0],
                "score": [0.5, 0.3, 0.9, 0.7],
            }
        )
        out = tmp_path / "sub.csv"
        yolo_preds_to_submission_csv(df_in, out, sort_by_image=True)
        df = pd.read_csv(out)
        assert df["image_id"].tolist() == [1, 1, 2, 2]
        assert df["score"].tolist() == [0.9, 0.3, 0.7, 0.5]

    def test_missing_column_raises_keyerror(self, tmp_path):
        df = _good_pred_df().drop(columns=["score"])
        with pytest.raises(KeyError, match="missing required columns"):
            yolo_preds_to_submission_csv(df, tmp_path / "sub.csv")

    def test_class_idx_out_of_range_raises_valueerror(self, tmp_path):
        df = _good_pred_df()
        df.loc[0, "class_idx"] = 32
        with pytest.raises(ValueError, match="class_idx outside 0-31"):
            yolo_preds_to_submission_csv(df, tmp_path / "sub.csv")

    def test_class_idx_negative_raises_valueerror(self, tmp_path):
        df = _good_pred_df()
        df.loc[0, "class_idx"] = -1
        with pytest.raises(ValueError, match="class_idx outside 0-31"):
            yolo_preds_to_submission_csv(df, tmp_path / "sub.csv")


class TestValidateSubmission:
    def test_template_passes_validation(self):
        result = validate_submission(TEMPLATE_CSV)
        assert result["row_count"] == 3
        assert result["image_count"] == 3
        assert result["max_dets_per_image"] == 1

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            validate_submission(tmp_path / "nope.csv")

    def test_missing_column_rejected(self, tmp_path):
        df = _good_csv_df().drop(columns=["score"])
        with pytest.raises(ValidationError, match="missing required columns"):
            validate_submission(_write_csv(df, tmp_path))

    def test_extra_column_rejected(self, tmp_path):
        df = _good_csv_df()
        df["extra"] = "junk"
        with pytest.raises(ValidationError, match="unexpected extra columns"):
            validate_submission(_write_csv(df, tmp_path))

    def test_duplicate_annotation_id_rejected(self, tmp_path):
        df = _good_csv_df()
        df.loc[1, "annotation_id"] = 1  # collide with row 0
        with pytest.raises(ValidationError, match="duplicate values"):
            validate_submission(_write_csv(df, tmp_path))

    def test_nan_in_score_rejected(self, tmp_path):
        df = _good_csv_df()
        df.loc[0, "score"] = np.nan
        with pytest.raises(ValidationError, match="NaN"):
            validate_submission(_write_csv(df, tmp_path))

    def test_inf_in_bbox_rejected(self, tmp_path):
        df = _good_csv_df()
        df.loc[0, "bbox_x"] = np.inf
        with pytest.raises(ValidationError, match="infinity"):
            validate_submission(_write_csv(df, tmp_path))

    def test_score_below_zero_rejected(self, tmp_path):
        df = _good_csv_df()
        df.loc[0, "score"] = -0.01
        with pytest.raises(ValidationError, match=r"\[0, 1\]"):
            validate_submission(_write_csv(df, tmp_path))

    def test_score_above_one_rejected(self, tmp_path):
        df = _good_csv_df()
        df.loc[0, "score"] = 1.01
        with pytest.raises(ValidationError, match=r"\[0, 1\]"):
            validate_submission(_write_csv(df, tmp_path))

    def test_score_exactly_at_one_accepted(self, tmp_path):
        df = _good_csv_df()
        df.loc[0, "score"] = 1.0
        validate_submission(_write_csv(df, tmp_path))

    def test_score_exactly_at_zero_accepted(self, tmp_path):
        df = _good_csv_df()
        df.loc[0, "score"] = 0.0
        validate_submission(_write_csv(df, tmp_path))

    def test_zero_bbox_width_rejected(self, tmp_path):
        df = _good_csv_df()
        df.loc[0, "bbox_width"] = 0.0
        with pytest.raises(ValidationError, match="bbox_width"):
            validate_submission(_write_csv(df, tmp_path))

    def test_negative_bbox_height_rejected(self, tmp_path):
        df = _good_csv_df()
        df.loc[0, "bbox_height"] = -1.0
        with pytest.raises(ValidationError, match="bbox_height"):
            validate_submission(_write_csv(df, tmp_path))

    def test_category_id_in_gap_rejected(self, tmp_path):
        """cat_id=12 is a gap value; any submission with it should fail."""
        df = _good_csv_df()
        df.loc[0, "category_id"] = 12
        with pytest.raises(ValidationError, match="not in the valid set"):
            validate_submission(_write_csv(df, tmp_path))

    def test_category_id_zero_indexed_bug_rejected(self, tmp_path):
        """Most common bug: submitting class_idx (0-31) instead of cat_id."""
        df = _good_csv_df()
        df.loc[0, "category_id"] = 0
        with pytest.raises(ValidationError, match="not in the valid set"):
            validate_submission(_write_csv(df, tmp_path))

    def test_category_id_42_rejected(self, tmp_path):
        """Out of range high."""
        df = _good_csv_df()
        df.loc[0, "category_id"] = 42
        with pytest.raises(ValidationError, match="not in the valid set"):
            validate_submission(_write_csv(df, tmp_path))

    def test_all_valid_cat_ids_accepted(self, tmp_path):
        """Build a 32-row submission, one per valid cat_id; must pass."""
        rows = []
        for i, cid in enumerate(sorted(VALID_COCO_CAT_IDS), start=1):
            rows.append({
                "annotation_id": i,
                "image_id": i,
                "category_id": cid,
                "bbox_x": 10.0, "bbox_y": 10.0,
                "bbox_width": 20.0, "bbox_height": 20.0,
                "score": 0.5,
            })
        df = pd.DataFrame(rows)
        result = validate_submission(_write_csv(df, tmp_path))
        assert len(result["category_counts"]) == 32

    def test_over_100_dets_per_image_warns_but_accepts(self, tmp_path):
        warnings_seen = []
        rows = []
        for i in range(110):
            rows.append({
                "annotation_id": i + 1,
                "image_id": 1,  # all on the same image
                "category_id": 1,
                "bbox_x": float(i), "bbox_y": float(i),
                "bbox_width": 10.0, "bbox_height": 10.0,
                "score": 0.5,
            })
        df = pd.DataFrame(rows)
        result = validate_submission(_write_csv(df, tmp_path), warn_callback=warnings_seen.append)
        assert any("100 detections" in w for w in warnings_seen)
        assert result["max_dets_per_image"] == 110

    def test_unknown_image_id_warns_when_test_json_provided(self, tmp_path):
        if not TEST_JSON.exists():
            pytest.skip("dataset_test.json not present")
        warnings_seen = []
        df = _good_csv_df()
        df.loc[0, "image_id"] = 999_999_999  # definitely not in test set
        result = validate_submission(
            _write_csv(df, tmp_path),
            test_json_path=TEST_JSON,
            warn_callback=warnings_seen.append,
        )
        assert any("not in the test set" in w for w in warnings_seen)
        assert 999_999_999 in result["image_ids_not_in_test"]


class TestLocalScore:
    """End-to-end sanity check: feeding ground truth as predictions yields mAP=1.0."""

    def test_perfect_predictions_score_one(self):
        if not (CV_FOLDS_PKL.exists() and TRAIN_JSON.exists()):
            pytest.skip("cv_folds.pkl or train_dataset.json missing")

        # Load fold 0 val annotations
        import pickle
        with CV_FOLDS_PKL.open("rb") as f:
            cv = pickle.load(f)
        val_ids = set(cv["folds"][0]["val_image_ids"])
        with TRAIN_JSON.open() as f:
            train = json.load(f)
        val_anns = [a for a in train["annotations"] if a["image_id"] in val_ids]

        from configs.cat_id_mapping import cat_id_to_idx

        rows = []
        for ann in val_anns:
            x, y, w, h = ann["bbox"]
            rows.append({
                "image_id": int(ann["image_id"]),
                "class_idx": cat_id_to_idx[int(ann["category_id"])],
                "x_topleft": float(x),
                "y_topleft": float(y),
                "width": float(w),
                "height": float(h),
                "score": 1.0,  # perfect confidence
            })
        pred_df = pd.DataFrame(rows)
        score = local_score(pred_df, fold_idx=0)
        # Expect very close to 1.0 (within float tolerance for pycocotools)
        assert score > 0.99, f"Perfect predictions scored {score:.4f}, expected > 0.99"

    def test_empty_predictions_score_zero(self):
        if not (CV_FOLDS_PKL.exists() and TRAIN_JSON.exists()):
            pytest.skip("cv_folds.pkl or train_dataset.json missing")
        empty_df = pd.DataFrame(columns=PRED_DF_REQUIRED_COLUMNS)
        score = local_score(empty_df, fold_idx=0)
        assert score == 0.0
