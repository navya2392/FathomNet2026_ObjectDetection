# Sample submission template

`sample_submission.template.csv` is the canonical schema reference for FathomNet 2026 submissions. Kaggle's docs reference a `sample_submission.csv` but it is NOT actually distributed — only 5 files ship (`README.md`, `download.py`, `requirements.txt`, `train_dataset.json`, `test_dataset.json`).

This template was hand-built from the official spec and verified against the official scorer in `docs/eval_notebook/map50-95.ipynb`.

## Schema (8 columns, exact order)

| Column | Type | Constraint | Example |
|---|---|---|---|
| `annotation_id` | int | unique per row, otherwise arbitrary | `1` |
| `image_id` | int | must match `dataset_test.json` image id | `1` |
| `category_id` | int | must be one of the 32 valid COCO category_ids (NOT 0-31) | `1` |
| `bbox_x` | float | top-left X in **pixels** (COCO format, NOT YOLO) | `476.0` |
| `bbox_y` | float | top-left Y in pixels | `790.0` |
| `bbox_width` | float | width in pixels — **must be > 0 (strict)** | `77.0` |
| `bbox_height` | float | height in pixels — **must be > 0 (strict)** | `149.0` |
| `score` | float | confidence in `[0, 1]` inclusive | `0.9` |

## Rows in the template

The template file has 3 rows that exercise the schema diversely:

1. `image_id=1, category_id=1 (amphipod)` — low-ID category
2. `image_id=2, category_id=13 (isopod)` — first category after the first ID gap (12 is missing)
3. `image_id=3, category_id=41 (urchin)` — highest-ID category

All three `image_id`s exist in `dataset_test.json`. All three `category_id`s are in the valid 32-value set.

## How to use

- **As a schema reference** when you forget the column order
- **As a unit-test fixture** for `src/submit.py::validate_submission()` — see `tests/test_submit.py`
- **As a smoke-test target** when wiring up your first inference script: convert ANY `pred_df` to a CSV via `yolo_preds_to_submission_csv()` and confirm the output matches this template's structure (column names, dtypes, row format)

## What this template is NOT

- Not a starter set of predictions to upload to Kaggle (would score 0.0).
- Not a copy of any official Kaggle file (no such file exists).
- Not exhaustive — real submissions have 1k-100k+ rows.

See also `src/submit.py` and master plan v5 "Submission Format" subsection.
