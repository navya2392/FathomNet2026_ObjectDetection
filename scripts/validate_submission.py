#!/usr/bin/env python3
"""CLI wrapper around src.submit.validate_submission.

Validates a Kaggle submission CSV against the FathomNet 2026 schema.
Exits 0 on pass, 1 on validation failure, 2 on bad usage / missing file.

Usage:
    python scripts/validate_submission.py <csv_path> [--test-json <json>] [--quiet]

The validation checks (all enforced by src.submit.validate_submission):
  1. Exactly the 8 required columns: annotation_id, image_id, category_id,
     bbox_x, bbox_y, bbox_width, bbox_height, score (no extras allowed).
  2. annotation_id values are unique.
  3. All numeric columns (bbox_*, score, category_id) are numeric, finite,
     and contain no NaN/inf.
  4. score values are in [0, 1] inclusive.
  5. bbox_width > 0 and bbox_height > 0 (strict).
  6. category_id values are in the 32-value COCO non-contiguous set
     (NOT 0-31 indices). Common bug: forgetting idx_to_cat_id mapping.
  7. (warning, not failure) >100 detections per image triggers a warning
     because pycocotools maxDets=100 will silently drop low-scoring extras.
  8. (warning, not failure, only with --test-json) image_ids not in the
     test set are flagged.

Example:
    # Standard pre-submit check
    python scripts/validate_submission.py submissions/track4_bgswap_multiscale_wbf.csv \\
      --test-json data/raw/test_dataset.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a FathomNet 2026 submission CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the submission CSV to validate.",
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        default=None,
        help="Optional path to data/raw/test_dataset.json. When provided, "
        "image_ids not in the test set are flagged as warnings.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the success summary; only print warnings/errors.",
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"ERROR: CSV not found at {args.csv_path}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from src.submit import validate_submission, ValidationError
    except ImportError as exc:
        print(f"ERROR: cannot import src.submit ({exc}). "
              f"Run from repo root or with PYTHONPATH set.", file=sys.stderr)
        return 2

    try:
        result = validate_submission(
            args.csv_path,
            test_json_path=args.test_json,
        )
    except ValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: unexpected exception during validation: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"PASS: {args.csv_path}")
        print(f"  rows                : {result['row_count']:,}")
        print(f"  unique image_ids    : {result['image_count']:,}")
        print(f"  max dets per image  : {result['max_dets_per_image']}")
        print(f"  category counts     : {len(result['category_counts'])} of 32 classes")
        if result.get("image_ids_not_in_test"):
            n = len(result["image_ids_not_in_test"])
            print(f"  WARN: {n} image_ids NOT in test set (will be silently dropped by Kaggle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
