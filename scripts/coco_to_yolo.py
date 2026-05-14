"""COCO -> YOLO label converter for FathomNet 2026.

WHY THIS EXISTS:
    Ultralytics ships convert_coco() in their utils, but it has a known
    bug for datasets with non-contiguous category_ids (FLAG 8 in master
    plan v5): it does category_id - 1 to derive class_idx, which silently
    produces wrong labels for FathomNet (cat_id 13 -> class_idx 12 is
    wrong; should be 11 because cat_id 12 is missing). Using their
    converter would corrupt every label file we feed YOLO and we'd
    train on garbage.

WHAT IT DOES:
    Reads data/raw/train_dataset.json (COCO format) and writes one YOLO
    label .txt per image to data/labels/. Each line is:
        class_idx cx_norm cy_norm w_norm h_norm
    where:
        class_idx is the 0-31 contiguous index from cat_id_to_idx
        (cx, cy) is bbox CENTER, normalized by image (width, height)
        (w, h)  is bbox dimensions, normalized by image (width, height)

    Images with zero annotations get an empty .txt -- YOLO treats those
    as negatives (no objects), which we want for PU learning.

    The output is fold-agnostic: ONE labels dir for the whole train set,
    plus fold-specific train/val image lists (built by make_yolo_fold.py).

USAGE:
    python scripts/coco_to_yolo.py
    python scripts/coco_to_yolo.py --train-json path/to/x.json --out-dir y/

    By default, paths come from the repo layout in master_checklist.txt.

VERIFICATION:
    The script prints a count summary (#images, #annotations written,
    #images with zero annotations) and verifies that every cat_id in the
    input was successfully mapped (loud failure if any unknown cat_id).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.cat_id_mapping import cat_id_to_idx  # noqa: E402

DEFAULT_TRAIN_JSON = REPO_ROOT / "data" / "raw" / "train_dataset.json"
# Labels sit beside images so Ultralytics' implicit discovery works:
# Ultralytics swaps "/images/" -> "/labels/" in each image path to find its
# label. Images live at data/raw/images/train/<uuid>.{png,jpg}, so labels
# go at data/raw/labels/train/<uuid>.txt.
DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "raw" / "labels" / "train"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--train-json",
        type=Path,
        default=DEFAULT_TRAIN_JSON,
        help=f"Path to COCO-format train annotations JSON (default: {DEFAULT_TRAIN_JSON.relative_to(REPO_ROOT)})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_LABELS_DIR,
        help=f"Where to write YOLO .txt labels (default: {DEFAULT_LABELS_DIR.relative_to(REPO_ROOT)})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute counts and verify mapping but DON'T write files (useful for sanity check).",
    )
    p.add_argument(
        "--clip-bbox",
        action="store_true",
        help="If a bbox extends past the image edge, clip it to the image. Off by default; "
             "the script will WARN per-row instead so you can investigate the data quality.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.train_json.exists():
        print(f"ERROR: train annotations file not found: {args.train_json}", file=sys.stderr)
        return 2

    with args.train_json.open() as f:
        coco = json.load(f)

    image_index = {img["id"]: img for img in coco["images"]}
    n_images = len(image_index)
    print(f"Loaded {n_images:,} images and {len(coco['annotations']):,} annotations from {args.train_json.name}")

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # Group annotations by image_id (single linear pass).
    anns_by_image: dict[int, list[dict]] = {img_id: [] for img_id in image_index}
    unknown_cat_ids: Counter = Counter()
    orphan_image_anns = 0
    for ann in coco["annotations"]:
        cat_id = int(ann["category_id"])
        if cat_id not in cat_id_to_idx:
            unknown_cat_ids[cat_id] += 1
            continue
        iid = int(ann["image_id"])
        if iid not in image_index:
            orphan_image_anns += 1
            continue
        anns_by_image[iid].append(ann)

    if orphan_image_anns:
        print(
            f"FATAL: {orphan_image_anns:,} annotations reference image_id "
            f"values not listed in coco['images'] (orphans). Fix train_dataset.json.",
            file=sys.stderr,
        )
        return 3

    if unknown_cat_ids:
        print(
            f"FATAL: encountered {sum(unknown_cat_ids.values()):,} annotations with unknown category_ids: "
            f"{dict(unknown_cat_ids)}. cat_id_mapping.py needs regeneration.",
            file=sys.stderr,
        )
        return 3

    bad_dim_images: list[tuple[int, str]] = []
    clipped_bboxes = 0
    skipped_zero_dim = 0
    written_lines = 0
    written_files = 0
    empty_files = 0

    for img_id, img in image_index.items():
        width = img.get("width")
        height = img.get("height")
        if not width or not height:
            bad_dim_images.append((img_id, img.get("file_name", "?")))
            continue

        anns = anns_by_image.get(img_id, [])
        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                skipped_zero_dim += 1
                continue

            x2, y2 = x + w, y + h
            if args.clip_bbox:
                x_clip = max(0.0, min(x, width))
                y_clip = max(0.0, min(y, height))
                x2_clip = max(0.0, min(x2, width))
                y2_clip = max(0.0, min(y2, height))
                if (x, y, x2, y2) != (x_clip, y_clip, x2_clip, y2_clip):
                    clipped_bboxes += 1
                x, y = x_clip, y_clip
                w, h = x2_clip - x_clip, y2_clip - y_clip
                if w <= 0 or h <= 0:
                    skipped_zero_dim += 1
                    continue
            else:
                if x < 0 or y < 0 or x2 > width or y2 > height:
                    print(
                        f"WARN: image_id={img_id} ({img.get('file_name')}): "
                        f"bbox ({x:.1f},{y:.1f},{w:.1f},{h:.1f}) extends past image "
                        f"({width}x{height}). Use --clip-bbox to auto-clip.",
                        file=sys.stderr,
                    )

            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            class_idx = cat_id_to_idx[int(ann["category_id"])]
            lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}")

        stem = Path(img["file_name"]).stem
        out_path = args.out_dir / f"{stem}.txt"

        if not args.dry_run:
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        written_files += 1
        written_lines += len(lines)
        if not lines:
            empty_files += 1

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files written       : {written_files:,}  ({'dry run' if args.dry_run else f'in {args.out_dir}'})")
    print(f"Total label lines   : {written_lines:,}")
    print(f"Empty label files   : {empty_files:,}  (images with no annotations -- intended for PU)")
    print(f"Bad-dim images      : {len(bad_dim_images):,}  (skipped, missing width/height in JSON)")
    print(f"Skipped zero-dim    : {skipped_zero_dim:,}  (bboxes with w<=0 or h<=0)")
    print(f"Clipped bboxes      : {clipped_bboxes:,}  ({'enabled' if args.clip_bbox else 'NOT clipped, see warnings'})")

    if bad_dim_images:
        print(f"\nFirst 5 bad-dim images: {bad_dim_images[:5]}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
