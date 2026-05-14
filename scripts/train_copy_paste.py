"""E19 — Class-aware copy-paste retrain driver.

Strategy: pre-generate K augmented copies of each training image whose label
contains at least one rare-class instance. Each copy has class-aware
copy-paste applied (rare-class crops pasted at non-overlapping locations
with per-class probability from build_default_schedule). Augmented images
+ updated YOLO label files are written to data/cp_aug/, and an expanded
fold-train file is generated that includes the originals plus the
augmented copies.

Trade-off vs runtime augmentation: we lose some per-batch random diversity
(each augmented copy is fixed for the run rather than re-rolled per epoch),
but gain robustness — no Ultralytics internal subclassing, no per-version
fragility. Mosaic/mixup at training time still produce inter-image diversity
on top of the static paste-augmented copies.

Pipeline:
  1. Read train_dataset.json + class-counts; build paste schedule
  2. build_instance_bank() over all training annotations
  3. For each image in fold{F}_train.txt:
       - Load image + YOLO label
       - If contains any rare-class instance (count <= rare_threshold),
         apply class_aware_copy_paste K times -> save K augmented (img, lbl)
  4. Write expanded fold-train file: originals + augmented copies
  5. yolo.train(...) with anchor-recipe hyperparameters; built-in copy_paste=0
     to avoid double-stacking (this experiment IS class-aware copy-paste).

Usage:
  python scripts/train_copy_paste.py \\
      --init <ANCHOR_BEST_PT> \\
      --fold 0 \\
      --augments-per-image 3 \\
      --epochs 25 --imgsz 1024 --batch 8 \\
      --name e19_copy_paste_fold0 \\
      --no-wandb
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # type: ignore
import numpy as np  # type: ignore
import yaml  # type: ignore

from src.copy_paste import (
    build_default_schedule,
    build_instance_bank,
    class_aware_copy_paste,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--augments-per-image", type=int, default=3,
                   help="K = how many augmented copies per rare-class image")
    p.add_argument("--rare-threshold", type=int, default=300,
                   help="A class is 'rare' if its train instance count <= this. "
                        "Images touching any rare class get pre-augmented.")
    p.add_argument("--max-paste-per-class", type=int, default=1)
    p.add_argument("--max-iou", type=float, default=0.1,
                   help="A paste location is rejected if IoU > this with any existing box")
    p.add_argument("--name", default="e19_copy_paste_fold0")
    p.add_argument("--project", default="runs/detect/weights/runs")
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--train-json",
        default=str(REPO_ROOT / "data" / "raw" / "train_dataset.json"),
    )
    p.add_argument(
        "--train-image-root",
        default=str(REPO_ROOT / "data" / "raw" / "images" / "train"),
    )
    p.add_argument(
        "--train-label-root",
        default=str(REPO_ROOT / "data" / "raw" / "labels" / "train"),
    )
    p.add_argument(
        "--folds-dir",
        default=str(REPO_ROOT / "data" / "folds"),
    )
    p.add_argument(
        "--out-folds-dir",
        default=str(REPO_ROOT / "data" / "folds_cp"),
    )
    p.add_argument(
        "--aug-dir",
        default=str(REPO_ROOT / "data" / "cp_aug"),
        help="Directory to write augmented images + labels into",
    )
    return p.parse_args()


def _read_yolo_labels(label_path: Path) -> tuple[list[int], list[tuple[float, float, float, float]]]:
    """Read a YOLO label .txt -> (class_indices, boxes_xywh_normalized)."""
    if not label_path.exists():
        return [], []
    classes, boxes = [], []
    for raw in label_path.read_text().splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        c = int(parts[0])
        cx, cy, w, h = (float(parts[i]) for i in range(1, 5))
        classes.append(c)
        boxes.append((cx, cy, w, h))
    return classes, boxes


def _yolo_xywhn_to_pixel(
    box_xywhn: tuple[float, float, float, float],
    img_h: int,
    img_w: int,
) -> tuple[int, int, int, int]:
    """YOLO normalized cx,cy,w,h -> pixel x,y,w,h (top-left)."""
    cx, cy, w, h = box_xywhn
    px = int(round((cx - w / 2.0) * img_w))
    py = int(round((cy - h / 2.0) * img_h))
    pw = int(round(w * img_w))
    ph = int(round(h * img_h))
    return (max(0, px), max(0, py), max(1, pw), max(1, ph))


def _pixel_to_yolo_xywhn(
    box_xywh_px: tuple[int, int, int, int],
    img_h: int,
    img_w: int,
) -> tuple[float, float, float, float]:
    x, y, w, h = box_xywh_px
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    return (cx, cy, w / img_w, h / img_h)


def _per_class_instance_counts(coco: dict, cat_id_to_idx: dict[int, int]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for ann in coco.get("annotations", []):
        cat_id = int(ann.get("category_id", -1))
        if cat_id in cat_id_to_idx:
            counts[cat_id_to_idx[cat_id]] += 1
    return dict(counts)


def _generate_augments(
    args: argparse.Namespace,
    src_train_txt: Path,
    instance_counts: dict[int, int],
    schedule: dict[int, float],
    bank,
    rng: random.Random,
) -> list[Path]:
    """Apply class-aware copy-paste to rare-class images and persist outputs.

    Returns the list of augmented .txt paths to be appended to the expanded
    fold-train file.
    """
    aug_dir = Path(args.aug_dir)
    aug_img_dir = aug_dir / "images"
    aug_lbl_dir = aug_dir / "labels"
    aug_img_dir.mkdir(parents=True, exist_ok=True)
    aug_lbl_dir.mkdir(parents=True, exist_ok=True)

    train_label_root = Path(args.train_label_root)
    rare_classes = {c for c, n in instance_counts.items() if n <= args.rare_threshold}
    print(f"  Rare classes (n <= {args.rare_threshold}): {sorted(rare_classes)}")

    augmented_paths: list[Path] = []
    n_skipped_no_rare = 0
    n_processed = 0
    n_paste_total = 0

    for raw in src_train_txt.read_text().splitlines():
        path = raw.strip()
        if not path:
            continue
        img_path = Path(path)
        stem = img_path.stem
        lbl_path = train_label_root / f"{stem}.txt"

        classes, boxes_n = _read_yolo_labels(lbl_path)
        if not any(c in rare_classes for c in classes):
            n_skipped_no_rare += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes_px = [_yolo_xywhn_to_pixel(b, H, W) for b in boxes_n]

        for k in range(args.augments_per_image):
            out_img, out_boxes_px, out_labels = class_aware_copy_paste(
                img,
                boxes_px,
                classes,
                bank,
                schedule,
                max_paste_per_class=args.max_paste_per_class,
                max_iou=args.max_iou,
                max_attempts_per_paste=30,
                rng=rng,
                verbose=False,
            )
            new_img_path = aug_img_dir / f"{stem}_cp{k}.jpg"
            new_lbl_path = aug_lbl_dir / f"{stem}_cp{k}.txt"
            cv2.imwrite(str(new_img_path), out_img)
            n_paste_total += max(0, len(out_labels) - len(classes))
            with new_lbl_path.open("w") as f:
                for cls_idx, b_px in zip(out_labels, out_boxes_px):
                    cx, cy, w, h = _pixel_to_yolo_xywhn(b_px, H, W)
                    f.write(f"{int(cls_idx)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            augmented_paths.append(new_img_path)
        n_processed += 1

    print(f"  Augmented {n_processed} rare-touching images -> {len(augmented_paths)} new copies "
          f"({n_paste_total} pastes, skipped {n_skipped_no_rare} non-rare)")
    return augmented_paths


def _write_expanded_train_txt(
    src_train_txt: Path,
    augmented_paths: list[Path],
    out_path: Path,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f_out:
        with src_train_txt.open() as f_in:
            for raw in f_in:
                line = raw.strip()
                if line:
                    f_out.write(line + "\n")
                    n += 1
        for p in augmented_paths:
            f_out.write(str(p.resolve()) + "\n")
            n += 1
    return n


def _write_fold_yaml(src_yaml: Path, out_yaml: Path, new_train_txt: Path) -> None:
    with src_yaml.open() as f:
        cfg = yaml.safe_load(f)
    cfg["train"] = str(new_train_txt.resolve())
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> int:
    args = parse_args()

    if args.smoke:
        args.epochs = 2
        args.batch = 2
        args.workers = 0
        args.augments_per_image = 1

    rng = random.Random(args.seed)

    train_json = Path(args.train_json)
    folds_dir = Path(args.folds_dir)
    out_folds_dir = Path(args.out_folds_dir)
    src_yaml = folds_dir / f"fold{args.fold}.yaml"
    src_train_txt = folds_dir / f"fold{args.fold}_train.txt"
    if not src_yaml.exists() or not src_train_txt.exists():
        print(f"[fatal] fold inputs missing in {folds_dir}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"E19 class-aware copy-paste retrain — {args.name}")
    print(f"  Init:                {args.init}")
    print(f"  Source fold:         {src_yaml}")
    print(f"  Augments / image:    {args.augments_per_image}")
    print(f"  Rare-class threshold: <= {args.rare_threshold} train instances")
    print(f"  Max paste / class:   {args.max_paste_per_class}    max IoU: {args.max_iou}")
    print(f"  Epochs: {args.epochs}    imgsz: {args.imgsz}    batch: {args.batch}")
    print("=" * 78)

    print("\n[1/5] Building instance bank ...")
    from configs.cat_id_mapping import cat_id_to_idx
    with train_json.open() as f:
        coco = json.load(f)
    bank = build_instance_bank(
        train_json,
        Path(args.train_image_root),
        cat_id_to_idx=cat_id_to_idx,
        verbose=True,
    )

    print("\n[2/5] Computing per-class instance counts + paste schedule ...")
    instance_counts = _per_class_instance_counts(coco, cat_id_to_idx)
    print(f"  classes seen: {len(instance_counts)}; "
          f"min={min(instance_counts.values())}, max={max(instance_counts.values())}")
    schedule = build_default_schedule(instance_counts)
    print(f"  schedule (top-5 highest paste prob):")
    for cid, p in sorted(schedule.items(), key=lambda x: -x[1])[:5]:
        print(f"    class {cid:3d}: paste_prob={p:.2f}  (n={instance_counts[cid]})")

    print("\n[3/5] Generating augmented copies ...")
    augmented_paths = _generate_augments(
        args, src_train_txt, instance_counts, schedule, bank, rng,
    )

    print("\n[4/5] Writing expanded fold-train file + YAML ...")
    out_train_txt = out_folds_dir / f"fold{args.fold}_train_cp_k{args.augments_per_image}.txt"
    n_total = _write_expanded_train_txt(src_train_txt, augmented_paths, out_train_txt)
    out_yaml = out_folds_dir / f"fold{args.fold}.yaml"
    _write_fold_yaml(src_yaml, out_yaml, out_train_txt)
    src_val_txt = folds_dir / f"fold{args.fold}_val.txt"
    if src_val_txt.exists():
        shutil.copy(src_val_txt, out_folds_dir / src_val_txt.name)
    print(f"  Wrote {out_train_txt} ({n_total:,} lines)")
    print(f"  Wrote {out_yaml}")

    print("\n[5/5] Launching Ultralytics training ...")
    from ultralytics import YOLO

    model = YOLO(args.init)
    train_kwargs = dict(
        data=str(out_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=False,
        patience=args.patience,
        seed=args.seed,
        cos_lr=True,
        close_mosaic=10,
        lr0=args.lr0,
        lrf=args.lrf,
        mosaic=0.5,
        mixup=0.15,
        # IMPORTANT: built-in copy_paste = 0 here. We've already pre-augmented
        # with class-aware paste; stacking Ultralytics' uniform copy_paste on
        # top would conflate the experiment.
        copy_paste=0.0,
        degrees=10.0,
        translate=0.2,
        scale=0.75,
        fliplr=0.5,
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    if args.no_wandb:
        import os
        os.environ["WANDB_DISABLED"] = "true"
    print(f"[ultralytics] model.train({train_kwargs})")
    model.train(**train_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
