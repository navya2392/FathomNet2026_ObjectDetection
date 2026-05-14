"""E17 — Repeat Factor Sampling (RFS) retrain driver.

Strategy: instead of subclassing Ultralytics' DataLoader/Trainer to inject a
custom Sampler (brittle, version-dependent), we precompute LVIS-style repeat
factors and expand the training-file list — each image path appears
``round(r_i)`` times. Ultralytics then trains normally on the expanded list,
giving the same per-image expected sampling frequency as a true RFS sampler
would.

Trade-off: this is "dataset-level" RFS rather than "epoch-level" stochastic-
rounded RFS. The expectation matches; the variance is tighter (no
stochastic rounding noise). For our use case (rare classes oversampled by
~5-15x) the difference is negligible and the robustness gain is large.

Pipeline:
  1. compute_repeat_factors(train_dataset.json, threshold=0.05)
  2. read fold{F}_train.txt
  3. For each path, look up its image_id in the COCO json by file_name,
     fetch r_i, append ``round(r_i)`` copies to the expanded train file
  4. Write a temp fold YAML pointing at the expanded train file (val unchanged)
  5. yolo.train(...) with anchor-recipe hyperparameters

Usage:
  python scripts/train_rfs.py \\
      --init <ANCHOR_BEST_PT> \\
      --fold 0 \\
      --rfs-threshold 0.05 \\
      --epochs 25 --imgsz 1024 --batch 8 \\
      --name e17_rfs_t005_fold0 \\
      --no-wandb
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # type: ignore

from src.rfs_sampler import compute_repeat_factors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="Initial weights (anchor best.pt)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--rfs-threshold", type=float, default=0.05,
                   help="LVIS RFS t hyperparameter; 0.05 oversamples categories with <323 images")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--name", default="e17_rfs_t005_fold0")
    p.add_argument("--project", default="runs/detect/weights/runs")
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs, batch=2, workers=0 — for fast verification")
    p.add_argument(
        "--train-json",
        default=str(REPO_ROOT / "data" / "raw" / "train_dataset.json"),
        help="COCO-format train_dataset.json (used by RFS to compute factors)",
    )
    p.add_argument(
        "--folds-dir",
        default=str(REPO_ROOT / "data" / "folds"),
        help="Directory containing fold{F}.yaml + fold{F}_train.txt + fold{F}_val.txt",
    )
    p.add_argument(
        "--out-folds-dir",
        default=str(REPO_ROOT / "data" / "folds_rfs"),
        help="Directory where the expanded fold YAML + train file will be written",
    )
    return p.parse_args()


def _build_filename_to_image_id(coco: dict) -> dict[str, int]:
    return {img["file_name"]: int(img["id"]) for img in coco.get("images", [])}


def _expand_train_file(
    fold_train_txt: Path,
    image_repeat_factors: dict[int, float],
    filename_to_image_id: dict[str, int],
    out_path: Path,
) -> dict[str, int]:
    """Write an expanded fold-train file where each path appears round(r_i) times.

    Returns a small histogram describing the expansion.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_out = 0
    n_missing = 0
    repeat_hist: dict[int, int] = {}

    with fold_train_txt.open() as f_in, out_path.open("w") as f_out:
        for raw_line in f_in:
            line = raw_line.strip()
            if not line:
                continue
            n_in += 1
            fname = Path(line).name
            iid = filename_to_image_id.get(fname)
            if iid is None:
                n_missing += 1
                f_out.write(line + "\n")
                n_out += 1
                continue
            r = image_repeat_factors.get(iid, 1.0)
            r_int = max(1, int(round(r)))
            repeat_hist[r_int] = repeat_hist.get(r_int, 0) + 1
            for _ in range(r_int):
                f_out.write(line + "\n")
                n_out += 1
    return {
        "n_input": n_in,
        "n_output": n_out,
        "expansion_ratio": (n_out / max(1, n_in)),
        "n_missing": n_missing,
        "repeat_histogram": repeat_hist,
    }


def _write_fold_yaml(
    src_yaml: Path,
    out_yaml: Path,
    new_train_txt: Path,
) -> None:
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

    train_json = Path(args.train_json)
    folds_dir = Path(args.folds_dir)
    out_folds_dir = Path(args.out_folds_dir)

    src_yaml = folds_dir / f"fold{args.fold}.yaml"
    src_train_txt = folds_dir / f"fold{args.fold}_train.txt"
    if not src_yaml.exists():
        print(f"[fatal] fold yaml missing: {src_yaml}", file=sys.stderr)
        return 1
    if not src_train_txt.exists():
        print(f"[fatal] fold train list missing: {src_train_txt}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"E17 RFS retrain — {args.name}")
    print(f"  Init:           {args.init}")
    print(f"  Source fold:    {src_yaml}")
    print(f"  RFS threshold:  t = {args.rfs_threshold}")
    print(f"  Epochs:         {args.epochs}    imgsz: {args.imgsz}    batch: {args.batch}")
    print("=" * 78)

    print("\n[1/4] Computing repeat factors ...")
    rfs = compute_repeat_factors(train_json, threshold=args.rfs_threshold, verbose=True)

    print("\n[2/4] Expanding fold-train file ...")
    with train_json.open() as f:
        coco = json.load(f)
    filename_to_image_id = _build_filename_to_image_id(coco)

    out_train_txt = out_folds_dir / f"fold{args.fold}_train_rfs_t{int(args.rfs_threshold*1000):03d}.txt"
    expand_stats = _expand_train_file(
        src_train_txt, rfs.image_repeat_factors, filename_to_image_id, out_train_txt,
    )
    print(f"  Wrote {out_train_txt}")
    print(f"  Input lines:    {expand_stats['n_input']:,}")
    print(f"  Output lines:   {expand_stats['n_output']:,}  "
          f"(x{expand_stats['expansion_ratio']:.2f})")
    print(f"  Missing in JSON:{expand_stats['n_missing']:,}")
    print(f"  Repeat hist:    {expand_stats['repeat_histogram']}")

    print("\n[3/4] Writing temp fold YAML ...")
    out_yaml = out_folds_dir / f"fold{args.fold}.yaml"
    _write_fold_yaml(src_yaml, out_yaml, out_train_txt)
    # Copy val list verbatim (same evaluation set as anchor for fair comparison).
    src_val_txt = folds_dir / f"fold{args.fold}_val.txt"
    if src_val_txt.exists():
        shutil.copy(src_val_txt, out_folds_dir / src_val_txt.name)
    print(f"  Wrote {out_yaml}")

    print("\n[4/4] Launching Ultralytics training ...")
    from ultralytics import YOLO  # delayed import for fast --help

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
        # Anchor-matching aug, mosaic halved per UH1 finding (high-mosaic edge density worsens train/test gap).
        mosaic=0.5,
        mixup=0.15,
        copy_paste=0.3,
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
