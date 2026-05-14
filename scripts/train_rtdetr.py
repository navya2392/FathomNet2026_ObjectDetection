#!/usr/bin/env python3
"""Train RT-DETR-l on a fold for E10 (Real-Time DEtection TRansformer baseline).

E10 in the v7 master plan: a transformer-based detector trained from scratch
(COCO init only — RT-DETR cannot consume YOLO-format MBARI 315k weights).
The single-model LB is expected to be 0.04-0.08 (well below the YOLOv8x
anchor at 0.0863); the value is in **architectural diversity** for the final
multi-arch WBF ensemble (E25). Cross-architecture predictions correlate less
with the YOLO family, so even an inferior single model can lift a WBF stack
by 0.01-0.02 LB.

Architecture choice
-------------------
* RT-DETR-l (~36M params) — lighter than YOLOv8x (68M), fits 1024×1024 b=8
  comfortably on a 32 GB GPU.
* Some literature suggests RT-DETR-x (~67M) for larger datasets, but at our
  scale (5 k labeled images) the smaller model regularizes better.

Augmentation defaults
---------------------
RT-DETR transformer head dislikes some aug modes that YOLO loves:
  * mosaic=1.0 with close_mosaic=10 (turn off in last 10 epochs) — keep, helps
  * mixup=0.0   — RT-DETR docs recommend off (set-prediction loss confused
                  by mixed targets)
  * erasing=0.0 — same reason
  * hsv_h, hsv_s, hsv_v, fliplr — standard, keep
  * degrees=0.0 translate=0.1 scale=0.5 — milder than YOLO heavy aug

The user can override any of these via CLI flags.

Usage
-----
    # Smoke test (2 epochs, batch=2)
    python scripts/train_rtdetr.py --epochs 2 --batch 2 --imgsz 640 --name e10_smoke

    # Full run (50 epochs, fold 0)
    python scripts/train_rtdetr.py --epochs 50 --imgsz 1024 --batch 8 --fold 0 \\
        --name e10_rtdetr_l_fold0

References
----------
* Lyu et al. "DETRs Beat YOLOs on Real-time Object Detection" (CVPR 2024)
  https://arxiv.org/abs/2304.08069
* Ultralytics RT-DETR docs: https://docs.ultralytics.com/models/rtdetr/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402  YOLO class also dispatches RT-DETR via .pt name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--init", default="rtdetr-l.pt",
                   help="Init weights. 'rtdetr-l.pt' auto-downloads from Ultralytics. "
                        "Pass a local .pt to resume from a previous run.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--name", default="e10_rtdetr_l_fold0")
    p.add_argument("--project", default="runs/detect/weights/runs")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--workers", type=int, default=8)
    # RT-DETR-friendly aug defaults
    p.add_argument("--mosaic", type=float, default=0.5,
                   help="Lowered from 1.0 per UH1 finding (texture/edge gap dominates train/test mismatch).")
    p.add_argument("--close-mosaic", type=int, default=10,
                   help="Disable mosaic in the last N epochs (0 = always on).")
    p.add_argument("--mixup", type=float, default=0.0,
                   help="RT-DETR set-prediction loss is confused by mixed targets; default off.")
    p.add_argument("--erasing", type=float, default=0.0)
    p.add_argument("--degrees", type=float, default=0.0)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--hsv-h", type=float, default=0.015)
    p.add_argument("--hsv-s", type=float, default=0.7)
    p.add_argument("--hsv-v", type=float, default=0.4)
    p.add_argument("--lr0", type=float, default=1e-4,
                   help="RT-DETR uses AdamW; default LR much smaller than SGD's 1e-2.")
    p.add_argument("--cos-lr", action="store_true", default=True)
    p.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")
    p.add_argument("--save-period", type=int, default=10)
    p.add_argument("--smoke", action="store_true",
                   help="2 ep, batch=2, imgsz=640 — sanity check only.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.epochs = 2
        args.batch = 2
        args.imgsz = 640
        args.workers = 0
        args.patience = 100  # don't early-stop a smoke

    fold_yaml = REPO_ROOT / "data" / "folds" / f"fold{args.fold}.yaml"
    if not fold_yaml.exists():
        print(f"[fatal] Fold YAML missing: {fold_yaml}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"E10 RT-DETR training — {args.name}")
    print(f"  Init:          {args.init}")
    print(f"  Fold YAML:     {fold_yaml}")
    print(f"  Epochs:        {args.epochs}    imgsz: {args.imgsz}    batch: {args.batch}")
    print(f"  patience:      {args.patience}    workers: {args.workers}")
    print(f"  mosaic={args.mosaic}  close_mosaic={args.close_mosaic}  "
          f"mixup={args.mixup}  erasing={args.erasing}")
    print(f"  degrees={args.degrees}  translate={args.translate}  scale={args.scale}")
    print(f"  fliplr={args.fliplr}  hsv=({args.hsv_h},{args.hsv_s},{args.hsv_v})")
    print(f"  lr0={args.lr0}    cos_lr={args.cos_lr}")
    print(f"  Save period:   every {args.save_period} ep")
    print("=" * 78)

    print("\n[1/2] Loading model ...")
    model = YOLO(args.init)
    print(f"  Loaded type: {type(model.model).__name__}")
    print(f"  Params: {sum(p.numel() for p in model.model.parameters())/1e6:.2f}M")

    print("\n[2/2] Training ...")
    t0 = time.time()
    results = model.train(
        data=str(fold_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
        cos_lr=args.cos_lr,
        patience=args.patience,
        workers=args.workers,
        save=True,
        save_period=args.save_period,
        amp=True,
        # Augmentation
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        erasing=args.erasing,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        lr0=args.lr0,
    )
    wall_min = (time.time() - t0) / 60.0
    print(f"\nDone in {wall_min:.1f} min")

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    print(f"  best.pt: {best_pt} (exists={best_pt.exists()})")
    print(f"  last.pt: {last_pt} (exists={last_pt.exists()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
