"""Multi-scale TTA wrapper around `predict_test_set.py` with WBF fusion.

Block reference: I.x in master_checklist.txt (Phase 7 EXP 7.5).

Why this exists
---------------
Per the May 1, 2026 host clarification (Kevin Barnard, see
notes/train_test_distribution.md), the FathomNet 2026 test set contains
resolutions that NEVER appear in training:
  - 720x486 (197 imgs!), 714x486, 720x366, 2048x1080, 1920x1079, 640x486

Training has only 1920x1080 and 3840x2160. Multi-scale TTA at inference
combines runs at several letterbox sizes and (optionally) horizontal flips,
then fuses boxes via WBF (Weighted Box Fusion).

Tier 1A (master_plan_v6): default scales include 480 and 1536; pass
``--include-hflip`` for an additional mirrored pass per scale (12 CSVs → WBF).

Pipeline
--------
1. For each view (scale [× hflip]):
     python scripts/predict_test_set.py --weights <PT> --imgsz <s> ...
2. python scripts/wbf_ensemble.py --submissions <list> --out ...

Compute
-------
Six scales without hflip is ~2× the old four-scale cost; adding hflip ~2× again.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
# Tier 1A extended TTA (v6): include small native test sizes + upper sweep.
DEFAULT_SCALES: List[int] = [480, 640, 832, 1024, 1280, 1536]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--weights", required=True, type=Path,
                   help="Path to trained .pt checkpoint")
    p.add_argument("--tag", required=True,
                   help="Tag for output CSVs (e.g. 'p2_mbari_e50'). "
                        "Output goes to submissions/<tag>_s<scale>.csv "
                        "and submissions/<tag>_multiscale_wbf.csv.")
    p.add_argument("--scales", type=int, nargs="+", default=DEFAULT_SCALES,
                   help=f"Scales to run (default {DEFAULT_SCALES}).")
    p.add_argument("--scale-weights", type=float, nargs="+", default=None,
                   help="Per-CSV WBF weights (same order as layered predictions: "
                        "each scale, optionally +hflip after it). Default: equal.")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "submissions",
                   help="Directory for output CSVs (default: submissions/)")
    p.add_argument("--test-images", type=Path,
                   default=REPO_ROOT / "data" / "raw" / "images" / "test",
                   help="Test image dir.")
    p.add_argument("--test-json", type=Path,
                   default=REPO_ROOT / "data" / "raw" / "test_dataset.json",
                   help="Test COCO JSON (used by both predict and WBF).")
    p.add_argument("--batch", type=int, default=16,
                   help="Inference batch size for predict_test_set.py.")
    p.add_argument("--conf", type=float, default=0.001,
                   help="Detection conf threshold for predict_test_set.py.")
    p.add_argument("--iou", type=float, default=0.65,
                   help="NMS IoU threshold for predict_test_set.py (per-scale).")
    p.add_argument("--max-det", type=int, default=100,
                   help="Max detections per image per scale.")
    p.add_argument("--wbf-iou", type=float, default=0.55,
                   help="WBF IoU threshold (cluster matching). Paper "
                        "recommendation 0.55 for COCO mAP@[.50:.95].")
    p.add_argument("--wbf-skip-thr", type=float, default=0.001,
                   help="WBF: drop input boxes below this score.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip per-scale prediction if output CSV already exists. "
                        "Useful for chaining onto post_training_pipeline.sh "
                        "which already produced the 640 CSV.")
    p.add_argument("--predict-script",
                   default=str(REPO_ROOT / "scripts" / "predict_test_set.py"),
                   help="Path to predict_test_set.py")
    p.add_argument("--wbf-script",
                   default=str(REPO_ROOT / "scripts" / "wbf_ensemble.py"),
                   help="Path to wbf_ensemble.py")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands but don't execute.")
    p.add_argument(
        "--include-hflip",
        action="store_true",
        help="Per scale, also run predict_test_set.py --hflip and fuse both "
             "CSV layers (Tier 1A). Doubles inference cost vs scales-only.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        metavar="N",
        help="Forwarded to predict_test_set.py — subset smoke (sorted filenames).",
    )
    p.add_argument(
        "--half",
        action="store_true",
        help="Forwarded to predict_test_set.py (--half FP16 inference).",
    )
    p.add_argument(
        "--device",
        default="",
        help="Forwarded to predict_test_set.py (e.g. 0 or cpu). Empty = omit.",
    )
    p.add_argument(
        "--tta",
        action="store_true",
        help="Forwarded to predict_test_set.py (Ultralytics augment TTA per scale).",
    )
    p.add_argument(
        "--per-class-conf-json",
        type=Path,
        default=None,
        help="Forwarded to predict_test_set.py (Tier 1C-i CSV filtering).",
    )
    p.add_argument(
        "--underwater-preproc",
        action="store_true",
        help="Forwarded to predict_test_set.py (Tier 1F / E4 / E9: underwater preproc).",
    )
    p.add_argument(
        "--underwater-method",
        choices=["msrcr", "clahe"],
        default="msrcr",
        help="Underwater preproc method (forwarded to predict_test_set.py).",
    )
    return p.parse_args()


def _run(cmd: list[str], dry_run: bool) -> int:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def _predict_extra_args(ns: argparse.Namespace) -> list[str]:
    """CLI fragments appended to each predict_test_set invocation."""
    extra: list[str] = []
    if ns.half:
        extra.append("--half")
    if ns.tta:
        extra.append("--tta")
    if ns.device:
        extra.extend(["--device", ns.device])
    if ns.max_images is not None:
        extra.extend(["--max-images", str(ns.max_images)])
    if getattr(ns, "per_class_conf_json", None) is not None:
        extra.extend(["--per-class-conf-json", str(ns.per_class_conf_json)])
    if getattr(ns, "underwater_preproc", False):
        extra.append("--underwater-preproc")
        method = getattr(ns, "underwater_method", "msrcr")
        extra.extend(["--underwater-method", method])
    return extra


def main() -> int:
    args = parse_args()
    if args.max_images is not None and args.max_images < 1:
        print("ERROR: --max-images must be >= 1", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_jobs: list[tuple[Path, list[str]]] = []
    for s in args.scales:
        csv_path = args.out_dir / f"{args.tag}_s{s}.csv"
        cmd_base = [
            sys.executable, args.predict_script,
            "--weights", str(args.weights),
            "--imgsz", str(s),
            "--batch", str(args.batch),
            "--conf", str(args.conf),
            "--iou", str(args.iou),
            "--max-det", str(args.max_det),
            "--test-images", str(args.test_images),
            "--test-json", str(args.test_json),
            "--quiet-ultralytics",
            *_predict_extra_args(args),
        ]
        csv_jobs.append((csv_path, cmd_base + ["--out", str(csv_path)]))
        if args.include_hflip:
            hf_path = args.out_dir / f"{args.tag}_s{s}_hflip.csv"
            csv_jobs.append(
                (hf_path, cmd_base + ["--out", str(hf_path), "--hflip"]),
            )

    n_csv = len(csv_jobs)
    if args.scale_weights is not None and len(args.scale_weights) != n_csv:
        print(
            f"ERROR: --scale-weights ({len(args.scale_weights)}) "
            f"must match number of prediction CSVs ({n_csv}).",
            file=sys.stderr,
        )
        return 2

    print("=" * 72)
    print("Multi-scale TTA + WBF inference pipeline")
    print("=" * 72)
    print(f"  Weights:    {args.weights}")
    print(f"  Tag:        {args.tag}")
    print(f"  Scales:     {args.scales}")
    print(f"  HFlip:      {args.include_hflip}")
    print(f"  Max images: {args.max_images if args.max_images is not None else 'all'}")
    print(f"  Half/tta:   half={args.half} tta={args.tta}")
    if args.device:
        print(f"  Device:     {args.device}")
    print(f"  CSV layers: {n_csv}")
    print(f"  Out dir:    {args.out_dir}")
    print(f"  Skip-exist: {args.skip_existing}")
    print(f"  Dry-run:    {args.dry_run}")
    print(f"  Per-class τ JSON: {args.per_class_conf_json or 'none'}")
    print()

    if not args.weights.exists() and not args.dry_run:
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr)
        return 2

    per_scale_csvs: list[Path] = []
    for csv_path, cmd in csv_jobs:
        per_scale_csvs.append(csv_path)
        if args.skip_existing and csv_path.exists():
            print(f"\n[SKIP] {csv_path} already exists; using as-is.")
            continue
        rc = _run(cmd, args.dry_run)
        if rc != 0:
            tag = csv_path.name
            print(f"\n[FAIL] predict_test_set.py exited {rc} ({tag})", file=sys.stderr)
            return rc

    if not args.dry_run:
        missing_csv = [p for p in per_scale_csvs if not p.exists()]
        if missing_csv:
            print(
                "\nERROR: Missing prediction CSV(s) before WBF "
                "(partial --skip-existing run or failed writes?):",
                file=sys.stderr,
            )
            for p in missing_csv:
                print(f"  - {p}", file=sys.stderr)
            return 2

    fused_csv = args.out_dir / f"{args.tag}_multiscale_wbf.csv"
    cmd = [
        sys.executable, args.wbf_script,
        "--submissions", *[str(p) for p in per_scale_csvs],
        "--out", str(fused_csv),
        "--test-json", str(args.test_json),
        "--iou-threshold", str(args.wbf_iou),
        "--skip-box-threshold", str(args.wbf_skip_thr),
    ]
    if args.scale_weights:
        cmd.extend(["--weights", *[str(w) for w in args.scale_weights]])
    rc = _run(cmd, args.dry_run)
    if rc != 0:
        print(f"\n[FAIL] wbf_ensemble.py exited {rc}", file=sys.stderr)
        return rc

    print("\n" + "=" * 72)
    print("MULTI-SCALE TTA DONE")
    print("=" * 72)
    print(f"Per-scale CSVs:")
    for p in per_scale_csvs:
        if not args.dry_run and p.exists():
            print(f"  - {p}  ({p.stat().st_size/1024:.1f} KB)")
        else:
            print(f"  - {p}")
    print(f"\nFused CSV:")
    print(f"  - {fused_csv}")
    print(f"\nSubmit with:")
    print(f"  kaggle competitions submit -c fathomnet-2026 \\")
    print(f"    -f {fused_csv} \\")
    print(
        f"    -m \"{args.tag} + multi-scale TTA WBF "
        f"(scales {args.scales}"
        f"{', hflip' if args.include_hflip else ''})\""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
