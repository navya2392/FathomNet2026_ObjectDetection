#!/usr/bin/env python3
"""E27 — Cross-architecture ensemble of E1 (YOLOv8x anchor) + T2 (RT-DETR-l)
+ T3 (YOLOv8x w/ hier-aux) via Weighted Box Fusion.

Why this experiment exists
--------------------------
Every single source-side optimization we have tried (E2-E9, T1) has scored
*below* the E1 anchor baseline (LB 0.0863) because they either over-fit
SOI/NOAA train statistics or amplify bad pseudo-labels on the MBARI test set.

The strategic pivot (May 3 2026, see docs/aggressive_master_plan_v7.md) is:
diversity > strength. We build three *different* detectors and let WBF
average their box outputs. Each model's individual biases (YOLOv8x vs
RT-DETR transformer vs hier-aux supervised classifier) are partially
uncorrelated, so the average can beat the best individual.

Inputs (the driver does NOT do TTA itself; it expects fused per-model CSVs)
---------------------------------------------------------------------------
By the time you run this, the autonomous chain has already produced:

  submissions/e1_aug1024_tta6_multiscale_wbf.csv         (E1, LB 0.0863)
  submissions/t2_rtdetr_l_fold0_multiscale_wbf.csv       (T2, RT-DETR-l)
  submissions/t3_hier_aux_fold0_multiscale_wbf.csv       (T3, Hier-aux)

Each is the output of `predict_test_multiscale_tta.py` (6-scale TTA + WBF).

What this script does
---------------------
1. Verifies all input CSVs exist.
2. Runs `wbf_ensemble.py` across them with multiple weight schedules:
   - Equal weights (1, 1, 1)
   - E1-heavy (2, 1, 1)  -- E1 anchor is the proven strongest single model
   - E1-anchored (3, 1, 1) -- only nudge E1 with the other two
3. Optionally auto-submits the equal-weight CSV to Kaggle.

Output naming
-------------
  submissions/e27_cross_arch_w<W1><W2><W3>_wbf.csv
  e.g. e27_cross_arch_w111_wbf.csv, e27_cross_arch_w211_wbf.csv

Usage
-----
    # Default: 3 weight configs, all 3 CSVs auto-located
    python scripts/cross_arch_ensemble.py

    # With auto-submit of the equal-weight result
    python scripts/cross_arch_ensemble.py --submit

    # Custom CSVs (e.g. add a 4th model)
    python scripts/cross_arch_ensemble.py \
        --inputs submissions/e1_*.csv submissions/t2_*.csv \
                 submissions/t3_*.csv submissions/e9_*.csv \
        --weights-grid "1,1,1,1" "2,1,1,1"

Block reference: E27 in docs/aggressive_master_plan_v7.md (post-T3 stack layer).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUTS = [
    REPO_ROOT / "submissions" / "e1_aug1024_tta6_multiscale_wbf.csv",
    REPO_ROOT / "submissions" / "t2_rtdetr_l_fold0_multiscale_wbf.csv",
    REPO_ROOT / "submissions" / "t3_hier_aux_fold0_multiscale_wbf.csv",
]

# Weight schedules to try. Each tuple is (label, weights). The first one
# is auto-submitted by default if --submit is passed.
DEFAULT_WEIGHT_GRID: list[tuple[str, list[float]]] = [
    ("111", [1.0, 1.0, 1.0]),  # equal — the baseline ensemble config
    ("211", [2.0, 1.0, 1.0]),  # E1-heavy — anchor proven best
    ("311", [3.0, 1.0, 1.0]),  # E1-dominant — only mild diversity injection
]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Per-model fused submission CSVs (output of predict_test_multiscale_tta.py).",
    )
    p.add_argument(
        "--weights-grid",
        nargs="+",
        default=None,
        help="Comma-separated weight tuples to try (e.g. '1,1,1' '2,1,1'). "
             "Default: equal, E1-heavy, E1-dominant.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "submissions",
        help="Output directory for fused CSVs.",
    )
    p.add_argument(
        "--out-prefix",
        default="e27_cross_arch",
        help="Output filename prefix (default: e27_cross_arch).",
    )
    p.add_argument(
        "--wbf-iou", type=float, default=0.55,
        help="WBF cluster matching IoU threshold (default 0.55).",
    )
    p.add_argument(
        "--wbf-skip-thr", type=float, default=0.001,
        help="WBF: drop input boxes below this score (default 0.001).",
    )
    p.add_argument(
        "--wbf-script",
        default=str(REPO_ROOT / "scripts" / "wbf_ensemble.py"),
        help="Path to wbf_ensemble.py.",
    )
    p.add_argument(
        "--test-json",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "test_dataset.json",
        help="Test COCO JSON for image dimensions.",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="After fusion, auto-submit the FIRST weight schedule's CSV to Kaggle.",
    )
    p.add_argument(
        "--submit-tag",
        default=None,
        help="Override which weight tag (e.g. '111', '211') to auto-submit. "
             "Default: first in --weights-grid order.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands but don't execute.",
    )
    return p.parse_args(argv)


def _parse_weights(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _run(cmd: list[str], dry_run: bool) -> int:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inputs: list[Path] = [Path(p) for p in args.inputs]
    if len(inputs) < 2:
        print("ERROR: need at least 2 input CSVs to fuse.", file=sys.stderr)
        return 2
    missing = [p for p in inputs if not p.exists()]
    if missing and not args.dry_run:
        print("ERROR: missing input CSV(s):", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        return 2

    if args.weights_grid:
        weight_grid: list[tuple[str, list[float]]] = []
        for spec in args.weights_grid:
            w = _parse_weights(spec)
            if len(w) != len(inputs):
                print(
                    f"ERROR: weight spec '{spec}' has {len(w)} entries, "
                    f"need {len(inputs)} (one per input).",
                    file=sys.stderr,
                )
                return 2
            tag = "".join(str(int(round(x))) for x in w)
            weight_grid.append((tag, w))
    else:
        weight_grid = DEFAULT_WEIGHT_GRID
        if len(inputs) != 3:
            print(
                f"ERROR: default weight grid is for 3 inputs, got {len(inputs)}. "
                f"Pass --weights-grid explicitly.",
                file=sys.stderr,
            )
            return 2

    print("=" * 72)
    print("E27 cross-architecture ensemble (WBF)")
    print("=" * 72)
    print(f"  Inputs ({len(inputs)}):")
    for p in inputs:
        size = p.stat().st_size / 1024 if p.exists() else 0.0
        print(f"    - {p}  ({size:.1f} KB)" if size else f"    - {p}")
    print(f"  WBF iou:           {args.wbf_iou}")
    print(f"  WBF skip-box-thr:  {args.wbf_skip_thr}")
    print(f"  Weight schedules:  {[t for t, _ in weight_grid]}")
    print(f"  Out dir:           {args.out_dir}")
    print(f"  Auto-submit:       {args.submit}")
    print()

    fused_paths: dict[str, Path] = {}
    for tag, weights in weight_grid:
        out_csv = args.out_dir / f"{args.out_prefix}_w{tag}_wbf.csv"
        cmd = [
            sys.executable, args.wbf_script,
            "--submissions", *[str(p) for p in inputs],
            "--weights", *[str(w) for w in weights],
            "--out", str(out_csv),
            "--test-json", str(args.test_json),
            "--iou-threshold", str(args.wbf_iou),
            "--skip-box-threshold", str(args.wbf_skip_thr),
        ]
        rc = _run(cmd, args.dry_run)
        if rc != 0:
            print(f"\n[FAIL] WBF for weights {weights} exited {rc}", file=sys.stderr)
            return rc
        fused_paths[tag] = out_csv

    print("\n" + "=" * 72)
    print("E27 ensemble done")
    print("=" * 72)
    for tag, p in fused_paths.items():
        size = p.stat().st_size / 1024 if (p.exists() and not args.dry_run) else 0.0
        print(f"  w{tag}: {p}  ({size:.1f} KB)" if size else f"  w{tag}: {p}")

    if args.submit and not args.dry_run:
        submit_tag = args.submit_tag or weight_grid[0][0]
        if submit_tag not in fused_paths:
            print(f"\nERROR: --submit-tag '{submit_tag}' not produced by this run.",
                  file=sys.stderr)
            return 2
        submit_csv = fused_paths[submit_tag]
        msg = (f"E27 cross-arch ensemble (E1 + T2 RT-DETR + T3 hier-aux), "
               f"WBF weights w{submit_tag}")
        cmd = [
            "kaggle", "competitions", "submit",
            "-c", "fathomnet-2026",
            "-f", str(submit_csv),
            "-m", msg,
        ]
        rc = _run(cmd, args.dry_run)
        if rc != 0:
            print(f"\n[FAIL] kaggle submit exited {rc}", file=sys.stderr)
            return rc
        print(f"\nSubmitted: {submit_csv}")
    else:
        print("\nNot auto-submitting. To submit one manually, e.g.:")
        first_tag, _ = weight_grid[0]
        first_csv = fused_paths[first_tag]
        print(f"  kaggle competitions submit -c fathomnet-2026 \\")
        print(f"    -f {first_csv} \\")
        print(f"    -m \"E27 cross-arch ensemble w{first_tag}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
