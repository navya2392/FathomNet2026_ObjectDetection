"""Phase 2 EXP 2.3 -- 5-way multi-pretrained-init bake-off (the v5.5 differentiator).

Block reference: C.5 in master_checklist.txt (v5.5 expansion).

What this script does
---------------------
Trains FIVE candidate detector inits identically (same fold, same epochs,
same hyperparams) on the FathomNet 2026 dataset and emits a ranked table
by val mAP@[.50:.95]. The winner becomes the PRIMARY init for Phase 3-7;
the runner-up becomes the SECONDARY init for Phase 6 ensemble diversity.

Why this experiment exists
--------------------------
v5.5 strategic pivot (May 1 ~00:25 PT): Laura Chrobak (host) clarified
that publicly-available marine-life detection models are allowed for
initialization with disclosure. Marine-pretrained checkpoints (Megalodon,
MBARI 315k, Megafishdetector) sit much closer to FathomNet's data
distribution than COCO weights do, so we expect them to dominate.
Most teams will NOT bother trying multi-init -- this is a cheap
($10 GPU, ~5 hr) way to gain 0.05+ mAP for free.

The five candidates
-------------------
| # | Init                       | Architecture | Pretraining           | Why considered                         |
|---|----------------------------|--------------|-----------------------|----------------------------------------|
| 1 | yolo11m.pt                 | YOLOv11m     | MS-COCO 2017           | Vanilla baseline anchor                |
| 2 | bioclip2 (wrapper)         | YOLOv11m+ViT | TreeOfLife-10M biology | Biology-domain ViT features (D.4)      |
| 3 | weights/marine_models/megalodon/best.pt | YOLOv8x | All FathomNet (single class) | FathomNet single-class objectness   |
| 4 | weights/marine_models/mbari_315k/best.pt | YOLOv8 | 315k MBARI deep-sea, multi-class | STRONGEST a priori candidate    |
| 5 | weights/marine_models/megafishdetector/megafishdetector_v0_yolov5m_1280p.pt | YOLOv5m | Multi-source fish | Fish-class subset benefit       |

Output
------
  notes/phase2_init_bakeoff.md    -- ranked table, ready to commit
  weights/runs/p2_bakeoff/<init>/  -- per-init Ultralytics run dirs
  W&B runs tagged ['phase2', 'bakeoff', <init_label>]

Decision rule (codified from v5.5 master plan EXP 2.3)
------------------------------------------------------
  - Take the TOP 2 by val mAP@[.50:.95] at the screening epoch count.
  - If rank[0].val_mAP < 0.15: pipeline bug. STOP and debug.
  - If rank[0] - rank[1] gap < 0.02: keep rank[2] for ensemble.
  - If rank[0] - rank[1] gap > 0.05: drop rank[1], use only rank[0].

Then promote winners to full 50-epoch training via train_phase2_baseline.py.

Usage
-----
    # Default screening sprint: 10 epochs each on fold 0 (~5 hr GPU on 4090)
    python scripts/eval_pretrained_inits.py

    # Faster smoke test: 3 epochs each (~30 min GPU)
    python scripts/eval_pretrained_inits.py --epochs 3

    # Skip BioCLIP2 (if Ultralytics integration fails on your setup)
    python scripts/eval_pretrained_inits.py --skip bioclip2

    # Only run a subset (useful for re-running after a failure)
    python scripts/eval_pretrained_inits.py --only mbari_315k megalodon

    # Override fold / batch / device
    python scripts/eval_pretrained_inits.py --fold 0 --batch 16 --device 0

    # Skip W&B (offline runs)
    python scripts/eval_pretrained_inits.py --no-wandb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class InitCandidate:
    """One candidate starting checkpoint for the bake-off."""

    name: str
    init_arg: str            # value to pass to train_phase2_baseline.TrainConfig.init
    description: str
    a_priori_rank: int       # subjective expected ranking (1 = best); for results table commentary
    expected_min_mAP: float  # below this at end of screening = pipeline bug, not init weakness


CANDIDATES: list[InitCandidate] = [
    InitCandidate(
        name="yolo11m_coco",
        init_arg="yolo11m.pt",
        description="YOLOv11m + MS-COCO weights (Phase 2 vanilla anchor)",
        a_priori_rank=4,
        expected_min_mAP=0.10,
    ),
    InitCandidate(
        name="bioclip2_yolo11m",
        init_arg="bioclip2",
        description="YOLOv11m wrapped with BioCLIP2 ViT-L/14 adapter (D.4)",
        a_priori_rank=3,
        expected_min_mAP=0.10,
    ),
    InitCandidate(
        name="megalodon_yolov8x",
        init_arg="weights/marine_models/megalodon/best.pt",
        description="MBARI Megalodon YOLOv8x (FathomNet, single 'object' class)",
        a_priori_rank=2,
        expected_min_mAP=0.12,
    ),
    InitCandidate(
        name="mbari_315k_yolov8",
        init_arg="weights/marine_models/mbari_315k/best.pt",
        description="MBARI 315k YOLOv8 (deep-sea benthic, multi-class) -- STRONGEST PRIOR",
        a_priori_rank=1,
        expected_min_mAP=0.15,
    ),
    InitCandidate(
        name="megafishdetector_yolov5m",
        init_arg=("weights/marine_models/megafishdetector/"
                  "megafishdetector_v0_yolov5m_1280p.pt"),
        description="Megafishdetector v0 YOLOv5m 1280p (multi-source fish)",
        a_priori_rank=5,
        expected_min_mAP=0.08,
    ),
]


@dataclass
class BakeOffArgs:
    epochs: int = 10
    fold: int = 0
    imgsz: int = 640
    batch: int = 16
    device: str = ""
    project: str = "weights/runs/p2_bakeoff"
    workers: int = 8
    seed: int = 0
    skip: list[str] = field(default_factory=list)
    only: list[str] = field(default_factory=list)
    no_wandb: bool = False
    out_md: Path = field(default=REPO_ROOT / "notes" / "phase2_init_bakeoff.md")
    fold_yaml_dir: Path = field(default=REPO_ROOT / "data" / "folds")
    dataset_path_override: Optional[Path] = None


def parse_args() -> BakeOffArgs:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=10,
                   help="Epochs per candidate (default 10 for screening sprint)")
    p.add_argument("--fold", type=int, default=0,
                   help="Which CV fold to use (default 0)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="")
    p.add_argument("--project", default="weights/runs/p2_bakeoff")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip", nargs="+", default=[],
                   choices=[c.name for c in CANDIDATES],
                   help="Skip these candidates")
    p.add_argument("--only", nargs="+", default=[],
                   choices=[c.name for c in CANDIDATES],
                   help="Only run these candidates")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--out-md", type=Path,
                   default=REPO_ROOT / "notes" / "phase2_init_bakeoff.md")
    p.add_argument("--fold-yaml-dir", type=Path,
                   default=REPO_ROOT / "data" / "folds")
    p.add_argument("--dataset-path-override", type=Path, default=None)
    a = p.parse_args()
    return BakeOffArgs(**vars(a))


def _is_runnable(c: InitCandidate, args: BakeOffArgs) -> tuple[bool, str]:
    """Check whether a candidate has its init artifact available."""
    if c.name in args.skip:
        return False, "user --skip"
    if args.only and c.name not in args.only:
        return False, "not in --only"

    if c.init_arg == "bioclip2":
        weights_dir = REPO_ROOT / "weights" / "bioclip2"
        if not weights_dir.exists() or not any(weights_dir.glob("*.bin")) and not any(weights_dir.glob("*.safetensors")):
            return False, "bioclip2 weights missing -- run snapshot_download (D.1)"
        return True, ""

    if "/" not in c.init_arg and "\\" not in c.init_arg:
        return True, ""

    p = (REPO_ROOT / c.init_arg).resolve()
    if not p.exists():
        actual_pts = list(p.parent.glob("*.pt")) if p.parent.exists() else []
        if actual_pts:
            return True, f"using {actual_pts[0].name} instead of {p.name}"
        return False, f"weights file missing: {p}. Run scripts/download_marine_models.py"
    return True, ""


def _resolve_init_arg(c: InitCandidate) -> str:
    """Return the actual --init string to pass to train_phase2_baseline.

    For path-based candidates, swap to whatever .pt actually exists in the
    expected directory (the marine-model downloaders may not write to the
    exact same filename across versions).
    """
    if c.init_arg == "bioclip2" or ("/" not in c.init_arg and "\\" not in c.init_arg):
        return c.init_arg
    p = (REPO_ROOT / c.init_arg).resolve()
    if p.exists():
        return str(p)
    actual_pts = list(p.parent.glob("*.pt")) if p.parent.exists() else []
    if actual_pts:
        return str(actual_pts[0])
    return str(p)


def run_one(c: InitCandidate, args: BakeOffArgs) -> dict:
    """Run a single candidate via train_phase2_baseline.train_one_run()."""
    from scripts.train_phase2_baseline import TrainConfig, train_one_run

    init_arg = _resolve_init_arg(c)
    run_name = f"bakeoff_{c.name}_e{args.epochs}_fold{args.fold}"
    cfg = TrainConfig(
        init=init_arg,
        epochs=args.epochs,
        imgsz=args.imgsz,
        fold=args.fold,
        batch=args.batch,
        device=args.device,
        name=run_name,
        project=args.project,
        workers=args.workers,
        seed=args.seed,
        no_wandb=args.no_wandb,
        extra_tags=["bakeoff"],
        fold_yaml_dir=args.fold_yaml_dir,
        dataset_path_override=args.dataset_path_override,
    )
    return train_one_run(cfg)


def _extract_map5095(results: dict) -> Optional[float]:
    """Look up val mAP@[.50:.95] from Ultralytics' final_metrics dict.

    Ultralytics has reported this metric under several keys across versions.
    We try them in priority order and return whichever exists.
    """
    final = results.get("final_metrics", {})
    candidates = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95",
        "val/mAP50-95(B)",
        "val/mAP50-95",
    ]
    for k in candidates:
        if k in final:
            return float(final[k])
    return None


def _extract_map50(results: dict) -> Optional[float]:
    final = results.get("final_metrics", {})
    candidates = ["metrics/mAP50(B)", "metrics/mAP50", "val/mAP50(B)", "val/mAP50"]
    for k in candidates:
        if k in final:
            return float(final[k])
    return None


def write_results_md(args: BakeOffArgs,
                     ran: list[tuple[InitCandidate, dict, str]],
                     skipped: list[tuple[InitCandidate, str]]) -> None:
    """Render the ranked results to notes/phase2_init_bakeoff.md."""
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for c, results, status in ran:
        rows.append({
            "candidate": c,
            "results": results,
            "status": status,
            "map5095": _extract_map5095(results) or 0.0,
            "map50": _extract_map50(results) or 0.0,
        })
    rows.sort(key=lambda r: r["map5095"], reverse=True)

    primary = rows[0] if rows else None
    secondary = rows[1] if len(rows) > 1 else None
    third = rows[2] if len(rows) > 2 else None

    decision_lines = []
    if primary is None:
        decision_lines.append("**No runs completed** -- nothing to decide. Re-run after fixing failures.")
    elif primary["map5095"] < 0.15:
        decision_lines.append(
            f"**STOP** -- best init only reached val mAP@[.50:.95] = {primary['map5095']:.4f} at "
            f"{args.epochs} epochs, below the 0.15 sanity threshold. Likely pipeline bug "
            f"(cat_id mapping FLAG 1, label format, or fold YAML). DO NOT proceed to Phase 3."
        )
    else:
        decision_lines.append(f"**PRIMARY init for Phase 3-7:** `{primary['candidate'].name}` "
                              f"(val mAP@[.50:.95] = {primary['map5095']:.4f})")
        if secondary is not None:
            gap = primary["map5095"] - secondary["map5095"]
            if gap < 0.02:
                decision_lines.append(
                    f"**SECONDARY init for Phase 6 ensemble:** `{secondary['candidate'].name}` "
                    f"(val mAP = {secondary['map5095']:.4f}, gap {gap:+.4f})"
                )
                if third is not None:
                    decision_lines.append(
                        f"  Gap < 0.02 -- ALSO keep `{third['candidate'].name}` "
                        f"(val mAP = {third['map5095']:.4f}) as a 3rd ensemble member."
                    )
            elif gap > 0.05:
                decision_lines.append(
                    f"**SECONDARY init: SKIPPED.** Gap = {gap:+.4f} > 0.05 -- "
                    f"runner-up `{secondary['candidate'].name}` "
                    f"(val mAP = {secondary['map5095']:.4f}) is too far behind to "
                    f"contribute meaningful ensemble diversity. Use only PRIMARY."
                )
            else:
                decision_lines.append(
                    f"**SECONDARY init for Phase 6 ensemble:** `{secondary['candidate'].name}` "
                    f"(val mAP = {secondary['map5095']:.4f}, gap {gap:+.4f})"
                )

    lines = [
        "# Phase 2 EXP 2.3 -- 5-way multi-pretrained-init bake-off",
        "",
        f"Generated by `scripts/eval_pretrained_inits.py` on "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        f"**Settings:** epochs={args.epochs}, fold={args.fold}, imgsz={args.imgsz}, "
        f"batch={args.batch}, device={args.device or 'auto'}",
        "",
        "## Ranked results (by val mAP@[.50:.95])",
        "",
        "| Rank | Candidate | val mAP@[.50:.95] | val mAP@.50 | Wall time | A priori rank | Description |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, start=1):
        c = r["candidate"]
        wall = r["results"].get("wall_time_sec", 0) / 60
        lines.append(
            f"| {i} | `{c.name}` | **{r['map5095']:.4f}** | {r['map50']:.4f} | "
            f"{wall:.1f} min | {c.a_priori_rank} | {c.description} |"
        )

    if skipped:
        lines += ["", "## Skipped candidates", ""]
        for c, why in skipped:
            lines.append(f"- `{c.name}` -- {why}")

    lines += [
        "",
        "## Decision rules (from master plan v5.5 EXP 2.3)",
        "",
        *[f"- {line}" for line in decision_lines],
        "",
        "## Next steps",
        "",
        "1. Promote PRIMARY (and SECONDARY if applicable) to full 50-epoch training:",
        "   ```",
        "   python scripts/train_phase2_baseline.py --init <primary_init_arg> \\",
        "       --epochs 50 --fold 0 --imgsz 640 --name p2_full_<primary_name>",
        "   ```",
        "2. Submit the PRIMARY full-training output to Kaggle as the FIRST anchor",
        "   submission (per first-submission anchoring rule).",
        "3. Record LB scores in `notes/phase2_results.md`.",
        "4. Move to Phase 3 (Block E in checklist) using PRIMARY init weights as starting point.",
        "",
        "## Per-candidate raw output paths",
        "",
    ]
    for r in rows:
        c = r["candidate"]
        lines.append(f"- `{c.name}`:")
        lines.append(f"  - best.pt: `{r['results'].get('best_pt', '')}`")
        lines.append(f"  - results.csv: `{r['results'].get('results_csv', '')}`")
        lines.append(f"  - run summary JSON: `{Path(r['results'].get('save_dir', '')) / 'phase2_run_summary.json'}`")
    lines.append("")

    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[bakeoff] Results table written to: {args.out_md}")


def main() -> int:
    args = parse_args()

    print("=" * 72)
    print("Phase 2 EXP 2.3 -- 5-way multi-pretrained-init BAKE-OFF (v5.5)")
    print("=" * 72)
    print(f"Epochs/candidate: {args.epochs}")
    print(f"Fold:             {args.fold}")
    print(f"imgsz:            {args.imgsz}")
    print(f"Batch:            {args.batch}")
    print(f"Device:           {args.device or 'auto'}")
    print(f"Project dir:      {args.project}")
    print(f"W&B:              {'OFF' if args.no_wandb else 'ON'}")
    if args.skip:
        print(f"Skip:             {args.skip}")
    if args.only:
        print(f"Only:             {args.only}")
    print()

    runnable: list[InitCandidate] = []
    skipped: list[tuple[InitCandidate, str]] = []
    for c in CANDIDATES:
        ok, why = _is_runnable(c, args)
        if not ok:
            print(f"  [SKIP] {c.name:<28} -- {why}")
            skipped.append((c, why))
        else:
            tag = f"  [QUEUE] {c.name:<28}"
            if why:
                tag += f"  ({why})"
            print(tag)
            runnable.append(c)

    if not runnable:
        print("\nNo runnable candidates. Exiting.")
        return 1

    print(f"\nQueued {len(runnable)} candidates, skipped {len(skipped)}.\n")

    ran: list[tuple[InitCandidate, dict, str]] = []
    for i, c in enumerate(runnable, start=1):
        print()
        print("#" * 72)
        print(f"# CANDIDATE {i}/{len(runnable)}: {c.name}")
        print(f"# {c.description}")
        print("#" * 72)
        try:
            results = run_one(c, args)
            ran.append((c, results, "OK"))
        except Exception as exc:
            print(f"\n[FAIL] {c.name}: {exc.__class__.__name__}: {exc}")
            ran.append((c, {"final_metrics": {}, "wall_time_sec": 0.0}, f"FAIL: {exc}"))

    write_results_md(args, ran, skipped)

    summary_json = Path(args.project).resolve() / "bakeoff_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w") as f:
        json.dump({
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in args.__dict__.items()},
            "ran": [
                {
                    "name": c.name,
                    "status": status,
                    "map5095": _extract_map5095(results),
                    "map50": _extract_map50(results),
                    "wall_time_sec": results.get("wall_time_sec", 0),
                    "best_pt": results.get("best_pt"),
                }
                for c, results, status in ran
            ],
            "skipped": [{"name": c.name, "reason": why} for c, why in skipped],
        }, f, indent=2, default=str)
    print(f"[bakeoff] Machine-readable summary written to: {summary_json}")

    fails = [(c, _, s) for c, _, s in ran if s != "OK"]
    if fails:
        print(f"\n[bakeoff] {len(fails)} run(s) failed. Inspect above and re-run with --only to retry.")
        return 1
    print("\n[bakeoff] All candidates ran. Read notes/phase2_init_bakeoff.md for the ranking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
