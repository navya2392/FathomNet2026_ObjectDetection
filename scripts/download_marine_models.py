"""Download the three marine-specific pretrained detectors used in Phase 2.

Block reference: D.7, D.8, D.9, D.10 in master_checklist.txt (v5.5 expansion).

Why this matters
----------------
Per the host's clarification of competition Rule 6 (April 30 Discussion-board
reply by Laura Chrobak), publicly-available marine-life detection models are
allowed for initialization with disclosure. We use these as candidates 3, 4,
and 5 in the Phase 2 EXP 2.3 5-way multi-pretrained-init bake-off.

Why marine-specific weights matter for FathomNet 2026
-----------------------------------------------------
COCO-pretrained YOLO weights were trained on 80 classes of land/web imagery
(people, cars, kitchen objects, etc.). FathomNet test imagery is deep-sea
benthic photography from MBARI's submersibles. The visual statistics of these
two distributions barely overlap: lighting, color, contrast, texture, object
shapes, and even image aspect ratios are different. Standard transfer-learning
theory says: closer source domain -> better target performance. Megalodon and
MBARI-315k were literally pretrained on the SAME data distribution as the
competition test set, so their weights live closer to the optimum we're
trying to reach.

Models pulled
-------------
1. FathomNet/Megalodon (HF: FathomNet/megalodon)
   - Architecture: Ultralytics YOLOv8x
   - Pretraining: ALL publicly-available FathomNet localizations (single
     "object" class -- generic salient-marine-object detector)
   - File: weights/marine_models/megalodon/<best.pt or similar>
   - Size: ~270 MB

2. FathomNet/MBARI-315k-yolov8 (HF: FathomNet/MBARI-315k-yolov8)
   - Architecture: Ultralytics YOLOv8 (m or l size)
   - Pretraining: 315,000 MBARI deep-sea benthic images, multi-class
     taxonomic detection
   - File: weights/marine_models/mbari_315k/<best.pt>
   - Size: ~140 MB
   - STRONGEST a priori candidate (multi-class + same data distribution)

3. Megafishdetector v0 (GitHub: warplab/megafishdetector)
   - Architecture: Ultralytics YOLOv5m
   - Pretraining: AIMs Ozfish, FathomNet subset, VIAME FishTrack, NOAA
     Puget Sound Nearshore Fish, DeepFish, NOAA Labelled Fishes in the Wild
   - File: weights/marine_models/megafishdetector/<best.pt>
   - Size: ~40 MB

Idempotent: skips files that already exist on disk.

CRITICAL DISCLOSURE GATE (B.0.5 in master_checklist.txt)
--------------------------------------------------------
DO NOT submit a Kaggle entry that uses any of these weights until the
external-models declaration post on the Kaggle Discussion board is EDITED
to include them. The current draft of that edit lives in
notes/external_models_post.md.

Usage
-----
    # On the RunPod pod (typical):
    python scripts/download_marine_models.py

    # Skip a model (e.g. if megafishdetector URL is broken):
    python scripts/download_marine_models.py --skip megafishdetector

    # Force re-download (overwrite existing):
    python scripts/download_marine_models.py --force

Authentication
--------------
HuggingFace gating: as of May 1 2026, none of these models require an HF
access token. If that changes, set HF_TOKEN env var (see A.5.5 in checklist).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights" / "marine_models"


@dataclass
class ModelSpec:
    """One marine-pretrained-detector to download."""

    name: str
    target_dir: Path
    download_fn: Callable[["ModelSpec", bool], Optional[Path]]
    description: str
    expected_min_size_mb: int

    def is_already_downloaded(self) -> bool:
        if not self.target_dir.exists():
            return False
        pt_files = list(self.target_dir.rglob("*.pt"))
        if not pt_files:
            return False
        total_mb = sum(p.stat().st_size for p in pt_files) / 1024 / 1024
        return total_mb >= self.expected_min_size_mb * 0.5


def _hf_snapshot(repo_id: str, target_dir: Path) -> Path:
    """Wrap huggingface_hub.snapshot_download with helpful error messages."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Run `pip install huggingface_hub`. "
            "It is in requirements.txt; if you're on a fresh RunPod pod that's "
            "the first thing to fix."
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"  -> snapshot_download('{repo_id}') -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    return target_dir


def download_megalodon(spec: ModelSpec, _force: bool) -> Optional[Path]:
    return _hf_snapshot("FathomNet/megalodon", spec.target_dir)


def download_mbari_315k(spec: ModelSpec, _force: bool) -> Optional[Path]:
    return _hf_snapshot("FathomNet/MBARI-315k-yolov8", spec.target_dir)


def download_megafishdetector(spec: ModelSpec, _force: bool) -> Optional[Path]:
    """Megafishdetector lives on GitHub releases, not HuggingFace.

    The most recent release tag is `v0.1` of warplab/megafishdetector
    which ships several model sizes. We grab the YOLOv5m at 1280p
    (recommended in the repo README for general detection).
    """
    spec.target_dir.mkdir(parents=True, exist_ok=True)
    candidate_urls = [
        "https://github.com/warplab/megafishdetector/releases/download/v0.1/megafishdetector_v0_yolov5m_1280p.pt",
    ]
    out_path = spec.target_dir / "megafishdetector_v0_yolov5m_1280p.pt"
    last_err: Optional[Exception] = None
    for url in candidate_urls:
        try:
            print(f"  -> wget {url}")
            print(f"     -> {out_path}")
            urllib.request.urlretrieve(url, out_path)  # noqa: S310 (trusted URL)
            return out_path
        except Exception as exc:
            last_err = exc
            print(f"     [warn] {exc.__class__.__name__}: {exc}")
            continue
    raise RuntimeError(
        f"All Megafishdetector download URLs failed. Last error: {last_err}. "
        "If GitHub release tarball is gone, check warplab/megafishdetector "
        "on GitHub for a newer release URL or HF mirror."
    )


SPECS: list[ModelSpec] = [
    ModelSpec(
        name="megalodon",
        target_dir=WEIGHTS_DIR / "megalodon",
        download_fn=download_megalodon,
        description="FathomNet Megalodon (YOLOv8x, single-class FathomNet detector, MBARI)",
        expected_min_size_mb=200,
    ),
    ModelSpec(
        name="mbari_315k",
        target_dir=WEIGHTS_DIR / "mbari_315k",
        download_fn=download_mbari_315k,
        description="MBARI 315k (YOLOv8, multi-class deep-sea benthic, MBARI/FathomNet)",
        expected_min_size_mb=80,
    ),
    ModelSpec(
        name="megafishdetector",
        target_dir=WEIGHTS_DIR / "megafishdetector",
        download_fn=download_megafishdetector,
        description="Megafishdetector v0 (YOLOv5m@1280p, generic fish detector, warplab)",
        expected_min_size_mb=20,
    ),
]


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=[s.name for s in SPECS],
        help="Skip these model names (still considered 'present' in summary if dir exists)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        choices=[s.name for s in SPECS],
        help="ONLY download these models (mutually exclusive with --skip in practice)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=WEIGHTS_DIR,
        help=f"Output dir for all marine model weights (default: {WEIGHTS_DIR.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FathomNet 2026 Phase 2 -- marine pretrained model downloader")
    print("Block: D.7-D.10 in master_checklist.txt (v5.5 expansion)")
    print("=" * 72)
    print()
    print(f"Output directory: {args.out}")
    print(f"Models requested:  {len(SPECS)}")
    if args.skip:
        print(f"Skipping:          {args.skip}")
    if args.only:
        print(f"ONLY downloading:  {args.only}")
    print(f"Force re-download: {args.force}")
    print()

    summary: list[tuple[str, str, str]] = []  # (name, status, info)

    for spec in SPECS:
        print("-" * 72)
        print(f"[{spec.name}] {spec.description}")
        print(f"  Target: {spec.target_dir.relative_to(REPO_ROOT)}")

        if spec.name in args.skip:
            print(f"  [SKIPPED] (requested)")
            summary.append((spec.name, "SKIPPED", "user --skip"))
            continue

        if args.only and spec.name not in args.only:
            print(f"  [SKIPPED] (not in --only list)")
            summary.append((spec.name, "SKIPPED", "not in --only"))
            continue

        if spec.is_already_downloaded() and not args.force:
            existing_pts = list(spec.target_dir.rglob("*.pt"))
            total = sum(p.stat().st_size for p in existing_pts)
            print(f"  [PRESENT] {len(existing_pts)} .pt file(s), total {_human_size(total)} "
                  f"(--force to overwrite)")
            summary.append((spec.name, "PRESENT", f"{len(existing_pts)} .pt, {_human_size(total)}"))
            continue

        if args.force and spec.target_dir.exists():
            print(f"  [FORCE] removing existing target dir")
            shutil.rmtree(spec.target_dir)

        try:
            spec.download_fn(spec, args.force)
            existing_pts = list(spec.target_dir.rglob("*.pt"))
            total = sum(p.stat().st_size for p in existing_pts)
            print(f"  [OK] {len(existing_pts)} .pt file(s), total {_human_size(total)}")
            summary.append((spec.name, "OK", f"{len(existing_pts)} .pt, {_human_size(total)}"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {exc.__class__.__name__}: {exc}")
            summary.append((spec.name, "FAIL", f"{exc.__class__.__name__}: {exc}"))
            continue

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"{'Name':<22} {'Status':<10} {'Info':<40}")
    print("-" * 72)
    for name, status, info in summary:
        print(f"{name:<22} {status:<10} {info:<40}")

    fails = [s for s in summary if s[1] == "FAIL"]
    if fails:
        print()
        print(f"!! {len(fails)} model(s) failed to download. Phase 2 EXP 2.3 will run "
              f"with whatever DID download (script `eval_pretrained_inits.py` skips "
              f"missing inits).")
        return 1

    print()
    print("All requested models present. Next step:")
    print("  python scripts/eval_pretrained_inits.py  # the 5-way multi-init bake-off")
    return 0


if __name__ == "__main__":
    sys.exit(main())
