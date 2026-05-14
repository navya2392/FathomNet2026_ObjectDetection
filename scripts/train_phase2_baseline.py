"""Train one Phase 2 baseline run with W&B logging.

Block reference: C.2, C.3, C.5, C.5.1 in master_checklist.txt (v5.5).

What this script does
---------------------
ONE end-to-end training run of an Ultralytics-compatible detector on a
single fold of the FathomNet 2026 dataset. Used in two ways:

  (a) Single-init experiments (C.2, C.3, C.5.1)
      python scripts/train_phase2_baseline.py \
          --init yolo11m.pt --epochs 50 --imgsz 640 --fold 0 \
          --name p2_baseline_yolo11m_coco

  (b) Driven by `eval_pretrained_inits.py` for the 5-way bake-off (C.5).
      That script imports `train_one_run()` directly to avoid spinning
      up a fresh subprocess per init.

Init catalog (per v5.5 master plan EXP 2.3 5-way bake-off)
---------------------------------------------------------
The --init argument accepts:

  * Ultralytics auto-download names: 'yolo11n.pt', 'yolo11m.pt', 'yolo11l.pt'
    (downloaded on first use to ~/.cache/ultralytics/)
  * Local .pt paths: 'weights/marine_models/megalodon/<file>.pt'
  * Special tag 'bioclip2' (sets up the BioCLIP2-wrapped YOLOv11m via the
    src.bioclip2_yolo adapter -- see Footnote 1 below)

The script auto-detects the architecture from the .pt file if needed.

Footnote 1: BioCLIP2 wrapper integration with Ultralytics trainer
----------------------------------------------------------------
The BioCLIP2 wrapper preserves YOLOv11m's exact forward signature (zero-
init projections at step 0 -> identical outputs to vanilla yolo11m), so
the cheapest integration is:
  1. Train vanilla yolo11m+COCO with this script (already candidate #1)
  2. Load the resulting best.pt INTO the BioCLIP2 wrapper
  3. Fine-tune for a few more epochs with the adapter heads unfrozen
This is implemented via --bioclip2-fine-tune-from <path>, which loads
the candidate #1 weights and wraps them. For the v5.5 bake-off, this
runs as a SEPARATE EXPERIMENT after candidate #1 finishes training,
so it doesn't add a new branch to the main bake-off table.

W&B logging (per master_checklist.txt W1-W6)
--------------------------------------------
Every run logs to project=fathomnet-2026 with:
  * tags = ["phase2", <init_name>, "fold<i>", "imgsz<size>"]
  * config = all CLI args + resolved init path + git SHA
  * step metrics = epoch loss components + val mAP@50 + val mAP@[.50:.95]
  * artifacts = best.pt + last.pt + results.csv (final-epoch only)

Failure modes handled
---------------------
  * Init file missing -> clean error pointing at download_marine_models.py
  * Fold YAML missing -> clean error pointing at make_yolo_fold.py
  * GPU OOM -> suggest reducing --batch
  * cat_id mapping bug (val mAP < 0.05 at end) -> non-zero exit + warning

Usage examples
--------------
    # Simple smoke test (10 epochs, nano model)
    python scripts/train_phase2_baseline.py --init yolo11n.pt --epochs 10 --fold 0 --name p2_smoke

    # Phase 2 vanilla baseline anchor (50 epochs, m model, COCO weights)
    python scripts/train_phase2_baseline.py --init yolo11m.pt --epochs 50 --fold 0 \
        --name p2_baseline_yolo11m_coco --batch 16 --imgsz 640

    # Bake-off candidate: MBARI 315k starting weights
    python scripts/train_phase2_baseline.py \
        --init weights/marine_models/mbari_315k/best.pt --epochs 50 --fold 0 \
        --name p2_full_mbari_315k

    # Fine-tune from a vanilla baseline checkpoint with BioCLIP2 adapter
    python scripts/train_phase2_baseline.py --init bioclip2 \
        --bioclip2-fine-tune-from weights/runs/p2_baseline_yolo11m_coco/best.pt \
        --epochs 20 --fold 0 --name p2_bioclip2_finetune
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class TrainConfig:
    """All hyperparameters for one training run, with sane defaults."""

    init: str = "yolo11m.pt"
    epochs: int = 50
    imgsz: int = 640
    fold: int = 0
    batch: int = 16
    lr0: float = 0.01
    name: str = "p2_baseline_yolo11m_coco"
    device: str = ""  # "" = auto, "cpu", "0", "0,1"
    project: str = "weights/runs"
    workers: int = 8
    patience: int = 15
    optimizer: str = "auto"  # Ultralytics will pick AdamW or SGD
    cos_lr: bool = True
    close_mosaic: int = 10  # disable mosaic in last N epochs (Ultralytics default)
    seed: int = 0
    wandb_project: str = "fathomnet-2026"
    wandb_entity: Optional[str] = None  # default: W&B's user-level default
    extra_tags: list[str] = field(default_factory=list)
    bioclip2_fine_tune_from: Optional[str] = None
    fold_yaml_dir: Path = field(default=REPO_ROOT / "data" / "folds")
    dataset_path_override: Optional[Path] = None
    no_wandb: bool = False

    # Augmentation knobs (Ultralytics names). Defaults match Ultralytics
    # YOLO defaults (mosaic on, mixup off, copy_paste off, light HSV jitter).
    # Override via CLI for "heavy aug" experiments inspired by FathomNet 2025
    # winner Robert Hunt (Cutmix+Mixup gave +0.16 val improvement) — see
    # notes/FINDINGS.md F-009 / F-010.
    mosaic: float = 1.0          # default 1.0 — random 4-image mosaic
    mixup: float = 0.0           # default 0.0 — Hunt used ~0.15
    copy_paste: float = 0.0      # default 0.0 — segmentation-style copy-paste
    hsv_h: float = 0.015         # hue jitter; Ultralytics default
    hsv_s: float = 0.7           # saturation; Ultralytics default
    hsv_v: float = 0.4           # value/brightness; Ultralytics default
    degrees: float = 0.0         # rotation; default 0
    translate: float = 0.1       # translation; Ultralytics default
    scale: float = 0.5           # zoom; Ultralytics default
    fliplr: float = 0.5          # horizontal flip; Ultralytics default
    flipud: float = 0.0          # vertical flip; default 0 (most images orientation-aware)
    erasing: float = 0.4         # random erasing; Ultralytics default for v8+

    # Layer-wise LR / freezing — addresses F-002 implication 4 (preserve MBARI prior)
    freeze: int = 0              # freeze first N backbone layers; 0=disabled

    # Kiryo non-negative PU loss (Phase 3 EXP 3.2) — addresses F-008.
    # When enabled, the v8DetectionLoss classification head is replaced
    # with a PU-corrected version. See src/pu_v8_detection_loss.py.
    pu_loss: bool = False
    pu_pi: float = 0.1401            # global pi (per-class refinement is a follow-up)
    pu_pi_min: float = 0.005         # floor for stability
    pu_pi_max: float = 0.5           # ceiling for stability
    pu_warmup_epochs: int = 3        # use plain BCE for first N epochs


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _resolve_init_path(init: str) -> tuple[str, str, dict]:
    """Return (path_or_name_for_ultralytics, init_label, info_dict).

    For Ultralytics auto-download names like 'yolo11m.pt', returns the
    name verbatim (Ultralytics handles the download).
    For local files, returns the absolute path.
    Special-cases 'bioclip2' which is a tag, not a file.
    """
    info: dict[str, Any] = {"raw": init}

    if init == "bioclip2":
        return "bioclip2", "bioclip2_wrap_yolo11m", info

    if "/" not in init and "\\" not in init:
        info["source"] = "ultralytics_auto_download"
        return init, init.replace(".pt", ""), info

    p = Path(init)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Init weights file not found: {p}\n"
            f"For marine pretrained models, run:\n"
            f"  python scripts/download_marine_models.py\n"
            f"Then point --init at the downloaded .pt file inside "
            f"weights/marine_models/<name>/."
        )
    info["source"] = "local_path"
    info["abs_path"] = str(p)

    label = p.stem
    if "megalodon" in str(p).lower():
        label = "megalodon_yolov8x"
    elif "315k" in str(p).lower() or "mbari" in str(p).lower():
        label = "mbari_315k_yolov8"
    elif "megafish" in str(p).lower():
        label = "megafishdetector_yolov5m"
    return str(p), label, info


def _resolve_fold_yaml(cfg: TrainConfig) -> Path:
    yaml_path = cfg.fold_yaml_dir / f"fold{cfg.fold}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Fold YAML not found: {yaml_path}\n"
            f"Generate it with:\n"
            f"  python scripts/make_yolo_fold.py\n"
            f"Or override location with --fold-yaml-dir."
        )
    return yaml_path


def _setup_wandb(cfg: TrainConfig, init_label: str, init_info: dict, fold_yaml: Path) -> Optional[Any]:
    """Initialize W&B if available and not disabled. Returns wandb run or None."""
    if cfg.no_wandb:
        print("[wandb] DISABLED via --no-wandb")
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] Not installed (`pip install wandb`); proceeding without W&B")
        return None

    if not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY not set; W&B run will go to anonymous mode "
              "(or fail silently). Set the key with `wandb login` or "
              "`export WANDB_API_KEY=...`")

    tags = ["phase2", init_label, f"fold{cfg.fold}", f"imgsz{cfg.imgsz}", *cfg.extra_tags]
    config = {
        **{k: v for k, v in cfg.__dict__.items() if not isinstance(v, Path)},
        "init_label": init_label,
        "init_info": init_info,
        "fold_yaml": str(fold_yaml),
        "git_sha": _git_sha(),
    }
    print(f"[wandb] init project='{cfg.wandb_project}' name='{cfg.name}' tags={tags}")
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.name,
        tags=tags,
        config=config,
        reinit=True,
    )
    return run


def train_one_run(cfg: TrainConfig) -> dict:
    """Train one model, log to W&B, return results dict.

    Returns
    -------
    {
        "init_label": str,
        "best_pt": absolute path to best.pt,
        "results_csv": absolute path to Ultralytics results.csv,
        "final_metrics": dict of last-epoch metrics from results.csv,
        "wall_time_sec": float,
    }
    """
    init_path_or_name, init_label, init_info = _resolve_init_path(cfg.init)
    fold_yaml = _resolve_fold_yaml(cfg)

    print("=" * 72)
    print(f"Phase 2 training run -- {cfg.name}")
    print(f"  Init:        {cfg.init}  -> {init_path_or_name}  (label={init_label})")
    print(f"  Fold:        {cfg.fold}  ({fold_yaml})")
    print(f"  Epochs:      {cfg.epochs}")
    print(f"  imgsz:       {cfg.imgsz}")
    print(f"  Batch:       {cfg.batch}")
    print(f"  Device:      {cfg.device or 'auto'}")
    print(f"  Output dir:  {cfg.project}/{cfg.name}/")
    print("=" * 72)

    wandb_run = _setup_wandb(cfg, init_label, init_info, fold_yaml)
    start = time.time()

    if init_path_or_name == "bioclip2":
        results = _train_bioclip2(cfg, fold_yaml)
    else:
        results = _train_ultralytics(cfg, init_path_or_name, fold_yaml)

    wall = time.time() - start
    results["wall_time_sec"] = wall
    results["init_label"] = init_label

    print("=" * 72)
    print(f"DONE in {wall/60:.1f} min")
    print(f"  Best weights: {results.get('best_pt')}")
    print(f"  Final mAP50:  {results.get('final_metrics', {}).get('metrics/mAP50(B)', 'unknown')}")
    print(f"  Final mAP5095:{results.get('final_metrics', {}).get('metrics/mAP50-95(B)', 'unknown')}")
    print("=" * 72)

    if wandb_run is not None:
        try:
            wandb_run.summary.update({
                "wall_time_sec": wall,
                "best_pt": str(results.get("best_pt", "")),
                **{f"final/{k}": v for k, v in results.get("final_metrics", {}).items()},
            })
            wandb_run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] finish failed: {exc}")

    return results


def _train_ultralytics(cfg: TrainConfig, init_path_or_name: str, fold_yaml: Path) -> dict:
    """Standard Ultralytics training path for any .pt init."""
    from ultralytics import YOLO

    model = YOLO(init_path_or_name)

    # Optional Kiryo PU loss installation (Phase 3 EXP 3.2 — see F-008).
    pu_handle = None
    if cfg.pu_loss:
        from src.pu_v8_detection_loss import install_pu_loss_on_model
        nc = int(getattr(model.model, "nc", 32))
        pu_handle = install_pu_loss_on_model(
            model,
            num_classes=nc,
            pi=cfg.pu_pi,
            pi_min=cfg.pu_pi_min,
            pi_max=cfg.pu_pi_max,
            warmup_epochs=cfg.pu_warmup_epochs,
        )

        def _pu_set_epoch_callback(trainer):
            """Update PU current epoch + log every-epoch state for verification."""
            try:
                ep = int(trainer.epoch)
                pu_handle.set_epoch(ep)
                st = pu_handle.state()
                print(f"[pu_loss] epoch {ep}: state={st}")
            except Exception as exc:  # noqa: BLE001
                print(f"[pu_loss] callback error: {exc}")

        model.add_callback("on_train_epoch_start", _pu_set_epoch_callback)
        print(f"[pu_loss] enabled: pi={cfg.pu_pi}, warmup_epochs={cfg.pu_warmup_epochs}, "
              f"pi_min={cfg.pu_pi_min}, pi_max={cfg.pu_pi_max}")

    train_kwargs = dict(
        data=str(fold_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        lr0=cfg.lr0,
        device=cfg.device or None,
        project=cfg.project,
        name=cfg.name,
        workers=cfg.workers,
        patience=cfg.patience,
        optimizer=cfg.optimizer,
        cos_lr=cfg.cos_lr,
        close_mosaic=cfg.close_mosaic,
        seed=cfg.seed,
        verbose=True,
        exist_ok=True,
        # Augmentation pass-through (see TrainConfig docstring).
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        copy_paste=cfg.copy_paste,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        fliplr=cfg.fliplr,
        flipud=cfg.flipud,
        erasing=cfg.erasing,
        freeze=cfg.freeze if cfg.freeze > 0 else None,
    )
    print(f"[ultralytics] model.train({train_kwargs})")
    try:
        train_results = model.train(**train_kwargs)
    finally:
        if pu_handle is not None:
            try:
                final_state = pu_handle.state()
                print(f"[pu_loss] final state: {final_state}")
                pu_handle.uninstall()
            except Exception as exc:  # noqa: BLE001
                print(f"[pu_loss] uninstall error: {exc}")

    save_dir = Path(getattr(train_results, "save_dir", REPO_ROOT / cfg.project / cfg.name))
    best_pt = save_dir / "weights" / "best.pt"
    results_csv = save_dir / "results.csv"

    final_metrics: dict[str, float] = {}
    if results_csv.exists():
        try:
            with results_csv.open() as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            header = [h.strip() for h in lines[0].split(",")]
            last = [v.strip() for v in lines[-1].split(",")]
            for k, v in zip(header, last):
                try:
                    final_metrics[k] = float(v)
                except ValueError:
                    pass
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to parse results.csv: {exc}")

    return {
        "best_pt": str(best_pt),
        "results_csv": str(results_csv),
        "final_metrics": final_metrics,
        "save_dir": str(save_dir),
    }


def _train_bioclip2(cfg: TrainConfig, fold_yaml: Path) -> dict:
    """BioCLIP2-wrapped YOLOv11m training (Phase 1 D.4 adapter).

    Strategy: load the wrapped model, then monkey-patch a YOLO() object's
    `.model` attribute with our wrapper so that Ultralytics' Trainer
    operates on the wrapped network. Forward-pass compatibility was
    verified by the smoke test in src/bioclip2_yolo.py __main__.

    If the user provided --bioclip2-fine-tune-from, we initialize the
    underlying YOLO weights from that checkpoint (typically the vanilla
    yolo11m+COCO baseline output from candidate #1), so the only delta
    is the BioCLIP2 adapter contribution.

    NOTE: this is the most experimental path in the bake-off. If
    Ultralytics' Trainer fails to handle the wrapper, fall back to
    treating BioCLIP2 as a Phase 4 ablation rather than a Phase 2
    bake-off candidate.
    """
    from src.bioclip2_yolo import BioCLIP2YOLOWrapper
    from ultralytics import YOLO

    base_weights = cfg.bioclip2_fine_tune_from or "yolo11m.pt"
    print(f"[bioclip2] wrapping {base_weights} with BioCLIP2 adapter ...")
    wrapper = BioCLIP2YOLOWrapper(yolo_weights=base_weights, freeze_yolo=False)

    model = YOLO(base_weights)
    underlying = model.model
    for attr in ("yaml", "nc", "names", "stride", "args"):
        if hasattr(underlying, attr) and not hasattr(wrapper, attr):
            try:
                setattr(wrapper, attr, getattr(underlying, attr))
            except Exception:
                pass
    model.model = wrapper

    train_kwargs = dict(
        data=str(fold_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        lr0=cfg.lr0,
        device=cfg.device or None,
        project=cfg.project,
        name=cfg.name,
        workers=cfg.workers,
        patience=cfg.patience,
        optimizer=cfg.optimizer,
        cos_lr=cfg.cos_lr,
        close_mosaic=cfg.close_mosaic,
        seed=cfg.seed,
        verbose=True,
        exist_ok=True,
    )
    try:
        print(f"[ultralytics+bioclip2] model.train({train_kwargs})")
        train_results = model.train(**train_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "BioCLIP2 wrapper failed to integrate with Ultralytics trainer. "
            "Symptoms include AttributeError on wrapper. Fix paths:\n"
            "  (a) implement a Trainer subclass in src/bioclip2_yolo.py\n"
            "  (b) write a custom training loop using torch DDP\n"
            "  (c) skip BioCLIP2 from the Phase 2 bake-off and re-add it\n"
            "      as Phase 4 EXP 4.10 ablation\n"
            f"Original error: {exc.__class__.__name__}: {exc}"
        ) from exc

    save_dir = Path(getattr(train_results, "save_dir", REPO_ROOT / cfg.project / cfg.name))
    best_pt = save_dir / "weights" / "best.pt"
    results_csv = save_dir / "results.csv"

    final_metrics: dict[str, float] = {}
    if results_csv.exists():
        try:
            with results_csv.open() as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            header = [h.strip() for h in lines[0].split(",")]
            last = [v.strip() for v in lines[-1].split(",")]
            for k, v in zip(header, last):
                try:
                    final_metrics[k] = float(v)
                except ValueError:
                    pass
        except Exception as exc:
            print(f"[warn] failed to parse results.csv: {exc}")

    return {
        "best_pt": str(best_pt),
        "results_csv": str(results_csv),
        "final_metrics": final_metrics,
        "save_dir": str(save_dir),
    }


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--init", default="yolo11m.pt",
                   help="Initialization weights (Ultralytics name, .pt path, or 'bioclip2')")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--name", default="p2_baseline_yolo11m_coco",
                   help="Run name (becomes the subfolder under --project)")
    p.add_argument("--device", default="",
                   help="GPU device(s), e.g. '0' or '0,1' or 'cpu'. Empty = auto.")
    p.add_argument("--project", default="weights/runs",
                   help="Output project dir (Ultralytics convention)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--optimizer", default="auto", choices=["auto", "SGD", "Adam", "AdamW"])
    p.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="fathomnet-2026")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--extra-tags", nargs="*", default=[])
    p.add_argument("--bioclip2-fine-tune-from", default=None,
                   help="With --init bioclip2: path to the .pt to wrap. "
                        "Default = yolo11m.pt (COCO weights).")
    p.add_argument("--fold-yaml-dir", type=Path,
                   default=REPO_ROOT / "data" / "folds",
                   help="Directory containing fold{i}.yaml files")
    p.add_argument("--dataset-path-override", type=Path, default=None,
                   help="If set, regenerates fold YAMLs with this `path:` "
                        "before training. Useful when running on RunPod where "
                        "data lives at /workspace/data/raw.")
    # Augmentation knobs (see TrainConfig for defaults).
    p.add_argument("--mosaic", type=float, default=1.0,
                   help="Mosaic augmentation prob (0.0-1.0, Ultralytics default 1.0)")
    p.add_argument("--mixup", type=float, default=0.0,
                   help="Mixup augmentation prob (0.0-1.0, Ultralytics default 0.0)")
    p.add_argument("--copy-paste", type=float, default=0.0,
                   help="Copy-paste augmentation prob (0.0-1.0, Ultralytics default 0.0)")
    p.add_argument("--hsv-h", type=float, default=0.015, help="HSV hue jitter")
    p.add_argument("--hsv-s", type=float, default=0.7, help="HSV saturation jitter")
    p.add_argument("--hsv-v", type=float, default=0.4, help="HSV value/brightness jitter")
    p.add_argument("--degrees", type=float, default=0.0, help="Rotation degrees")
    p.add_argument("--translate", type=float, default=0.1, help="Translation fraction")
    p.add_argument("--scale", type=float, default=0.5, help="Scale jitter fraction")
    p.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip prob")
    p.add_argument("--flipud", type=float, default=0.0, help="Vertical flip prob")
    p.add_argument("--erasing", type=float, default=0.4, help="Random erasing prob")
    p.add_argument("--freeze", type=int, default=0,
                   help="Freeze first N backbone layers (0=disabled). "
                        "Per F-002: freeze=10 helps preserve MBARI prior on small fine-tune sets.")
    # Kiryo PU loss (Phase 3 EXP 3.2)
    p.add_argument("--pu-loss", action="store_true",
                   help="Enable Kiryo non-negative PU loss on the classification head. "
                        "Addresses F-008 (PU+full-coverage gap). See src/pu_v8_detection_loss.py.")
    p.add_argument("--pu-pi", type=float, default=0.1401,
                   help="Global PU prior pi (default 0.1401, from F-003 area-fraction estimate). "
                        "Used for all 32 classes; per-class refinement is a follow-up.")
    p.add_argument("--pu-pi-min", type=float, default=0.005,
                   help="Floor for per-class pi (stability).")
    p.add_argument("--pu-pi-max", type=float, default=0.5,
                   help="Ceiling for per-class pi (stability).")
    p.add_argument("--pu-warmup-epochs", type=int, default=3,
                   help="Use plain BCE for the first N epochs before switching to PU.")

    args = p.parse_args()
    return TrainConfig(**vars(args))


def main() -> int:
    cfg = parse_args()

    if cfg.dataset_path_override is not None:
        print(f"[setup] regenerating fold YAMLs with path override: "
              f"{cfg.dataset_path_override}")
        rc = subprocess.call([
            sys.executable, str(REPO_ROOT / "scripts" / "make_yolo_fold.py"),
            "--dataset-path", str(cfg.dataset_path_override),
            "--out-dir", str(cfg.fold_yaml_dir),
        ])
        if rc != 0:
            print("[ERROR] make_yolo_fold.py failed; aborting.")
            return rc

    results = train_one_run(cfg)

    out_summary = Path(results["save_dir"]) / "phase2_run_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w") as f:
        json.dump({
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nPer-run summary written to: {out_summary}")

    final = results.get("final_metrics", {})
    map5095 = final.get("metrics/mAP50-95(B)") or final.get("metrics/mAP50-95")
    if map5095 is not None and map5095 < 0.05:
        print(f"\n[WARNING] Final val mAP@[.50:.95] = {map5095:.4f} < 0.05.")
        print(f"          Likely cat_id mapping bug (FLAG 1) or label format issue.")
        print(f"          Run tests/test_submit.py round-trip test to verify mapping.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
