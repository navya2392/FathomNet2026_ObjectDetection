#!/usr/bin/env python3
"""Train YOLOv8x with Domain-Adversarial Neural Network (DANN) head.

E8 in the v7 master plan: explicit feature-level domain adaptation between
labeled SOI/NOAA training images (source domain, label=0) and unlabeled
MBARI test images (target domain, label=1).

Architecture
------------
Custom training loop that wraps an Ultralytics YOLO model and adds:

  * A forward hook on `model.model[backbone_layer]` (SPPF=9 for YOLOv8x)
    that captures the feature map (B, 640, H/32, W/32).
  * A `DANNHead` (GRL + small CNN classifier).
  * A target dataloader over unlabeled MBARI test images.
  * Per-step: source detection loss + λ × domain loss.
  * λ ramped via the canonical sigmoid schedule λ(p)=2/(1+e^(-10p))-1.

We bypass Ultralytics' DetectionTrainer because mixing labeled+unlabeled
batches doesn't fit cleanly into its callback system and trainer monkey-
patching is brittle. We use Ultralytics' `v8DetectionLoss` directly so the
detection loss is bit-identical to the anchor recipe; we lose mosaic/mixup
augmentation (replaced by light flip+hsv) but for a 25-epoch fine-tune
from the anchor this is acceptable.

Total loss per batch:
    L = L_det(source) + λ * (CE(d_src, 0) + CE(d_tgt, 1)) / 2

At inference time the DANN head is detached: best.pt contains only the
backbone+detector and is fully compatible with the standard inference
scripts (predict_phase2_baseline.py, predict_multiscale_tta.py).

Usage
-----
    # Smoke test (2 epochs, batch=2, no target augmentation)
    python scripts/train_dann.py --init <anchor.pt> --smoke

    # Full run
    python scripts/train_dann.py \
        --init runs/detect/weights/runs/p2_full_mbari_315k_aug_e50_imgsz1024_fold0/weights/best.pt \
        --epochs 25 --imgsz 1024 --batch 8 --target-batch 4 \
        --max-lambda 0.1 --name e8_dann_fold0

Hyperparameter notes
--------------------
* `--max-lambda 0.1` is conservative for detection (lit: 0.05-0.2).
* `--epochs 25` warm-starts from the anchor (already 50ep of detection).
* `--target-batch 4` halves source batch (8) for VRAM headroom.
* `--backbone-layer 9` = SPPF output of YOLOv8x.

References
----------
* Ganin & Lempitsky 2015 (https://arxiv.org/abs/1409.7495)
* Saito et al. 2019 (https://arxiv.org/abs/1812.04798)
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402

from src.dann import (  # noqa: E402
    DANNHead,
    LambdaSchedule,
    install_feature_hook,
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _letterbox_np(img: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio, pad to imgsz×imgsz with grey border."""
    import cv2

    h0, w0 = img.shape[:2]
    s = imgsz / max(h0, w0)
    if s != 1:
        img = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    top, left = (imgsz - h) // 2, (imgsz - w) // 2
    bottom, right = imgsz - h - top, imgsz - w - left
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return img, s, (top, left)


class LabeledYoloDataset(Dataset):
    """Loads YOLO-format (image + .txt label) pairs from a fold's train list.

    Light augmentation: hflip (p=0.5), HSV jitter (p=1.0). No mosaic/mixup —
    we're warm-starting from the anchor so heavy aug is less critical.

    Returns dict matching what `v8DetectionLoss` expects:
        {
            "img": (B, 3, H, W) float32 in [0, 1],
            "cls": (N_total, 1) class indices (long),
            "bboxes": (N_total, 4) xywh-normalized,
            "batch_idx": (N_total,) which image each box belongs to,
        }
    """

    def __init__(
        self,
        image_paths: list[Path],
        imgsz: int = 1024,
        augment: bool = True,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        fliplr: float = 0.5,
    ):
        import cv2  # noqa
        self.cv2 = cv2
        self.image_paths = image_paths
        self.imgsz = imgsz
        self.augment = augment
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.fliplr = fliplr

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def _label_path_for(image_path: Path) -> Path:
        # YOLO convention: replace /images/ -> /labels/ and ext -> .txt
        s = str(image_path)
        if "/images/" in s:
            s = s.replace("/images/", "/labels/")
        elif "\\images\\" in s:
            s = s.replace("\\images\\", "\\labels\\")
        return Path(s).with_suffix(".txt")

    def _read_labels(self, label_path: Path) -> np.ndarray:
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        with label_path.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                rows.append([float(p) for p in parts[:5]])
        return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)

    def _augment_hsv(self, img: np.ndarray) -> np.ndarray:
        if not self.augment or (self.hsv_h == 0 and self.hsv_s == 0 and self.hsv_v == 0):
            return img
        r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1
        hue, sat, val = self.cv2.split(self.cv2.cvtColor(img, self.cv2.COLOR_BGR2HSV))
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(img.dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(img.dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(img.dtype)
        out = self.cv2.merge((self.cv2.LUT(hue, lut_hue), self.cv2.LUT(sat, lut_sat), self.cv2.LUT(val, lut_val)))
        return self.cv2.cvtColor(out, self.cv2.COLOR_HSV2BGR)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        img = self.cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read {img_path}")

        labels = self._read_labels(self._label_path_for(img_path))  # (N, 5) cls,xc,yc,w,h
        if self.augment:
            img = self._augment_hsv(img)

        h0, w0 = img.shape[:2]
        img_lb, s, (pad_top, pad_left) = _letterbox_np(img, self.imgsz)
        # Adjust labels into letterboxed coords (still normalized to [0,1] of the new canvas)
        if len(labels) > 0:
            # original norm xc,yc,w,h -> pixel
            labels_xyxy = labels.copy()
            labels_xyxy[:, 1] = labels[:, 1] * w0 * s + pad_left
            labels_xyxy[:, 2] = labels[:, 2] * h0 * s + pad_top
            labels_xyxy[:, 3] = labels[:, 3] * w0 * s
            labels_xyxy[:, 4] = labels[:, 4] * h0 * s
            # Renormalize against new canvas
            labels_xyxy[:, 1] /= self.imgsz
            labels_xyxy[:, 2] /= self.imgsz
            labels_xyxy[:, 3] /= self.imgsz
            labels_xyxy[:, 4] /= self.imgsz
            labels = labels_xyxy

        if self.augment and random.random() < self.fliplr:
            img_lb = img_lb[:, ::-1, :].copy()
            if len(labels) > 0:
                labels[:, 1] = 1.0 - labels[:, 1]

        # BGR -> RGB, HWC -> CHW, [0,1]
        img_lb = self.cv2.cvtColor(img_lb, self.cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(np.ascontiguousarray(img_lb)).permute(2, 0, 1).float() / 255.0

        return {
            "img": img_t,
            "cls": torch.from_numpy(labels[:, 0:1]) if len(labels) else torch.zeros((0, 1)),
            "bboxes": torch.from_numpy(labels[:, 1:5]) if len(labels) else torch.zeros((0, 4)),
        }


def labeled_collate(batch: list[dict]) -> dict:
    """Collate that matches v8DetectionLoss's expected batch format.

    Note: batch_idx is float32 (not long). Ultralytics' default YOLODataset
    builds batch_idx as float32 because v8DetectionLoss.preprocess() does
    `torch.cat([batch_idx.view(-1,1), cls.view(-1,1), bboxes], 1)`, which
    requires homogeneous dtypes. Using long here either errors or up-casts
    silently — we match the framework's convention.
    """
    imgs = torch.stack([b["img"] for b in batch], dim=0)
    cls = torch.cat([b["cls"] for b in batch], dim=0)
    bboxes = torch.cat([b["bboxes"] for b in batch], dim=0)
    batch_idx = torch.cat([
        torch.full((b["cls"].size(0),), float(i), dtype=torch.float32)
        for i, b in enumerate(batch)
    ], dim=0)
    return {"img": imgs, "cls": cls, "bboxes": bboxes, "batch_idx": batch_idx}


class UnlabeledImageDataset(Dataset):
    """Loads test images, applies letterbox, returns float tensor in [0,1]."""

    def __init__(self, image_dir: Path, imgsz: int = 1024, exts=(".jpg", ".jpeg", ".png")):
        import cv2
        self.cv2 = cv2
        self.imgsz = imgsz
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Target image dir not found: {image_dir}")
        self.paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
        if not self.paths:
            raise FileNotFoundError(f"No images in {image_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.cv2.imread(str(self.paths[idx]))
        if img is None:
            raise RuntimeError(f"Failed to read {self.paths[idx]}")
        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        img, _, _ = _letterbox_np(img, self.imgsz)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0


def cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_train_images(fold_yaml: Path) -> list[Path]:
    """Parse fold YAML and return all training image paths."""
    import yaml
    cfg = yaml.safe_load(fold_yaml.read_text())
    base = Path(cfg.get("path", fold_yaml.parent))
    train_field = cfg.get("train")
    if train_field is None:
        raise KeyError(f"Fold YAML {fold_yaml} missing 'train' key")

    # train field can be: a path to a list file, a list of paths, or a directory
    train_paths: list[Path] = []
    if isinstance(train_field, list):
        for p in train_field:
            train_paths.extend(_resolve_path_or_listfile(base, p))
    else:
        train_paths.extend(_resolve_path_or_listfile(base, train_field))
    return [p for p in train_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png")]


def _resolve_path_or_listfile(base: Path, p: str) -> list[Path]:
    pp = Path(p)
    if not pp.is_absolute():
        pp = (base / pp).resolve()
    if pp.is_file() and pp.suffix == ".txt":
        # YOLO list-file format: one image path per line
        out: list[Path] = []
        for line in pp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            q = Path(line)
            if not q.is_absolute():
                q = (base / q).resolve()
            out.append(q)
        return out
    if pp.is_dir():
        return list(pp.rglob("*.[jJpP]*[gG]"))
    if pp.exists():
        return [pp]
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--init", required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--target-batch", type=int, default=4)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--target-dir", default=None)
    p.add_argument("--name", default="e8_dann_fold0")
    p.add_argument("--project", default="runs/detect/weights/runs")
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--max-lambda", type=float, default=0.1)
    p.add_argument("--lambda-gamma", type=float, default=10.0)
    p.add_argument("--lambda-warmup-epochs", type=int, default=3)
    p.add_argument("--backbone-layer", type=int, default=9)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--save-period", type=int, default=5)
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs, batch=2, 50 source images for sanity check")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.smoke:
        args.epochs = 2
        args.batch = 2
        args.target_batch = 2
        args.workers = 0

    target_dir = args.target_dir or str(REPO_ROOT / "data" / "raw" / "images" / "test")
    fold_yaml = REPO_ROOT / "data" / "folds" / f"fold{args.fold}.yaml"
    if not fold_yaml.exists():
        print(f"[fatal] Fold YAML missing: {fold_yaml}", file=sys.stderr)
        return 1

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"E8 DANN training — {args.name}")
    print(f"  Init:           {args.init}")
    print(f"  Source fold:    {fold_yaml}")
    print(f"  Target dir:     {target_dir}")
    print(f"  Epochs:         {args.epochs}    imgsz: {args.imgsz}")
    print(f"  Source batch:   {args.batch}    Target batch: {args.target_batch}")
    print(f"  λ schedule:     gamma={args.lambda_gamma}, max={args.max_lambda}, "
          f"warmup={args.lambda_warmup_epochs} ep")
    print(f"  Hook layer:     model.{args.backbone_layer}")
    print(f"  Save dir:       {save_dir}")
    print("=" * 78)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. Load YOLO model ----
    print("\n[1/6] Loading YOLO model ...")
    yolo = YOLO(args.init)
    inner = yolo.model.to(device).train()
    if not getattr(inner, "criterion", None):
        inner.criterion = inner.init_criterion()
    print(f"  Inner detection model params: {sum(p.numel() for p in inner.parameters())/1e6:.2f}M")

    # ---- 2. Install backbone hook + DANN head ----
    print("\n[2/6] Installing backbone hook + DANN head ...")
    feat_hook = install_feature_hook(inner, layer_idx=args.backbone_layer)
    with torch.no_grad():
        probe = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
        _ = inner(probe)
    if feat_hook.in_channels == 0:
        feat_hook.remove()
        raise RuntimeError(
            f"Probe failed — no 4D feature at layer {args.backbone_layer}"
        )
    print(f"  Hooked layer.{args.backbone_layer}: in_channels={feat_hook.in_channels}")

    dann_head = DANNHead(in_channels=feat_hook.in_channels).to(device)
    print(f"  DANN head params: {sum(p.numel() for p in dann_head.parameters())/1e6:.3f}M")
    lambda_sched = LambdaSchedule(gamma=args.lambda_gamma, max_lambda=args.max_lambda)

    # ---- 3. Datasets / dataloaders ----
    print("\n[3/6] Building datasets ...")
    image_paths = _read_train_images(fold_yaml)
    if args.smoke:
        image_paths = image_paths[:50]
    print(f"  Source: {len(image_paths)} labeled images from fold{args.fold}")
    src_dataset = LabeledYoloDataset(
        image_paths=image_paths, imgsz=args.imgsz, augment=True,
    )
    src_loader = DataLoader(
        src_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=labeled_collate,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    tgt_dataset = UnlabeledImageDataset(target_dir, imgsz=args.imgsz)
    if args.smoke:
        tgt_dataset.paths = tgt_dataset.paths[:20]
    tgt_loader = DataLoader(
        tgt_dataset,
        batch_size=args.target_batch,
        shuffle=True,
        num_workers=max(0, args.workers // 2),
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    tgt_iter = cycle(tgt_loader)
    print(f"  Target: {len(tgt_dataset)} unlabeled images")
    print(f"  src batches/epoch: {len(src_loader)}    tgt batches: {len(tgt_loader)}")

    # ---- 4. Optimizer + scheduler ----
    print("\n[4/6] Building optimizer + scheduler ...")
    params = list(inner.parameters()) + list(dann_head.parameters())
    optimizer = torch.optim.SGD(
        params, lr=args.lr0, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True,
    )
    n_src_batches = len(src_loader)
    total_steps = n_src_batches * args.epochs
    warmup_steps = int(round(args.warmup_epochs * n_src_batches))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lrf + (1.0 - args.lrf) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = GradScaler(enabled=device.type == "cuda")

    # ---- 5. Training loop ----
    print("\n[5/6] Training ...\n")
    print(f"{'epoch':>5} {'lambda':>7} {'lr':>9} {'L_det':>7} {'L_dom':>7} {'acc_dom':>8} {'sec/ep':>8}")

    csv_path = save_dir / "results.csv"
    csv_path.write_text("epoch,lambda,lr,L_det,L_dom,acc_dom,sec_per_epoch\n")

    best_loss = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.lambda_warmup_epochs:
            lam = 0.0
        else:
            p = (epoch - args.lambda_warmup_epochs) / max(1, args.epochs - args.lambda_warmup_epochs)
            lam = lambda_sched(p)
        dann_head.set_lambda(lam)

        ep_t0 = time.time()
        ep_det = ep_dom = ep_acc = 0.0
        n = 0

        inner.train()
        dann_head.train()

        for s_batch in src_loader:
            global_step += 1
            optimizer.zero_grad(set_to_none=True)

            s_batch_dev = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                           for k, v in s_batch.items()}

            # Memory note: we split into two forward+backward pairs rather than
            # holding both forward graphs in memory simultaneously. Gradients
            # accumulate into the same params, so optimizer.step() sees the sum.
            # Mathematically equivalent to one combined backward but ~half the
            # peak VRAM (critical at imgsz=1024 b=8 on a 32 GB GPU).

            # ---- Source pass: detection loss + domain loss(label=0) ----
            with autocast(enabled=device.type == "cuda"):
                s_loss, _ = inner(s_batch_dev)
                # v9.1 fix: Ultralytics 8.4.39 v8DetectionLoss.loss returns
                # a (3,)-shape tensor (box, cls, dfl); the standard Trainer
                # sums it before .backward(). We bypass the Trainer, so sum
                # explicitly. Safe on scalar (.sum() of 0-d returns 0-d).
                if s_loss.ndim > 0:
                    s_loss = s_loss.sum()
                s_feat = feat_hook.feature
                s_dom_logits = dann_head(s_feat)
                s_dom_target = torch.zeros(
                    s_dom_logits.size(0), dtype=torch.long, device=device,
                )
                s_dom_loss = F.cross_entropy(s_dom_logits, s_dom_target)
                src_total = s_loss + lam * 0.5 * s_dom_loss
            scaler.scale(src_total).backward()

            # Snapshot acc-relevant tensors before the source graph is freed.
            with torch.no_grad():
                s_acc = (s_dom_logits.argmax(1) == s_dom_target).float().mean().item()
            s_loss_val = float(s_loss.detach().item())
            s_dom_loss_val = float(s_dom_loss.detach().item())

            # ---- Target pass: domain loss(label=1) only ----
            # During the lambda-warmup epochs (lam == 0) the target loss has no
            # effect on parameters; skip it entirely to save a forward pass.
            t_imgs = next(tgt_iter).to(device, non_blocking=True)
            if lam > 0.0:
                with autocast(enabled=device.type == "cuda"):
                    _ = inner(t_imgs)
                    t_feat = feat_hook.feature
                    t_dom_logits = dann_head(t_feat)
                    t_dom_target = torch.ones(
                        t_dom_logits.size(0), dtype=torch.long, device=device,
                    )
                    t_dom_loss = F.cross_entropy(t_dom_logits, t_dom_target)
                    tgt_total = lam * 0.5 * t_dom_loss
                scaler.scale(tgt_total).backward()
                with torch.no_grad():
                    t_acc = (t_dom_logits.argmax(1) == t_dom_target).float().mean().item()
                t_dom_loss_val = float(t_dom_loss.detach().item())
            else:
                t_acc = 0.0
                t_dom_loss_val = 0.0

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_det += s_loss_val
            ep_dom += 0.5 * (s_dom_loss_val + t_dom_loss_val)
            ep_acc += 0.5 * (s_acc + t_acc)
            n += 1

        ep_dt = time.time() - ep_t0
        if n > 0:
            ep_det /= n
            ep_dom /= n
            ep_acc /= n
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} {lam:7.4f} {cur_lr:9.6f} {ep_det:7.4f} {ep_dom:7.4f} {ep_acc:8.3f} {ep_dt:8.1f}")
        with csv_path.open("a") as f:
            f.write(f"{epoch},{lam:.6f},{cur_lr:.6f},{ep_det:.4f},{ep_dom:.4f},{ep_acc:.4f},{ep_dt:.1f}\n")

        last_pt = weights_dir / "last.pt"
        _save_yolo_checkpoint(inner, last_pt, feat_hook=feat_hook, epoch=epoch)
        if ep_det < best_loss:
            best_loss = ep_det
            best_pt = weights_dir / "best.pt"
            _save_yolo_checkpoint(inner, best_pt, feat_hook=feat_hook, epoch=epoch)
        if args.save_period > 0 and epoch % args.save_period == 0:
            ckpt = weights_dir / f"epoch{epoch:03d}.pt"
            _save_yolo_checkpoint(inner, ckpt, feat_hook=feat_hook, epoch=epoch)

    print(f"\n[6/6] Done.")
    print(f"  best.pt:    {weights_dir / 'best.pt'}")
    print(f"  last.pt:    {weights_dir / 'last.pt'}")
    print(f"  results.csv: {csv_path}")
    feat_hook.remove()
    return 0


def _save_yolo_checkpoint(inner: nn.Module, path: Path, feat_hook=None, epoch: int = 0) -> None:
    """Save a YOLO-compatible checkpoint with only the detection model.

    Strips the DANN feature hook so downstream inference (YOLO(path)) loads
    cleanly. Forward hooks reference live closures and break on deserialization.
    """
    import copy

    inner.eval()
    cpu_copy = copy.deepcopy(inner).cpu()
    for m in cpu_copy.modules():
        if hasattr(m, "_forward_hooks"):
            m._forward_hooks.clear()
        if hasattr(m, "_forward_pre_hooks"):
            m._forward_pre_hooks.clear()
        if hasattr(m, "_backward_hooks"):
            m._backward_hooks.clear()
    cpu_copy = cpu_copy.half()
    ckpt = {
        "epoch": epoch,
        "best_fitness": None,
        "model": cpu_copy,
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": {"e8_dann": True},
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    torch.save(ckpt, str(path))
    inner.train()


if __name__ == "__main__":
    sys.exit(main())
