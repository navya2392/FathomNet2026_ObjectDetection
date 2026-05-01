"""Class-aware copy-paste augmentation for the Phase 4 EXP 4.4 experiment.

Block reference: F.5 in master_checklist.txt.

Reference
---------
Ghiasi, Golnaz et al. "Simple Copy-Paste is a Strong Data Augmentation
Method for Instance Segmentation." CVPR 2021.
https://arxiv.org/abs/2012.07177

Adapted for FathomNet 2026's extreme class imbalance per the v5.4
"per-class copy-paste schedule" decision rule (master plan, Phase 4
EXP 4.4 section).

What problem does this solve
----------------------------
With sea slug at 7 instances and isopod at <50, even RFS oversampling
(see src/rfs_sampler.py) only multiplies the SAME 7 sea slug images
into more batches. The model sees the same boxes over and over, which
helps generalization much less than seeing the same instances in
different visual contexts.

Class-aware copy-paste solves this by literally pasting GT crops of
rare classes onto OTHER images:

1. Pre-build an "instance bank" of GT crops from the training set,
   indexed by class.
2. At training time, for each image, with per-class probability p_c,
   sample one or more crops of class c and paste them into the image
   at random non-overlapping locations.
3. Add the corresponding bbox annotations to the image's label.

Per-class probability schedule (v5.4 codification)
--------------------------------------------------
| Tier      | Train instance count | Paste probability per image |
|-----------|----------------------|------------------------------|
| critical  | < 30                 | 0.7 - 0.9                    |
| rare      | 30 - 99              | 0.4 - 0.6                    |
| mid       | 100 - 999            | 0.2 - 0.3                    |
| common    | >= 1000              | 0.05 - 0.1                   |

The default schedule below uses the midpoint of each range.

When pasting fails
------------------
Pasting can be skipped per-image if:
- The class has zero crops in the bank (not yet seen any GT)
- N attempts to find a non-overlapping paste location all fail
  (image is too crowded with existing GT)
- The crop is larger than the target image (degenerate case)

These failures are silent by default; pass `verbose=True` to log them.

Status (May 1, 2026)
--------------------
SKELETON + TESTS. The bank-build + paste algorithm is unit-tested.
Integration with Ultralytics' DataLoader is the Phase 4 EXP 4.4 work
(F.5) and lives in a follow-up PR.
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class CropEntry:
    """One GT instance crop in the instance bank."""

    image_path: str
    bbox_xywh: tuple[int, int, int, int]
    class_idx: int


@dataclass
class InstanceBank:
    """A bank of GT crops indexed by class for copy-paste augmentation.

    The bank stores PATHS + BBOXES, not the actual pixel arrays. Crops
    are loaded from disk on-demand at paste time. This keeps the bank
    small (a few MB) and lets it be pickled.
    """

    by_class: dict[int, list[CropEntry]] = field(default_factory=lambda: defaultdict(list))
    image_root: Optional[Path] = None

    def add(self, entry: CropEntry) -> None:
        self.by_class.setdefault(entry.class_idx, []).append(entry)

    def n_crops_for(self, class_idx: int) -> int:
        return len(self.by_class.get(class_idx, []))

    def total(self) -> int:
        return sum(len(v) for v in self.by_class.values())

    def summary(self) -> str:
        lines = [
            f"InstanceBank (image_root={self.image_root})",
            f"  Total crops:    {self.total():,}",
            f"  Total classes:  {len(self.by_class)}",
            "",
            "Per-class crop counts:",
        ]
        for cid in sorted(self.by_class):
            lines.append(f"  class {cid:3d}: {len(self.by_class[cid]):,} crops")
        return "\n".join(lines)


def build_instance_bank(
    train_json_path: str | Path,
    image_root: str | Path,
    *,
    cat_id_to_idx: Optional[dict[int, int]] = None,
    min_box_pixels: int = 16,
    verbose: bool = True,
) -> InstanceBank:
    """Build an InstanceBank from a COCO-format JSON.

    Parameters
    ----------
    train_json_path :
        Path to the COCO-format train_dataset.json.
    image_root :
        Directory under which file_name entries live (e.g. data/raw/images/train).
    cat_id_to_idx :
        Map from COCO category_id to 0-31 class_idx. If None, uses the
        identity (assumes COCO ids are already 0-31, which is FALSE for
        FathomNet 2026 -- import from configs.cat_id_mapping).
    min_box_pixels :
        Skip boxes whose smaller dimension is below this. Tiny boxes are
        too low-resolution to paste convincingly. Default 16px.
    verbose :
        Print a summary at the end.

    Returns
    -------
    InstanceBank
    """
    train_json_path = Path(train_json_path)
    image_root = Path(image_root)
    if not train_json_path.exists():
        raise FileNotFoundError(f"train JSON not found: {train_json_path}")

    if cat_id_to_idx is None:
        from configs.cat_id_mapping import cat_id_to_idx as _imported
        cat_id_to_idx = _imported

    with train_json_path.open() as f:
        coco = json.load(f)

    image_id_to_filename = {int(img["id"]): img["file_name"] for img in coco["images"]}

    bank = InstanceBank(image_root=image_root)
    skipped_small = 0
    skipped_unknown_cat = 0

    for ann in coco["annotations"]:
        iid = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        if w < min_box_pixels or h < min_box_pixels:
            skipped_small += 1
            continue
        if cat_id not in cat_id_to_idx:
            skipped_unknown_cat += 1
            continue

        fn = image_id_to_filename.get(iid)
        if fn is None:
            continue
        bank.add(CropEntry(
            image_path=str(image_root / fn),
            bbox_xywh=(int(x), int(y), int(w), int(h)),
            class_idx=cat_id_to_idx[cat_id],
        ))

    if verbose:
        print(bank.summary())
        print(f"  Skipped (too small):     {skipped_small:,}")
        print(f"  Skipped (unknown cat):   {skipped_unknown_cat:,}")

    return bank


DEFAULT_TIER_PROBABILITIES: dict[str, float] = {
    "critical": 0.85,
    "rare": 0.50,
    "mid": 0.25,
    "common": 0.075,
}

DEFAULT_TIER_THRESHOLDS: dict[str, tuple[int, int]] = {
    "critical": (0, 29),
    "rare":     (30, 99),
    "mid":      (100, 999),
    "common":   (1000, 10**9),
}


def build_default_schedule(
    instance_counts: dict[int, int],
    *,
    tier_probabilities: Optional[dict[str, float]] = None,
    tier_thresholds: Optional[dict[str, tuple[int, int]]] = None,
) -> dict[int, float]:
    """Build a per-class paste-probability schedule from per-class instance counts.

    Parameters
    ----------
    instance_counts :
        {class_idx: train instance count}. Source: Phase 0 cell that
        writes data/class_counts.json.
    tier_probabilities :
        Override the default probabilities (see DEFAULT_TIER_PROBABILITIES).
    tier_thresholds :
        Override the default tier brackets (see DEFAULT_TIER_THRESHOLDS).

    Returns
    -------
    dict[int, float]
        {class_idx: paste probability per image} for ALL keys in
        instance_counts. Probabilities live in [0, 1].
    """
    probs = tier_probabilities or DEFAULT_TIER_PROBABILITIES
    thr = tier_thresholds or DEFAULT_TIER_THRESHOLDS

    schedule: dict[int, float] = {}
    for cid, n in instance_counts.items():
        for tier_name, (lo, hi) in thr.items():
            if lo <= n <= hi:
                schedule[cid] = probs[tier_name]
                break
        else:
            schedule[cid] = probs.get("common", 0.05)
    return schedule


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU between two xywh boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _find_non_overlapping_location(
    img_h: int,
    img_w: int,
    crop_h: int,
    crop_w: int,
    existing_boxes_xywh: list[tuple[int, int, int, int]],
    *,
    max_iou: float = 0.1,
    max_attempts: int = 30,
    rng: Optional[random.Random] = None,
) -> Optional[tuple[int, int]]:
    """Try to find an (x, y) where pasting a crop_w x crop_h box doesn't
    overlap existing boxes too much. Returns None on failure."""
    if rng is None:
        rng = random.Random()
    if crop_h > img_h or crop_w > img_w:
        return None

    for _ in range(max_attempts):
        x = rng.randint(0, img_w - crop_w)
        y = rng.randint(0, img_h - crop_h)
        candidate = (x, y, crop_w, crop_h)
        ok = all(_box_iou(candidate, b) <= max_iou for b in existing_boxes_xywh)
        if ok:
            return (x, y)
    return None


def class_aware_copy_paste(
    img: np.ndarray,
    boxes_xywh: list[tuple[int, int, int, int]],
    labels: list[int],
    bank: InstanceBank,
    schedule: dict[int, float],
    *,
    max_paste_per_class: int = 1,
    max_iou: float = 0.1,
    max_attempts_per_paste: int = 30,
    rng: Optional[random.Random] = None,
    image_loader=None,
    verbose: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[int]]:
    """Apply class-aware copy-paste to one training sample.

    Parameters
    ----------
    img :
        Source image, shape (H, W, 3), dtype uint8.
    boxes_xywh :
        Existing GT boxes for this image, in pixel xywh.
    labels :
        Class indices (0-31) for each existing box.
    bank :
        InstanceBank built via build_instance_bank().
    schedule :
        {class_idx: paste probability} from build_default_schedule().
    max_paste_per_class :
        How many crops of each class to paste, AT MOST, when paste fires.
        Default 1.
    max_iou : float
        A candidate paste location is rejected if it has IoU > max_iou
        with ANY existing box (including previously-pasted ones in this call).
    max_attempts_per_paste :
        Random-location attempts per crop before giving up.
    rng :
        Optional random.Random instance for reproducible tests.
    image_loader :
        Callable (image_path -> np.ndarray) to load source crops. If None,
        uses cv2.imread. Pass a stub for unit-testing without disk I/O.
    verbose :
        If True, print every paste / skip decision.

    Returns
    -------
    (img_out, boxes_out, labels_out)
        Modified image (same shape, dtype uint8) and updated boxes/labels lists.
        If no pastes occurred, the originals are returned unmodified.
    """
    if rng is None:
        rng = random.Random()

    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"img must be HxWx3 uint8, got shape={img.shape} dtype={img.dtype}")
    if len(boxes_xywh) != len(labels):
        raise ValueError(f"boxes ({len(boxes_xywh)}) vs labels ({len(labels)}) length mismatch")

    if image_loader is None:
        try:
            import cv2

            def image_loader(p: str) -> np.ndarray:
                arr = cv2.imread(p)
                if arr is None:
                    raise FileNotFoundError(f"cv2.imread failed: {p}")
                return arr
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for default image loading. "
                "Install via `pip install opencv-python-headless`, or pass a custom image_loader."
            ) from exc

    img_h, img_w = img.shape[:2]
    out_img = img.copy()
    out_boxes = list(boxes_xywh)
    out_labels = list(labels)

    for class_idx, prob in schedule.items():
        if rng.random() >= prob:
            continue
        n_crops = bank.n_crops_for(class_idx)
        if n_crops == 0:
            if verbose:
                print(f"  [skip] class {class_idx}: no crops in bank")
            continue
        for _ in range(max_paste_per_class):
            entry = bank.by_class[class_idx][rng.randrange(n_crops)]
            try:
                src_img = image_loader(entry.image_path)
            except Exception as exc:
                if verbose:
                    print(f"  [skip] class {class_idx}: load failed {exc}")
                continue
            sx, sy, sw, sh = entry.bbox_xywh
            crop = src_img[sy:sy + sh, sx:sx + sw]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            loc = _find_non_overlapping_location(
                img_h, img_w, ch, cw, out_boxes,
                max_iou=max_iou, max_attempts=max_attempts_per_paste, rng=rng,
            )
            if loc is None:
                if verbose:
                    print(f"  [skip] class {class_idx}: no non-overlap location after {max_attempts_per_paste}")
                continue
            x, y = loc
            out_img[y:y + ch, x:x + cw] = crop
            out_boxes.append((x, y, cw, ch))
            out_labels.append(class_idx)
            if verbose:
                print(f"  [paste] class {class_idx} -> ({x}, {y}, {cw}x{ch})")

    return out_img, out_boxes, out_labels


def write_bank(bank: InstanceBank, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(bank, f)
    return out_path.resolve()


def load_bank(in_path: str | Path) -> InstanceBank:
    in_path = Path(in_path)
    with in_path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, InstanceBank):
        raise TypeError(f"{in_path} did not contain an InstanceBank; got {type(obj).__name__}")
    return obj


__all__ = [
    "CropEntry",
    "InstanceBank",
    "build_instance_bank",
    "build_default_schedule",
    "class_aware_copy_paste",
    "write_bank",
    "load_bank",
    "DEFAULT_TIER_PROBABILITIES",
    "DEFAULT_TIER_THRESHOLDS",
]
