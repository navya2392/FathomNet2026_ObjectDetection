"""Per-class NMS IoU threshold sweep for Phase 7 EXP 7.6.

Block reference: I.8 in master_checklist.txt.

What this does
--------------
Standard NMS uses a single IoU threshold (e.g., 0.7) for ALL classes.
But the optimal NMS threshold is class-dependent:

- Densely-clustered classes (e.g., sea urchins, where multiple
  individuals legitimately overlap) need a HIGHER NMS threshold,
  otherwise NMS suppresses legitimate distinct detections.
- Sparse / isolated classes (e.g., a single sea slug) tolerate a
  LOWER NMS threshold; tighter NMS reduces FP duplicates.

This script:
1. Runs inference on the val fold ONCE at conf=0.001 (cheap), saving
   raw boxes BEFORE NMS to disk.
2. Sweeps a grid of per-class NMS thresholds (default {0.3, 0.5, 0.7})
   on the saved raw boxes (CPU-only, ~30 min total).
3. For each candidate threshold per class, computes that class's AP
   on val, picks the best.
4. Writes the per-class threshold map to notes/phase7_per_class_nms.md.

The map is then consumed at submission time -- inference uses class-
specific NMS thresholds via Ultralytics' agnostic_nms=False + a
custom post-processing step.

Cost
----
Inference once (saved): ~30 sec on RTX 4090.
Per-class threshold sweep: ~30 min CPU.
Total: ~30 min, ~$0.20 on RunPod.

Status (May 1, 2026)
--------------------
WRITTEN. Uses pycocotools for the per-class AP. Has NOT been run
end-to-end -- needs val fold predictions and `pip install pycocotools`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _box_iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two box sets in xywh, returns (Na, Nb)."""
    ax = a[:, 0:1]
    ay = a[:, 1:2]
    aw = a[:, 2:3]
    ah = a[:, 3:4]
    bx = b[:, 0]
    by = b[:, 1]
    bw = b[:, 2]
    bh = b[:, 3]

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    ix1 = np.maximum(ax, bx)
    iy1 = np.maximum(ay, by)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = (aw * ah) + (bw * bh) - inter
    return np.where(union > 0, inter / union, 0.0)


def per_class_nms(
    boxes_xywh: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    iou_thresholds: dict[int, float],
    default_iou: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply NMS with a per-class IoU threshold.

    Parameters
    ----------
    boxes_xywh : (N, 4) array of xywh boxes.
    scores : (N,) confidence scores.
    labels : (N,) class indices.
    iou_thresholds : {class_idx: iou_threshold} per-class thresholds.
        Classes not in this dict use default_iou.
    default_iou : float
        Fallback NMS threshold for classes not in the dict.

    Returns
    -------
    (kept_boxes, kept_scores, kept_labels)
    """
    if len(boxes_xywh) == 0:
        return boxes_xywh, scores, labels

    keep_indices: list[int] = []
    for cls in np.unique(labels):
        cls_mask = labels == cls
        cls_idx = np.where(cls_mask)[0]
        cls_boxes = boxes_xywh[cls_idx]
        cls_scores = scores[cls_idx]

        order = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]
        cls_orig = cls_idx[order]

        thr = iou_thresholds.get(int(cls), default_iou)

        suppressed = np.zeros(len(cls_boxes), dtype=bool)
        for i in range(len(cls_boxes)):
            if suppressed[i]:
                continue
            keep_indices.append(int(cls_orig[i]))
            if i + 1 < len(cls_boxes):
                ious = _box_iou_xywh(cls_boxes[i:i + 1], cls_boxes[i + 1:])[0]
                suppressed[i + 1:] |= ious > thr

    keep_indices.sort()
    keep_idx = np.array(keep_indices, dtype=np.int64)
    return boxes_xywh[keep_idx], scores[keep_idx], labels[keep_idx]


def evaluate_per_class_ap(
    raw_predictions: dict[int, np.ndarray],
    raw_scores: dict[int, np.ndarray],
    raw_labels: dict[int, np.ndarray],
    val_gt_json_path: Path,
    iou_thresholds: dict[int, float],
    *,
    default_iou: float = 0.5,
) -> dict[int, float]:
    """Apply per-class NMS, write predictions JSON, run pycocotools eval, return per-class AP@.50:.95."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("pycocotools required. `pip install pycocotools`.") from exc

    from configs.cat_id_mapping import idx_to_cat_id

    pred_anns = []
    for image_id, boxes in raw_predictions.items():
        scores = raw_scores[image_id]
        labels = raw_labels[image_id]
        kept_boxes, kept_scores, kept_labels = per_class_nms(
            boxes, scores, labels,
            iou_thresholds=iou_thresholds, default_iou=default_iou,
        )
        for k in range(len(kept_boxes)):
            cls_idx = int(kept_labels[k])
            if cls_idx not in idx_to_cat_id:
                continue
            x, y, w, h = kept_boxes[k]
            pred_anns.append({
                "image_id": int(image_id),
                "category_id": int(idx_to_cat_id[cls_idx]),
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(kept_scores[k]),
            })

    if not pred_anns:
        return {}

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(pred_anns, f)
        pred_path = f.name

    coco_gt = COCO(str(val_gt_json_path))
    coco_dt = coco_gt.loadRes(pred_path)
    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    e.evaluate()
    e.accumulate()

    per_class_ap: dict[int, float] = {}
    for cls_idx in idx_to_cat_id:
        cat_id = idx_to_cat_id[cls_idx]
        if cat_id not in e.params.catIds:
            continue
        cat_pos = list(e.params.catIds).index(cat_id)
        precision = e.eval["precision"][:, :, cat_pos, 0, -1]
        valid = precision[precision > -1]
        ap = float(valid.mean()) if len(valid) > 0 else 0.0
        per_class_ap[cls_idx] = ap

    Path(pred_path).unlink(missing_ok=True)
    return per_class_ap


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-class NMS IoU threshold sweep on val predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-preds-pkl", type=Path, required=True,
                   help="Pickled raw predictions BEFORE NMS, dict of "
                        "{image_id: dict(boxes_xywh=..., scores=..., labels=...)}.")
    p.add_argument("--val-gt-json", type=Path, required=True,
                   help="Val fold ground-truth COCO JSON.")
    p.add_argument("--candidate-thresholds", type=float, nargs="+",
                   default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--out-json", type=Path,
                   default=_REPO_ROOT / "notes" / "phase7_per_class_nms.json")
    p.add_argument("--out-md", type=Path,
                   default=_REPO_ROOT / "notes" / "phase7_per_class_nms.md")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.raw_preds_pkl.exists():
        print(f"ERROR: raw predictions not found: {args.raw_preds_pkl}", file=sys.stderr)
        return 2

    import pickle
    with args.raw_preds_pkl.open("rb") as f:
        raw = pickle.load(f)
    raw_predictions = {iid: np.asarray(v["boxes_xywh"]) for iid, v in raw.items()}
    raw_scores = {iid: np.asarray(v["scores"]) for iid, v in raw.items()}
    raw_labels = {iid: np.asarray(v["labels"]) for iid, v in raw.items()}

    from configs.cat_id_mapping import idx_to_cat_id
    all_classes = sorted(idx_to_cat_id)

    print(f"Sweeping {len(args.candidate_thresholds)} thresholds across {len(all_classes)} classes...")
    candidate_results: dict[float, dict[int, float]] = {}
    for thr in args.candidate_thresholds:
        global_thresholds = {c: thr for c in all_classes}
        ap_dict = evaluate_per_class_ap(
            raw_predictions, raw_scores, raw_labels,
            args.val_gt_json, iou_thresholds=global_thresholds,
        )
        candidate_results[thr] = ap_dict
        print(f"  thr={thr}: per-class AP mean={np.mean(list(ap_dict.values())):.4f}")

    best_per_class: dict[int, tuple[float, float]] = {}
    for cls in all_classes:
        best_thr, best_ap = max(
            ((thr, candidate_results[thr].get(cls, 0.0)) for thr in args.candidate_thresholds),
            key=lambda x: x[1],
        )
        best_per_class[cls] = (best_thr, best_ap)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(c): {"best_thr": t, "best_ap": a} for c, (t, a) in best_per_class.items()}
    with args.out_json.open("w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nWrote per-class threshold JSON: {args.out_json}")

    lines = [
        "# Phase 7 EXP 7.6 — per-class NMS threshold tuning",
        "",
        f"Sweep thresholds: {args.candidate_thresholds}",
        "",
        "| class_idx | best NMS thr | best per-class AP@[.50:.95] |",
        "|---:|---:|---:|",
    ]
    for cls in sorted(best_per_class):
        thr, ap = best_per_class[cls]
        lines.append(f"| {cls} | {thr} | {ap:.4f} |")
    lines.append("")
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote per-class threshold report: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
