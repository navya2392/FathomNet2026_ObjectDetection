"""Run one round of Soft Teacher pseudo-labeling for Phase 3 EXP 3.5 / Phase 5 EXP 5.2.

Block reference: E.6 + G.3 in master_checklist.txt.

Pipeline (one round)
--------------------
1. Load a teacher checkpoint (the model trained in the prior round / phase).
2. Run inference on EVERY train image, getting raw bbox predictions.
3. Convert the raw predictions to PseudoLabel objects.
4. Apply the 4-stage filter pipeline from src.soft_teacher.apply_pseudo_label_filters:
   a. Score >= score_threshold
   b. Drop predictions for classes already GT-labeled in this image
   c. Drop predictions overlapping existing GT (IoU > iou_threshold)
   d. Per-image and per-class global caps
5. Merge accepted pseudo-labels with original GT into a new COCO JSON via
   src.soft_teacher.merge_gt_and_pseudo.
6. Write the merged JSON to data/raw/train_dataset.softteacher_roundN.json.
7. Write a Markdown report to notes/phase3_softteacher_roundN.md with stats.

The next training round uses the merged JSON as its --data input.

This script does NOT re-train. It only generates the pseudo-labels +
merged JSON. Re-training is a separate `train_phase2_baseline.py` call
with the new YAML pointing at the merged JSON. Why decoupled?

- The pseudo-labeling step is dataset-wide; you want to inspect the
  filter stats before committing GPU time to a retrain.
- If the filter stats look bad (e.g., teacher generated 50,000 sea
  urchin pseudo-labels), you can adjust thresholds and re-run filtering
  cheaply without re-running the slow teacher inference step.

Cost
----
Teacher inference on 6,463 train images at 640x640 on RTX 4090: ~3 min.
Filter + merge: ~5 sec. Negligible.

Status (May 1, 2026)
--------------------
WRITTEN. Has NOT been run end-to-end -- needs a Phase 2 / Phase 3
teacher checkpoint and `pip install ultralytics` on the pod.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.soft_teacher import (
    PseudoLabel,
    apply_pseudo_label_filters,
    merge_gt_and_pseudo,
)


def _build_gt_by_image(coco: dict, cat_id_to_idx: dict[int, int]) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    """Build {image_id: [(class_idx, bbox_xywh), ...]} from a COCO json."""
    out: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for ann in coco["annotations"]:
        cat_id = int(ann["category_id"])
        if cat_id not in cat_id_to_idx:
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            continue
        cls = cat_id_to_idx[cat_id]
        x, y, w, h = bbox
        out[int(ann["image_id"])].append((cls, (float(x), float(y), float(w), float(h))))
    return out


def _build_image_id_lookup(coco: dict) -> dict[str, int]:
    """{file_name: image_id}."""
    return {img["file_name"]: int(img["id"]) for img in coco["images"]}


def run_teacher_inference(
    weights_path: Path,
    image_root: Path,
    image_id_lookup: dict[str, int],
    cat_idx_to_id: dict[int, int],
    *,
    imgsz: int = 640,
    conf: float = 0.05,
    iou: float = 0.7,
    batch: int = 16,
    device: str = "0",
    verbose: bool = True,
) -> list[PseudoLabel]:
    """Run the teacher Ultralytics model on all train images and yield PseudoLabel objects."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics package required. `pip install ultralytics`.") from exc

    model = YOLO(str(weights_path))
    image_paths = sorted(p for p in image_root.iterdir() if p.is_file())
    if verbose:
        print(f"Running teacher inference on {len(image_paths):,} images "
              f"(weights={weights_path}, imgsz={imgsz}, conf={conf})")

    pseudo: list[PseudoLabel] = []
    t0 = time.time()
    n_processed = 0

    results_iter = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        batch=batch,
        device=device,
        verbose=False,
        save=False,
        stream=True,
    )

    for image_path, result in zip(image_paths, results_iter):
        image_id = image_id_lookup.get(image_path.name)
        if image_id is None:
            continue
        if result.boxes is None or len(result.boxes) == 0:
            n_processed += 1
            continue
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)

        for k in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[k].tolist()]
            cls_idx = int(labels[k])
            score = float(scores[k])
            if cls_idx not in cat_idx_to_id:
                continue
            pseudo.append(PseudoLabel(
                image_id=image_id,
                class_idx=cls_idx,
                bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                score=score,
            ))
        n_processed += 1
        if verbose and n_processed % 200 == 0:
            elapsed = time.time() - t0
            rate = n_processed / max(elapsed, 1e-6)
            eta = (len(image_paths) - n_processed) / max(rate, 1e-6) / 60
            print(f"  [{n_processed}/{len(image_paths)}] {rate:.1f} img/s, "
                  f"ETA {eta:.1f} min, raw pseudo-labels: {len(pseudo):,}")

    if verbose:
        print(f"\nTeacher inference complete: {n_processed:,} images in {(time.time() - t0) / 60:.1f} min")
        print(f"Total raw pseudo-labels: {len(pseudo):,}")
    return pseudo


def write_round_report(
    out_path: Path,
    *,
    weights_path: Path,
    train_json_path: Path,
    n_raw: int,
    filter_stats,
    n_merged: int,
    output_json_path: Path,
    per_class_after_filter: dict[int, int],
) -> None:
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# Soft Teacher round report",
        "",
        f"_Generated by `scripts/run_soft_teacher_round.py` on {now}_",
        "",
        f"## Inputs",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| teacher weights | `{weights_path}` |",
        f"| original GT JSON | `{train_json_path}` |",
        f"| merged output JSON | `{output_json_path}` |",
        "",
        f"## Filter pipeline stats",
        "",
        f"| Stage | Count | Note |",
        f"|---|---:|---|",
        f"| Raw teacher predictions | {n_raw:,} | Before any filtering |",
        f"| After score threshold | {filter_stats.n_kept_after_score:,} | |",
        f"| After class-already-GT-in-image filter | {filter_stats.n_kept_after_class_in_image_filter:,} | |",
        f"| After IoU-with-GT filter | {filter_stats.n_kept_after_iou_with_gt:,} | |",
        f"| After per-image cap | {filter_stats.n_kept_after_per_image_cap:,} | |",
        f"| Final accepted pseudo-labels | {n_merged:,} | Merged into new JSON |",
        "",
        f"## Per-class accepted pseudo-labels",
        "",
        f"| class_idx | accepted count |",
        f"|---:|---:|",
    ]
    for cidx in sorted(per_class_after_filter):
        lines.append(f"| {cidx} | {per_class_after_filter[cidx]:,} |")
    lines.extend([
        "",
        f"## Sanity checks",
        "",
        f"- If a single class dominates accepted pseudo-labels (>50% of total), "
        f"consider tightening `--max-per-class-globally` for that class.",
        f"- If accepted total is < 10% of raw, score_threshold may be too high.",
        f"- If accepted total is > 50% of raw, score_threshold may be too low (low-quality pseudo).",
        "",
        f"## Next step",
        "",
        f"Re-train the student on the merged JSON:",
        "",
        f"```bash",
        f"python scripts/train_phase2_baseline.py \\",
        f"    --init <prior_best.pt> --epochs 50 --imgsz 640 --fold 0 \\",
        f"    --data data/folds/fold0_softteacher.yaml \\",
        f"    --name p3_softteacher_round_X",
        f"```",
        "",
        f"You'll need to build `fold0_softteacher.yaml` pointing at the merged JSON; ",
        f"that's a 5-line edit of `fold0.yaml`.",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one round of Soft Teacher pseudo-labeling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", type=Path, required=True,
                   help="Teacher Ultralytics .pt checkpoint.")
    p.add_argument("--train-json", type=Path,
                   default=_REPO_ROOT / "data" / "raw" / "train_dataset.json")
    p.add_argument("--image-root", type=Path,
                   default=_REPO_ROOT / "data" / "raw" / "images" / "train")
    p.add_argument("--out-json", type=Path, required=True,
                   help="Output merged COCO JSON path.")
    p.add_argument("--report", type=Path, default=None,
                   help="Markdown report output. Defaults to notes/softteacher_<out_json_stem>.md.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.05,
                   help="Inference confidence threshold (low; SAFE filtering happens later).")
    p.add_argument("--score-threshold", type=float, default=0.5,
                   help="POST-INFERENCE filter threshold (Soft Teacher paper uses 0.7; we use 0.5).")
    p.add_argument("--iou-with-gt-threshold", type=float, default=0.3)
    p.add_argument("--max-per-image", type=int, default=50)
    p.add_argument("--max-per-class-globally", type=int, default=None,
                   help="Per-class global cap; default no cap. Use e.g. 5000 for sea urchin to prevent flood.")
    p.add_argument("--device", default="0", help="CUDA device for teacher inference. Use 'cpu' to test on laptop.")
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.weights.exists():
        print(f"ERROR: teacher weights not found: {args.weights}", file=sys.stderr)
        return 2
    if not args.train_json.exists():
        print(f"ERROR: train json not found: {args.train_json}", file=sys.stderr)
        return 2

    from configs.cat_id_mapping import cat_id_to_idx, idx_to_cat_id

    with args.train_json.open() as f:
        coco = json.load(f)

    image_id_lookup = _build_image_id_lookup(coco)
    gt_by_image = _build_gt_by_image(coco, cat_id_to_idx)

    pseudo_raw = run_teacher_inference(
        weights_path=args.weights,
        image_root=args.image_root,
        image_id_lookup=image_id_lookup,
        cat_idx_to_id=idx_to_cat_id,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        batch=args.batch,
    )

    max_per_class_global = (
        {idx: args.max_per_class_globally for idx in idx_to_cat_id}
        if args.max_per_class_globally is not None
        else None
    )

    print(f"\nApplying filter pipeline...")
    accepted, stats = apply_pseudo_label_filters(
        pseudo_raw,
        gt_by_image=gt_by_image,
        score_threshold=args.score_threshold,
        iou_with_gt_threshold=args.iou_with_gt_threshold,
        suppress_class_already_gt_in_image=True,
        max_per_image=args.max_per_image,
        max_per_class_globally=max_per_class_global,
        return_stats=True,
    )

    per_class = Counter(p.class_idx for p in accepted)
    print(f"  accepted: {len(accepted):,} pseudo-labels across {len(per_class)} classes")

    print(f"\nMerging with original GT...")
    merged = merge_gt_and_pseudo(coco, accepted, idx_to_cat_id=idx_to_cat_id)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(merged, f)
    print(f"Wrote merged JSON: {args.out_json}")
    print(f"  Total annotations: {len(merged['annotations']):,} "
          f"(was {len(coco['annotations']):,}; +{len(merged['annotations']) - len(coco['annotations']):,})")

    report_path = args.report or _REPO_ROOT / "notes" / f"softteacher_{args.out_json.stem}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_round_report(
        report_path,
        weights_path=args.weights,
        train_json_path=args.train_json,
        n_raw=len(pseudo_raw),
        filter_stats=stats,
        n_merged=len(accepted),
        output_json_path=args.out_json,
        per_class_after_filter=dict(per_class),
    )
    print(f"Wrote round report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
