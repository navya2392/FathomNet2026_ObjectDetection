"""Open-vocabulary inference for rare-class pseudo-labeling (Phase 5 EXP 5.4 stretch).

Block reference: F.9.5 in master_checklist.txt.

What problem does this solve
----------------------------
Phase 0 EDA showed 14 classes have <100 train instances and 7 classes
have <30. Even with RFS + copy-paste + Soft Teacher, the model has very
little supervision for these classes. Open-vocabulary detectors
(GroundingDINO, OWL-ViT) were trained on internet-scale image-text
pairs and can detect "sea slug" or "amphipod" zero-shot from text
prompts -- WITHOUT seeing a single FathomNet labeled instance.

Strategy
--------
1. Build a text query for each rare class (manual curation; class names
   alone often work, but adding visual hints like "deep-sea sea slug"
   can help).
2. Run an open-vocabulary detector on the FathomNet TRAIN set images
   with these queries.
3. Filter the resulting boxes:
   - Confidence >= threshold (typically 0.3 for OWL-ViT, 0.35 for GroundingDINO).
   - Box size sanity (drop tiny noise boxes < 16x16 pixels).
   - Suppress boxes overlapping existing GT (we trust labelers).
4. Convert to a JSON of PseudoLabel objects (matches src/soft_teacher.py
   schema), then merge into the training set via merge_gt_and_pseudo().

Why we're using OWL-ViT not GroundingDINO
-----------------------------------------
OWL-ViT loads cleanly from `transformers` (no custom build). GroundingDINO
requires building an extension; on RunPod it works but adds 5-10 min
of setup. For a zero-CV-eval, "fire-once-at-night-and-see-what-happens"
job, OWL-ViT is the right tradeoff.

Phase 5 EXP 5.4 ablates GroundingDINO too if OWL-ViT helps; if OWL-ViT
doesn't help, we skip GroundingDINO (cheap enough to test, expensive
enough not to do twice).

Cost
----
OWL-ViT-base on a 4090: ~0.5 sec/image. 6,463 train images -> ~55 min.
With multi-query batching: ~30 min. Costs ~$0.30.

Status (May 1, 2026)
--------------------
SCRIPT WRITTEN. Has NOT been run end-to-end -- requires Phase 6 to
have finished and the rare-class list to be locked in. Output JSON
plugs into src/soft_teacher.py via merge_gt_and_pseudo().
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.soft_teacher import PseudoLabel


DEFAULT_RARE_CLASS_QUERIES: dict[int, str] = {
    22: "a deep sea sea slug nudibranch",
    14: "a deep sea isopod crustacean",
    23: "a deep sea cephalopod squid",
    9: "a deep sea pycnogonid sea spider",
    8: "a deep sea worm",
    24: "a deep sea cucumber holothurian",
    27: "a deep sea coral cnidarian",
}


def load_owlvit_model(model_id: str = "google/owlvit-base-patch32"):
    """Lazy-import + load OWL-ViT. Returns (processor, model)."""
    try:
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
    except ImportError as exc:
        raise RuntimeError(
            "transformers package required. Install via "
            "`pip install transformers`."
        ) from exc
    import torch
    processor = OwlViTProcessor.from_pretrained(model_id)
    model = OwlViTForObjectDetection.from_pretrained(model_id)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return processor, model


def predict_one_image(
    image_path: Path,
    processor,
    model,
    queries: list[str],
    *,
    score_threshold: float = 0.3,
    min_box_pixels: int = 16,
):
    """Run OWL-ViT on one image with a list of text queries.

    Returns a list of (query_idx, score, bbox_xywh_pixels) tuples.
    """
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    inputs = processor(text=[queries], images=img, return_tensors="pt")
    if next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[img_h, img_w]],
                                device=next(model.parameters()).device)
    results = processor.post_process_object_detection(
        outputs=outputs, threshold=score_threshold, target_sizes=target_sizes,
    )[0]

    out = []
    for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        w = x2 - x1
        h = y2 - y1
        if w < min_box_pixels or h < min_box_pixels:
            continue
        out.append((int(label_idx), float(score), (x1, y1, w, h)))
    return out


def run_open_vocab_pseudo_labeling(
    train_json_path: Path,
    image_root: Path,
    *,
    queries_by_class_idx: dict[int, str] = DEFAULT_RARE_CLASS_QUERIES,
    score_threshold: float = 0.3,
    min_box_pixels: int = 16,
    max_per_image_per_class: int = 3,
    model_id: str = "google/owlvit-base-patch32",
    image_id_to_filename: Optional[dict[int, str]] = None,
    verbose: bool = True,
) -> list[PseudoLabel]:
    """Run open-vocabulary detection across all train images and return PseudoLabel list.

    Parameters
    ----------
    train_json_path :
        train_dataset.json (used to get the image_id -> file_name map).
    image_root :
        Directory containing the actual image files.
    queries_by_class_idx :
        {class_idx (0-31): text query} for each rare class to detect.
    score_threshold :
        Minimum OWL-ViT confidence.
    min_box_pixels :
        Drop boxes smaller than this in either dimension.
    max_per_image_per_class :
        Per-image, per-class cap (top-K by score).
    model_id :
        HuggingFace model id. 'google/owlvit-base-patch32' is fast;
        '...-large-patch14' is more accurate but 4x slower.
    image_id_to_filename :
        Override the {image_id: file_name} mapping. If None, parse from train_json.
    verbose :
        Print progress every 100 images.

    Returns
    -------
    list[PseudoLabel]
        Ready to feed into src.soft_teacher.apply_pseudo_label_filters().
    """
    if image_id_to_filename is None:
        with train_json_path.open() as f:
            coco = json.load(f)
        image_id_to_filename = {int(img["id"]): img["file_name"] for img in coco["images"]}

    if verbose:
        print(f"Loading OWL-ViT ({model_id})...")
    processor, model = load_owlvit_model(model_id)

    class_indices = list(queries_by_class_idx.keys())
    queries = [queries_by_class_idx[c] for c in class_indices]
    if verbose:
        print(f"Queries:")
        for c, q in zip(class_indices, queries):
            print(f"  class {c}: {q!r}")

    pseudo: list[PseudoLabel] = []
    n_processed = 0
    t0 = time.time()
    skipped_missing = 0

    image_ids = sorted(image_id_to_filename)
    for iid in image_ids:
        fn = image_id_to_filename[iid]
        path = image_root / fn
        if not path.exists():
            skipped_missing += 1
            continue
        try:
            preds = predict_one_image(
                path, processor, model, queries,
                score_threshold=score_threshold,
                min_box_pixels=min_box_pixels,
            )
        except Exception as exc:
            if verbose:
                print(f"  [skip] {path}: {exc.__class__.__name__}: {exc}")
            continue

        per_class_keep: dict[int, list[tuple[float, tuple[float, float, float, float]]]] = {}
        for query_idx, score, bbox in preds:
            cidx = class_indices[query_idx]
            per_class_keep.setdefault(cidx, []).append((score, bbox))
        for cidx, items in per_class_keep.items():
            items.sort(key=lambda x: -x[0])
            for score, bbox in items[:max_per_image_per_class]:
                pseudo.append(PseudoLabel(
                    image_id=iid, class_idx=cidx, bbox_xywh=bbox, score=score,
                ))

        n_processed += 1
        if verbose and n_processed % 100 == 0:
            elapsed = time.time() - t0
            rate = n_processed / max(elapsed, 1e-6)
            eta = (len(image_ids) - n_processed) / max(rate, 1e-6) / 60
            print(f"  [{n_processed}/{len(image_ids)}] {rate:.2f} img/s, "
                  f"ETA {eta:.1f} min, kept {len(pseudo):,} pseudo-labels so far")

    if verbose:
        from collections import Counter
        per_class = Counter(p.class_idx for p in pseudo)
        print(f"\nDone. {n_processed:,} images processed in {(time.time() - t0) / 60:.1f} min")
        print(f"  Skipped (missing file): {skipped_missing:,}")
        print(f"  Total pseudo-labels:    {len(pseudo):,}")
        print(f"  Per-class breakdown:")
        for cidx in sorted(per_class):
            print(f"    class {cidx}: {per_class[cidx]:,}")

    return pseudo


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Open-vocabulary pseudo-labeling with OWL-ViT for rare classes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--train-json",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "train_dataset.json",
    )
    p.add_argument(
        "--image-root",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "images" / "train",
    )
    p.add_argument(
        "--queries-json",
        type=Path,
        default=None,
        help="Path to a JSON file mapping class_idx -> text query. If absent, uses DEFAULT_RARE_CLASS_QUERIES.",
    )
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--min-box-pixels", type=int, default=16)
    p.add_argument("--max-per-image-per-class", type=int, default=3)
    p.add_argument(
        "--model-id",
        default="google/owlvit-base-patch32",
        help="HF model id. Use '...-large-patch14' for higher accuracy at 4x cost.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON file (list of PseudoLabel-shaped dicts).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.queries_json:
        with args.queries_json.open() as f:
            raw = json.load(f)
        queries = {int(k): v for k, v in raw.items()}
    else:
        queries = DEFAULT_RARE_CLASS_QUERIES

    pseudo = run_open_vocab_pseudo_labeling(
        train_json_path=args.train_json,
        image_root=args.image_root,
        queries_by_class_idx=queries,
        score_threshold=args.score_threshold,
        min_box_pixels=args.min_box_pixels,
        max_per_image_per_class=args.max_per_image_per_class,
        model_id=args.model_id,
        verbose=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    serialized = [
        {
            "image_id": p.image_id,
            "class_idx": p.class_idx,
            "bbox_xywh": list(p.bbox_xywh),
            "score": p.score,
        }
        for p in pseudo
    ]
    with args.out.open("w") as f:
        json.dump(serialized, f, indent=2)
    print(f"\nWrote {len(serialized):,} pseudo-labels to {args.out}")
    print(f"Next: feed these to src.soft_teacher.apply_pseudo_label_filters() then merge_gt_and_pseudo()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
