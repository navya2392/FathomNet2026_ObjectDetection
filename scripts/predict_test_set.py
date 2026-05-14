"""Run a trained model on the FathomNet 2026 test set and emit a Kaggle CSV.

Block reference: C.4 (first submission), F.11 (Phase 4), J.x (Phase 7).

End-to-end pipeline
-------------------
  trained .pt
    -> ultralytics.YOLO.predict() over data/raw/images/test/*
    -> src.submit.from_ultralytics_results (DataFrame in pred_df schema)
    -> src.submit.yolo_preds_to_submission_csv (8-column CSV)
    -> src.submit.validate_submission (pre-flight check; matches Kaggle)

Optional: if --val-fold N is provided AND a YOLO val-style labels dir
exists for that fold, also runs the official scorer against the val
split for a sanity-check val mAP@[.50:.95].

This is the SINGLE script you should call to make a Kaggle CSV. Phase 7
ensemble inference (`scripts/ensemble_predict.py` -- TBD) will use the same
src.submit helpers but replace the single .pt with a list of weights +
WBF combiner.

Why we don't use Ultralytics' built-in `model.predict(save_txt=True)`
--------------------------------------------------------------------
Ultralytics' default text output uses YOLO-format normalized xywh and
0-indexed class IDs. Kaggle wants COCO-format absolute xywh and
NON-CONTIGUOUS category_ids (1-41 with gaps). The reverse mapping lives
in src/submit.py::idx_to_cat_id and is the FLAG 1 trap that Phase 0
warned about.

Usage
-----
    # Standard: single model, default test imgsz, conf threshold 0.001
    # (low conf since the scorer evaluates with up to 100 detections per image)
    python scripts/predict_test_set.py \
        --weights weights/runs/p2_baseline_yolo11m_coco/weights/best.pt \
        --out submissions/p2_baseline.csv

    # With test-time augmentation (slower, +0.5-1 mAP typical)
    python scripts/predict_test_set.py \
        --weights weights/runs/p2_full_mbari_315k/weights/best.pt \
        --out submissions/p2_mbari.csv \
        --tta --imgsz 1024

    # On RunPod with custom data location
    python scripts/predict_test_set.py \
        --weights /workspace/weights/runs/p2_full/best.pt \
        --out /workspace/submissions/p2_full.csv \
        --test-images /workspace/data/raw/images/test \
        --test-json /workspace/data/raw/test_dataset.json

    # Tier 1A: horizontal-flip view (boxes mapped back to original coords)
    python scripts/predict_test_set.py \
        --weights weights/runs/.../best.pt \
        --out submissions/tag_hflip.csv \
        --imgsz 1024 --hflip

    # Tier 1A smoke: first N images only (sorted filenames — not for real submit)
    python scripts/predict_test_set.py ... --max-images 50 --no-validate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TEST_IMAGES = REPO_ROOT / "data" / "raw" / "images" / "test"
DEFAULT_TEST_JSON = REPO_ROOT / "data" / "raw" / "test_dataset.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--weights", required=True, type=Path,
                   help="Path to trained .pt checkpoint")
    p.add_argument("--out", required=True, type=Path,
                   help="Output CSV path (parent dir auto-created)")
    p.add_argument("--test-images", type=Path, default=DEFAULT_TEST_IMAGES,
                   help=f"Test image dir (default: {DEFAULT_TEST_IMAGES.relative_to(REPO_ROOT)})")
    p.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON,
                   help=f"Test COCO JSON (default: {DEFAULT_TEST_JSON.relative_to(REPO_ROOT)})")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference image size (default 640; try 1024 or 1280 for low-res test images)")
    p.add_argument("--batch", type=int, default=16,
                   help="Inference batch size (default 16)")
    p.add_argument("--conf", type=float, default=0.001,
                   help="Confidence threshold (default 0.001 -- very low; "
                        "we keep many detections so the scorer can pick "
                        "up to 100 per image)")
    p.add_argument("--iou", type=float, default=0.65,
                   help="NMS IoU threshold (default 0.65)")
    p.add_argument("--max-det", type=int, default=100,
                   help="Max detections per image (default 100, matches Kaggle scorer)")
    p.add_argument("--device", default="",
                   help="GPU device(s); empty = auto")
    p.add_argument("--tta", action="store_true",
                   help="Enable test-time augmentation (slower, +0.5-1 mAP typical)")
    p.add_argument("--half", action="store_true",
                   help="FP16 inference (faster on supported GPUs)")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip validate_submission() pre-flight check (NOT recommended)")
    p.add_argument("--quiet-ultralytics", action="store_true",
                   help="Reduce Ultralytics chatter (verbose=False)")
    p.add_argument("--hflip", action="store_true",
                   help="Predict on horizontally flipped images and map boxes back "
                        "to original coordinates (Tier 1A TTA). Uses per-image "
                        "OpenCV loads; ignores --batch > 1 for variable resolutions.")
    p.add_argument("--underwater-preproc", action="store_true",
                   help="Apply underwater preprocessing to each test image before "
                        "inference (Tier 1F / E4 / E9). Default method is MSRCR; "
                        "use --underwater-method clahe for the legacy gray-world+CLAHE "
                        "pipeline. Forces per-image loading via OpenCV. Must also "
                        "be applied at training time for consistent train/test distribution.")
    p.add_argument("--underwater-method", choices=["msrcr", "clahe"], default="msrcr",
                   help="Underwater preproc method when --underwater-preproc is set. "
                        "msrcr (default) — Multi-Scale Retinex w/ Color Restoration, "
                        "evidenced superior in Springer 2024 underwater detection lit. "
                        "clahe — original gray-world + CLAHE pipeline (E4 anchor).")
    p.add_argument("--max-images", type=int, default=None,
                   metavar="N",
                   help="Only the first N test images (sorted by basename). "
                        "Tier 1A smoke / debug — omit for full test-set CSV.")
    p.add_argument(
        "--per-class-conf-json",
        type=Path,
        default=None,
        help="Tier 1C-i: JSON from scripts/tune_per_class_conf.py — filter detections "
             "by per-COCO-category score thresholds before CSV export.",
    )
    return p.parse_args()


def _build_image_id_lookup(test_json_path: Path) -> dict[str, int]:
    """Build {filename: image_id} from test_dataset.json."""
    with test_json_path.open() as f:
        test = json.load(f)
    if "images" not in test:
        raise ValueError(f"{test_json_path} has no 'images' key (not a COCO JSON?)")
    lookup: dict[str, int] = {}
    for img in test["images"]:
        fn = img.get("file_name")
        iid = img.get("id")
        if fn is None or iid is None:
            continue
        lookup[fn] = int(iid)
        lookup[Path(fn).name] = int(iid)
    return lookup


def main() -> int:
    args = parse_args()

    if not args.weights.exists():
        print(f"ERROR: weights file not found: {args.weights}", file=sys.stderr)
        return 2
    if not args.test_images.exists():
        print(f"ERROR: test image dir not found: {args.test_images}", file=sys.stderr)
        return 2
    if not args.test_json.exists():
        print(f"ERROR: test JSON not found: {args.test_json}", file=sys.stderr)
        return 2
    if args.max_images is not None and args.max_images < 1:
        print("ERROR: --max-images must be >= 1", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FathomNet 2026 -- predict on test set + write Kaggle CSV")
    print("=" * 72)
    print(f"  Weights:     {args.weights}")
    print(f"  Test images: {args.test_images}")
    print(f"  Test JSON:   {args.test_json}")
    print(f"  Output CSV:  {args.out}")
    print(f"  imgsz:       {args.imgsz}")
    print(f"  batch:       {args.batch}")
    print(f"  conf:        {args.conf}")
    print(f"  iou:         {args.iou}")
    print(f"  max_det:     {args.max_det}")
    print(f"  TTA:         {args.tta}")
    print(f"  HFlip infer: {args.hflip}")
    mi = args.max_images
    print(f"  Max images:  {mi if mi is not None else 'all'}")
    print(f"  Half:        {args.half}")
    print(f"  Device:      {args.device or 'auto'}")
    conf_json = args.per_class_conf_json
    print(f"  Per-class τ: {conf_json if conf_json is not None else 'off'}")
    print(f"  UW preproc:  {args.underwater_preproc}")
    print()

    if conf_json is not None:
        if not conf_json.exists():
            print(f"ERROR: --per-class-conf-json not found: {conf_json}", file=sys.stderr)
            return 2

    print("[1/4] Building image_id_lookup from test_dataset.json ...")
    lookup = _build_image_id_lookup(args.test_json)
    print(f"      {len(lookup):,} entries (incl. basename aliases)")

    print("[2/4] Loading model and running inference ...")
    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    predict_kwargs_base = dict(
        imgsz=args.imgsz,
        batch=max(1, args.batch),
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device or None,
        augment=args.tta,
        half=args.half,
        verbose=not args.quiet_ultralytics,
        save=False,
        save_txt=False,
        save_conf=False,
        stream=True,
    )

    t_start = time.time()
    pred_count_per_img: list[int] = []
    image_count = 0
    from src.submit import (
        filter_pred_df_by_per_class_conf,
        from_ultralytics_results,
        load_per_class_conf_thresholds,
        unmirror_horizontal_xywh,
        yolo_preds_to_submission_csv,
        validate_submission,
    )
    import pandas as pd

    from src.inference_paths import capped_sorted_test_paths

    need_explicit_paths = args.hflip or args.underwater_preproc or args.max_images is not None
    paths_explicit: list[Path] | None = None
    if need_explicit_paths:
        paths_explicit = capped_sorted_test_paths(args.test_images, args.max_images)
        if not paths_explicit:
            print(f"ERROR: no images found in {args.test_images}", file=sys.stderr)
            return 2
        if args.max_images is not None:
            print(f"      [note] --max-images={args.max_images} subset ({len(paths_explicit)} paths)")

    uw_preproc_fn = None
    if args.underwater_preproc:
        if args.underwater_method == "msrcr":
            from src.underwater_preproc import underwater_preprocess_msrcr
            uw_preproc_fn = underwater_preprocess_msrcr
            print("      [note] --underwater-preproc enabled: MSRCR per image (Springer 2024)")
        else:
            from src.underwater_preproc import underwater_preprocess
            uw_preproc_fn = underwater_preprocess
            print("      [note] --underwater-preproc enabled: gray-world + CLAHE per image")

    rows_accum = []
    if args.hflip or (args.underwater_preproc and not args.hflip):
        try:
            import cv2
        except ImportError:
            print("ERROR: --hflip/--underwater-preproc requires OpenCV "
                  "(pip install opencv-python).", file=sys.stderr)
            return 2
        assert paths_explicit is not None
        paths = paths_explicit
        eff_batch = 1
        if args.batch > 1:
            print(f"      [note] per-image loading (batch=1, {len(paths):,} images); "
                  f"--batch {args.batch} ignored.")
        predict_kwargs_perimg = dict(
            predict_kwargs_base,
            batch=eff_batch,
            stream=True,
        )
        for img_path in paths:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"ERROR: cv2.imread failed for {img_path}", file=sys.stderr)
                return 2
            source = bgr
            if uw_preproc_fn is not None:
                source = uw_preproc_fn(source)
            if args.hflip:
                source = cv2.flip(source, 1)
            try:
                res = next(iter(model.predict(source=source, **predict_kwargs_perimg)))
            except StopIteration:
                print(f"ERROR: predict returned no results for {img_path}",
                      file=sys.stderr)
                return 2
            image_count += 1
            n_dets = 0 if res.boxes is None else len(res.boxes)
            pred_count_per_img.append(n_dets)
            if image_count % 100 == 0:
                elapsed = time.time() - t_start
                ips = image_count / max(elapsed, 1e-6)
                print(f"      ... {image_count:,} images processed ({ips:.1f} img/s)")
            df_one = from_ultralytics_results(
                [res], lookup, override_fnames=[img_path.name])
            if args.hflip:
                shape = getattr(res, "orig_shape", None)
                if shape is None or len(shape) < 2:
                    print(f"ERROR: invalid orig_shape for {img_path}: {shape!r}",
                          file=sys.stderr)
                    return 2
                ow = float(shape[1])
                df_one = unmirror_horizontal_xywh(df_one, ow)
            if not df_one.empty:
                rows_accum.append(df_one)
    elif paths_explicit is not None:
        predict_kwargs = dict(
            predict_kwargs_base,
            source=[str(p) for p in paths_explicit],
        )
        for res in model.predict(**predict_kwargs):
            image_count += 1
            n_dets = 0 if res.boxes is None else len(res.boxes)
            pred_count_per_img.append(n_dets)
            if image_count % 100 == 0:
                elapsed = time.time() - t_start
                ips = image_count / max(elapsed, 1e-6)
                print(f"      ... {image_count:,} images processed ({ips:.1f} img/s)")
            df_one = from_ultralytics_results([res], lookup)
            if not df_one.empty:
                rows_accum.append(df_one)
    else:
        predict_kwargs = dict(
            predict_kwargs_base,
            source=str(args.test_images),
        )
        for res in model.predict(**predict_kwargs):
            image_count += 1
            n_dets = 0 if res.boxes is None else len(res.boxes)
            pred_count_per_img.append(n_dets)
            if image_count % 100 == 0:
                elapsed = time.time() - t_start
                ips = image_count / max(elapsed, 1e-6)
                print(f"      ... {image_count:,} images processed ({ips:.1f} img/s)")
            df_one = from_ultralytics_results([res], lookup)
            if not df_one.empty:
                rows_accum.append(df_one)

    elapsed = time.time() - t_start
    print(f"      DONE: {image_count:,} images in {elapsed/60:.1f} min "
          f"({image_count/max(elapsed,1e-6):.1f} img/s)")
    if pred_count_per_img:
        import statistics
        print(f"      Detections per image: "
              f"mean={statistics.mean(pred_count_per_img):.1f}, "
              f"median={statistics.median(pred_count_per_img):.0f}, "
              f"max={max(pred_count_per_img)}")
    no_det_count = sum(1 for n in pred_count_per_img if n == 0)
    if no_det_count:
        print(f"      [warn] {no_det_count:,} images had ZERO detections "
              f"({100*no_det_count/max(image_count,1):.1f}%)")

    if not rows_accum:
        print("\nERROR: no detections produced. Check conf threshold or model weights.",
              file=sys.stderr)
        return 3
    pred_df = pd.concat(rows_accum, ignore_index=True)
    print(f"      Total detection rows (pre threshold filter): {len(pred_df):,}")

    if args.per_class_conf_json is not None:
        thr_map, default_thr = load_per_class_conf_thresholds(args.per_class_conf_json)
        if args.conf != default_thr:
            print(
                f"      [note] CLI --conf={args.conf} differs from JSON default_conf={default_thr}; "
                "per-class filter uses JSON defaults for missing keys.",
                flush=True,
            )
        pred_df = filter_pred_df_by_per_class_conf(
            pred_df, thr_map, default_threshold=default_thr
        )
        print(f"      Rows after per-class conf filter: {len(pred_df):,}")
        if pred_df.empty:
            print(
                "\nERROR: every detection was removed by per-class thresholds.",
                file=sys.stderr,
            )
            return 3

    print(f"[3/4] Writing CSV to {args.out} ...")
    out_path = yolo_preds_to_submission_csv(pred_df, args.out)
    csv_size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"      Wrote {csv_size_mb:.2f} MB CSV")

    if args.no_validate:
        print("[4/4] Skipping validate_submission() (--no-validate set)")
    else:
        print("[4/4] Running validate_submission() pre-flight check ...")
        warnings: list[str] = []
        report = validate_submission(
            out_path,
            test_json_path=args.test_json,
            warn_callback=lambda msg: warnings.append(msg),
        )
        n_rows = report.get('n_rows')
        n_rows_str = f"{n_rows:,}" if isinstance(n_rows, int) else "?"
        print(f"      [PASS] {n_rows_str} rows validated.")
        if warnings:
            print(f"      [warnings: {len(warnings)}]")
            for w in warnings[:5]:
                print(f"        - {w}")
            if len(warnings) > 5:
                print(f"        ... and {len(warnings)-5} more")

    print()
    if args.max_images is not None:
        print("[warn] Partial inference (--max-images): CSV is NOT the full test set; "
              "do not submit to Kaggle unless intentional.")
        print()
    print("=" * 72)
    print("READY FOR KAGGLE UPLOAD")
    print("=" * 72)
    print(f"CSV:          {out_path}")
    print(f"Submit cmd:   kaggle competitions submit -c fathomnet-2026 \\")
    print(f"                -f {out_path} \\")
    print(f"                -m \"<your message about this run>\"")
    print()
    print("REMINDER (R8): 10 submissions/day. Reserve slots for genuine probes;")
    print("use src.submit.local_score for sanity checks (free).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
