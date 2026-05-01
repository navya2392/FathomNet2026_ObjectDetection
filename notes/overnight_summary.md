# Overnight summary — Apr 30 to May 1

> Read this first. Everything below is uncommitted, sitting in your
> working tree, ready for review. Nothing was pushed to GitHub.

## What landed (8 deliverables)

### Code (3 files + 1 modified)

| File | What it does | LOC | Verified |
|---|---|---|---|
| `src/submit.py` | Submission CSV writer + validator + local mAP scorer (B.5) | 440 | yes — 29 tests pass |
| `tests/test_submit.py` | Unit tests for everything in `src/submit.py` | 230 | yes — 29/29 |
| `tests/__init__.py` | Empty marker | 0 | n/a |
| `scripts/coco_to_yolo.py` | COCO → YOLO label converter, bypasses Ultralytics' broken `convert_coco()` (B.6 / FLAG 8) | 165 | yes — dry-run on 6,463 imgs, 0 errors |
| `scripts/make_yolo_fold.py` | Builds 5-fold YAMLs + image lists for YOLO training (B.7) | 130 | yes — generated all 15 files |
| `requirements.txt` (modified) | Added `pytest==9.0.3` (was missing) | +1 | yes |

### Docs / templates (2 files)

- `docs/sample_submission.template.csv` — 3-row schema reference (image_ids 1, 2, 3; category_ids 1, 13, 41 to exercise gap-handling)
- `docs/sample_submission.template.md` — explanation of the schema, why we built it, how to use it

### Generated data artifacts (15 files in `data/folds/`)

Built by running `make_yolo_fold.py`:

- `fold{0..4}_train.txt` — image paths, ~5,170 lines each
- `fold{0..4}_val.txt`   — image paths, ~1,293 lines each
- `fold{0..4}.yaml`      — Ultralytics-format dataset config

Note: `data/folds/` is in `.gitignore`. These files are local-only and can be regenerated any time by running `python scripts/make_yolo_fold.py`. Do NOT commit them.

### Notes (3 files)

- `notes/external_models_post.md` — DRAFT post for the Kaggle Discussion board declaring YOLOv11 + BioCLIP2 as external pretrained models (covers B.0.5 hard gate). **Read, edit your name into the body, post before any leaderboard submission.**
- `notes/phase1_bioclip2_research.md` — design doc for plugging BioCLIP2 into YOLOv11. Covers what BioCLIP2 actually is at the tensor level, what YOLO needs from a backbone, three integration options (frozen-features+adapter recommended), and concrete next-step instructions for D.1-D.5.
- `notes/overnight_summary.md` — this file.

## Test results (verbatim)

```
============================= 29 passed in 2.51s ==============================
```

All boundary cases pass:

- `class_idx=0  -> category_id=1` (low boundary)
- `class_idx=11 -> category_id=13` (skips the gap at 12 — FLAG 1 protection)
- `class_idx=31 -> category_id=41` (high boundary)
- `category_id=12` rejected (gap value)
- `category_id=0` rejected (the "submitted indices instead of cat_ids" bug)
- `category_id=42` rejected (out-of-range)
- `score=0.0` and `score=1.0` accepted (boundary)
- `score<0` and `score>1` rejected
- `bbox_w=0` and `bbox_h<0` rejected (strict > 0)
- NaN / Inf in any numeric column rejected
- 110-detections-per-image emits warning but accepts (pycocotools maxDets=100)
- **Round-trip test: feeding GT-as-predictions through `local_score()` returns mAP=1.0** ← this is the strongest end-to-end check

## Dataset facts confirmed overnight

- Train images: **6,463** in `data/raw/images/train/` (mixed `.png` + `.jpg`)
- Test images: **1,425** in `data/raw/images/test/` (all `.png`)
- Total annotations: **22,225** (avg ~3.4 per image)
- **0 train images have zero annotations** — every image has at least one labeled box
- All 32 categories from `dataset_train.json` map cleanly to `class_idx` 0-31
- Image dimensions all present in the JSON, no degenerate bboxes (`w<=0` or `h<=0`), no out-of-bounds boxes

## What you should do (in order)

### Right now (10 min)

1. **Review the diffs.** `git status` shows 5 untracked files + 1 modified.
   ```powershell
   git diff requirements.txt
   git status
   ```
2. **Run the tests yourself** to verify in your shell:
   ```powershell
   .\venv\Scripts\python.exe -m pytest tests/test_submit.py -v
   ```
   Expect: `29 passed`.
3. **Skim `notes/overnight_summary.md`** (this file) and `notes/phase1_bioclip2_research.md`. The other notes file you can read when needed.

### Today (~30 min total)

4. **Commit if happy** (suggested split into 3 commits for a clean history):
   ```powershell
   # Commit 1: B.5 + tests
   git add src/submit.py tests/ requirements.txt
   git commit -m "B.5: submission helpers + 29-test suite (pytest added)"

   # Commit 2: B.6 + B.7
   git add scripts/coco_to_yolo.py scripts/make_yolo_fold.py
   git commit -m "B.6/B.7: COCO->YOLO converter + per-fold YAML builder"

   # Commit 3: docs + notes
   git add docs/sample_submission.template.* notes/
   git commit -m "Add submission template doc + Phase-0/1 notes"

   git push
   ```
5. **Generate labels for real** (~6 sec, ~10 MB on disk in `.gitignore`d folder):
   ```powershell
   .\venv\Scripts\python.exe scripts/coco_to_yolo.py
   ```
   Output goes to `data/raw/labels/train/`. After this, fold YAMLs are training-ready.
6. **Verify Ultralytics can load fold 0:**
   ```powershell
   .\venv\Scripts\python.exe -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset(r'data\folds\fold0.yaml')"
   ```
   Expect a stats summary, no errors. If it complains about missing labels, check that step 5 ran successfully.

### Today, when ready

7. ~~**A.9 — RunPod account + $100 credit**~~ ✅ **DONE late Apr 30**: account created, $100 credit added, SSH key (`runpod_ed25519`) generated and uploaded to RunPod Settings. Just needs to be marked `[x] done` in `master_checklist.txt`.
8. **B.0.5 — Post the external-models declaration** (~5 min): edit `notes/external_models_post.md` to put your name in, post to the Kaggle Discussion board, paste the URL into `master_checklist.txt` next to B.0.5, mark B.0.5 done. **HARD GATE: do not click Submit on Kaggle until this is posted.**
9. **D.1 — Download BioCLIP2 weights** (~15 min, ~2.5 GB). See `notes/phase1_bioclip2_research.md` section 7 step 1.

## What's NOT done that I had on the list

- **`README.md` update** — I wanted to refresh it to reflect v5.3 plan, but I deferred to keep overnight scope manageable. Lower priority than getting code in working order. Drop me a line tomorrow if you want me to do it.

## Heads-up notes

- **`coco_to_yolo.py` default output path was changed** mid-night from `data/labels/` to `data/raw/labels/train/`. This is so Ultralytics' implicit `/images/` ↔ `/labels/` path swap works cleanly with the `data/raw/images/{train,test}/` structure. Already verified with a fresh dry-run.
- **`pytest` was missing from `requirements.txt`.** Added it (v9.0.3). If you regenerate the requirements with `pip freeze > requirements.txt` later, it'll stick.
- **Nothing was committed.** Every change is reviewable. If you don't like something, `git checkout -- <file>` reverts it cleanly.

## Open questions for you (low urgency)

- **YOLOv11 size:** `yolo11m` (50M params) vs `yolo11l` (86M) vs `yolo11x` (133M)? My recommendation in the BioCLIP2 doc assumed `yolo11m` for speed. We can revisit on RunPod once we know GPU.
- **First Kaggle submission strategy:** vanilla YOLOv11m baseline first (to anchor the leaderboard), or skip straight to BioCLIP2-augmented? Anchoring the leaderboard first gives us a sanity-check score we can compare every later experiment against. Recommended.
