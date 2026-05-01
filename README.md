# FathomNet 2026

Positive-unlabeled object detection on deep-sea marine imagery. CLEF 2026 / FathomNet competition.

- **Competition:** https://kaggle.com/competitions/fathomnet-2026
- **Kaggle deadline:** May 7, 2026 (11:59 PM PT)
- **Working notes paper deadline:** May 28, 2026 (CLEF CEUR-WS)
- **Strategy:** see [`docs/fathomnet_2026_master_plan_v5.md`](docs/fathomnet_2026_master_plan_v5.md) (currently v5.5, AGGRESSIVE mode)
- **Day-by-day execution:** see [`docs/master_checklist.txt`](docs/master_checklist.txt)
- **Morning briefing for Day N:** see [`notes/overnight_summary.md`](notes/overnight_summary.md)

## Goal

Top-3 / plausibly top-1 finish via:

1. **Multi-pretrained-init bake-off** (Phase 2 EXP 2.3 — the differentiator) — 5 candidate inits compared, top 2 promoted.
2. **PU + class imbalance defenses** (Phase 3) — Kiryo PU loss + Equalized Focal Loss + Repeat Factor Sampling + Soft Teacher.
3. **Underwater-specific augmentation** (Phase 4) — gray-world + CLAHE preprocessing, class-aware copy-paste with per-class probability schedule.
4. **Ensemble + post-processing** (Phase 6-7) — 5-fold × 3 seeds yolo11l + RT-DETR-X + best-marine-init, fused via WBF, with SAHI for low-res test images and per-class NMS tuning.

Estimated final mAP@[.50:.95]: 0.29 - 0.50 (vs. current LB leader at 0.321).

## Repository structure

```
configs/                        # Config + reference data (cat_id mapping, taxonomy)
data/                           # Local-only (gitignored): images, labels, fold YAMLs
docs/
  fathomnet_2026_master_plan_v5.md   # Strategic guide (v5.5 AGGRESSIVE)
  master_checklist.txt               # Day-by-day execution checklist
  sample_submission.template.csv     # Submission CSV schema reference
  eval_notebook/map50-95.ipynb       # Kaggle's official scorer (vendored)
notebooks/
  phase0_eda.ipynb                   # Phase 0 dataset EDA (run end-to-end Apr 30)
  phase2_baseline.ipynb              # Phase 2 walkthrough + runnable cells
notes/
  external_models_post.md            # Kaggle Discussion post (declares pretrained models)
  phase1_bioclip2_research.md        # BioCLIP2 backbone integration design
  overnight_summary.md               # Morning briefing of last overnight session
scripts/
  download_images.py                 # Pulls dataset images from Kaggle (Phase 0)
  coco_to_yolo.py                    # COCO -> YOLO label converter (FLAG 8 fix)
  make_yolo_fold.py                  # Builds 5-fold YAMLs + image lists
  inspect_yolo_backbone.py           # Captures YOLOv11m P3/P4/P5 shapes
  download_marine_models.py          # Pulls Megalodon, MBARI 315k, Megafishdetector
  train_phase2_baseline.py           # Phase 2 training (Ultralytics + W&B + multi-init)
  eval_pretrained_inits.py           # The 5-way bake-off (Phase 2 EXP 2.3)
  predict_test_set.py                # End-to-end .pt -> Kaggle CSV
  wandb_smoke_test.py                # Verify W&B login and project access
src/
  bioclip2_yolo.py                   # BioCLIP2 backbone adapter (Phase 1 D.4)
  submit.py                          # Submission CSV writer + validator + local mAP
  efl_loss.py                        # Equalized Focal Loss (Phase 3 EXP 3.3)
  pu_loss.py                         # Kiryo PU loss (Phase 3 EXP 3.2)
  rfs_sampler.py                     # Repeat Factor Sampling (Phase 3 EXP 3.4)
  soft_teacher.py                    # Soft Teacher pseudo-labeling (Phase 3 EXP 3.5)
  underwater_preproc.py              # gray-world + CLAHE (Phase 4 EXP 4.2)
  copy_paste.py                      # Class-aware copy-paste (Phase 4 EXP 4.4)
tests/
  test_submit.py                     # 29 tests for src/submit.py
  test_efl_loss.py                   # 9 tests for src/efl_loss.py
  test_pu_loss.py                    # 13 tests for src/pu_loss.py
  test_rfs_sampler.py                # 11 tests for src/rfs_sampler.py
  test_soft_teacher.py               # 12 tests for src/soft_teacher.py
  test_underwater_preproc.py         # 14 tests for src/underwater_preproc.py
  test_copy_paste.py                 # 11 tests for src/copy_paste.py
weights/                        # Local-only (gitignored): model weights
submissions/                    # Local-only (gitignored): generated CSVs
```

## Status

### Completed

- [x] CLEF + Kaggle registration
- [x] Cursor + Python 3.11 venv + Kaggle API
- [x] Phase 0 EDA (`notebooks/phase0_eda.ipynb` — 7 artifacts)
- [x] Master plan v5.5 (AGGRESSIVE mode)
- [x] B.0.5 — Kaggle Discussion declaration of pretrained models
- [x] B.5 — Submission helpers (`src/submit.py` + 29 tests)
- [x] B.6 — COCO → YOLO label conversion
- [x] B.7 — 5-fold YAML generation
- [x] D.1-D.5 — BioCLIP2 backbone adapter (`src/bioclip2_yolo.py`) + smoke tests
- [x] Phase 2 production scripts (4 scripts, end-to-end)
- [x] Phase 3 modules (`pu_loss.py`, `efl_loss.py`, `rfs_sampler.py`, `soft_teacher.py`) — skeletons + tests
- [x] Phase 4 modules (`underwater_preproc.py`, `copy_paste.py`) — skeletons + tests

### Today's hard gates

- [ ] Edit Kaggle Discussion post to add Megalodon / MBARI 315k / Megafishdetector / GroundingDINO / OWL-ViT
- [ ] Deploy RunPod RTX 4090 pod (Block C.1)
- [ ] Run Phase 2 multi-init bake-off (~5 hr GPU, ~$3)
- [ ] Promote top 2 inits to full 50-epoch training (~5 hr GPU, ~$3)
- [ ] First Kaggle submission (vanilla baseline, anchors leaderboard)

## Quickstart (Windows / local laptop)

```powershell
.\venv\Scripts\Activate.ps1

# Verify auth + run Phase 0 EDA
kaggle competitions files fathomnet-2026
jupyter notebook notebooks/phase0_eda.ipynb

# Run all unit tests (require torch + opencv + ultralytics on the pod)
pytest tests/ -v

# Inspect the 5-way bake-off plan (does not execute)
jupyter notebook notebooks/phase2_baseline.ipynb
```

If `KAGGLE_API_TOKEN` is not set in this shell:

```powershell
[Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", "KGAT_your_token", "User")
```

Copy `.env.example` to `.env` and fill in W&B + HF tokens.

## Quickstart (RunPod RTX 4090 pod)

After `git clone` + `pip install -r requirements.txt`:

```bash
# Pull pretrained weights
python -c "from huggingface_hub import snapshot_download; snapshot_download('imageomics/bioclip-2', local_dir='weights/bioclip2')"
python scripts/download_marine_models.py

# Phase 2 vanilla baseline + first Kaggle submission
python scripts/train_phase2_baseline.py --init yolo11m.pt --epochs 50 --imgsz 640 --fold 0 \
    --name p2_baseline_yolo11m_coco
python scripts/predict_test_set.py \
    --weights weights/runs/p2_baseline_yolo11m_coco/weights/best.pt \
    --out submissions/p2_baseline.csv
kaggle competitions submit -c fathomnet-2026 -f submissions/p2_baseline.csv \
    -m "Phase 2 vanilla yolo11m+COCO baseline"

# THE differentiator: 5-way multi-pretrained-init bake-off
python scripts/eval_pretrained_inits.py --epochs 10 --fold 0 --imgsz 640
# Read notes/phase2_init_bakeoff.md for the ranking + decision rules

# Promote top 2 winners to full 50-epoch training, submit each
# (script will print the exact CLI for each)
```

## Budget

Single line item: RunPod GPU rental.

- Target: ~$210 across 7 days (May 1-7)
- Cap: $300
- Already paid: $100 credit on RunPod account
- Need to top up: ~$110-150 over the next week

Everything else (W&B, HF, Kaggle, GitHub) is free tier.

## License

Code: MIT (see `LICENSE`). Model weights and pretrained checkpoints are subject to their upstream licenses (Ultralytics AGPL-3.0, BioCLIP2 MIT, Megalodon CC-BY 4.0, etc).
