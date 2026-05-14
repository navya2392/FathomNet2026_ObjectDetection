# Setup

## Prerequisites

- Python 3.11
- CUDA-capable GPU (24 GB+ VRAM for training at imgsz=1024; CPU is fine for the demo notebook on a small sample)
- ~50 GB free disk (dataset + weights + checkpoints)
- A Kaggle account with the [FathomNet 2026 competition](https://www.kaggle.com/competitions/fathomnet-2026) accepted (to download data)

## Install

```bash
git clone https://github.com/<your-user>/fathomnet-2026.git
cd fathomnet-2026

python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows PowerShell
.\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in W&B + HuggingFace tokens if you want experiment tracking and gated model downloads.

## Download data and weights

Data and weights are not committed. Follow the per-folder instructions:

- [`data/README.md`](data/README.md) — Kaggle CLI download + preprocessing
- [`weights/README.md`](weights/README.md) — YOLO11, MBARI 315k, RT-DETR-l

## Run the demo notebook

The fastest way to see what the project produces:

```bash
jupyter notebook notebooks/final_pipeline.ipynb
```

The notebook is committed **with cached outputs**, so you can browse the rendered results directly on GitHub without running it locally.

## Run training

The anchor model (E1, MBARI-315k-initialized YOLOv8x at imgsz=1024 for 50 epochs):

```bash
python scripts/train_phase2_baseline.py \
    --init weights/mbari_315k/best.pt \
    --epochs 50 \
    --imgsz 1024 \
    --fold 0 \
    --name p2_aug_e50_1024_fold0
```

The cross-architecture ensemble member (T2, RT-DETR-l):

```bash
python scripts/train_rtdetr.py --epochs 50 --imgsz 1024 --fold 0 --name t2_rtdetr_l_fold0
```

## Run inference

6-scale TTA + WBF on either model:

```bash
python scripts/predict_test_multiscale_tta.py \
    --weights weights/runs/p2_aug_e50_1024_fold0/weights/best.pt \
    --scales 480 640 832 1024 1280 1536 \
    --out submissions/e1.csv
```

Cross-architecture ensemble (final E27 = WBF of E1 + T2):

```bash
python scripts/cross_arch_ensemble.py \
    --submissions submissions/e1.csv submissions/t2.csv \
    --weights 1.0 1.0 \
    --iou-thr 0.55 \
    --out submissions/e27.csv
```

## Run tests

```bash
pytest tests/ -v
```

Tests cover the `src/` modules (PU loss, EFL, RFS sampler, copy-paste, soft teacher, underwater preprocessing, submission utilities, hierarchical loss, inference paths). They run without GPU or competition data.
