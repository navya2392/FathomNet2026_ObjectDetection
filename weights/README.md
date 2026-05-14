# Weights

Model weights are **not committed** to this repository (they exceed GitHub's file-size limits).

## Required weights

### YOLO11m base
The Ultralytics base weight is fetched automatically on first training run, or you can pre-download:

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt -O weights/yolo11m.pt
```

### MBARI 315k YOLOv8 (primary init)
The Phase 2 bake-off (see [`../README.md`](../README.md) → *What worked → MBARI 315k pretrained init*) showed MBARI 315k as the best init by a wide margin. Available from the FathomNet model zoo:

<https://huggingface.co/FathomNet/MBARI-315k-yolov8>

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('FathomNet/MBARI-315k-yolov8', local_dir='weights/mbari_315k')"
```

### RT-DETR-l (T2 ensemble component)
Used as the cross-architecture diversity member in the final E27 ensemble:

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt -O weights/rtdetr-l.pt
```

## Expected layout

```
weights/
├── yolo11m.pt
├── mbari_315k/
│   └── best.pt
├── rtdetr-l.pt
└── runs/                # Ultralytics writes training outputs here (gitignored)
```
