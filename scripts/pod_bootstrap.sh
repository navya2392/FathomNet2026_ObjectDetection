#!/usr/bin/env bash
# Pod bootstrap script for the Phase 2 5-way bake-off.
#
# Run this ONCE on a fresh RunPod RTX 4090 pod after `git clone`.
# Idempotent: safe to re-run if anything fails partway through.
#
# Usage:
#   chmod +x scripts/pod_bootstrap.sh
#   ./scripts/pod_bootstrap.sh
#
# Reads expected env vars:
#   WANDB_API_KEY         -- REQUIRED. From https://wandb.ai/authorize
#   KAGGLE_API_TOKEN      -- REQUIRED (bearer style, KGAT_...). Modern Kaggle auth.
#                            (Legacy fallback: set KAGGLE_USERNAME + KAGGLE_KEY instead.)
#   HUGGINGFACE_HUB_TOKEN -- OPTIONAL. hf_... read token; just removes rate-limit warnings.
#
# Set these BEFORE running the script:
#   export WANDB_API_KEY=wandb_v1_...
#   export KAGGLE_API_TOKEN=KGAT_...
#   export HUGGINGFACE_HUB_TOKEN=hf_...    # optional

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/fathomnet-2026}"
LOG_FILE="${REPO_ROOT}/logs/pod_bootstrap.$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/weights/marine_models"

step() {
    echo ""
    echo "============================================================================"
    echo "[$(date +%H:%M:%S)] $*"
    echo "============================================================================"
}

require_env() {
    local var="$1"
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: env var $var is not set. See script header for required vars."
        exit 2
    fi
}

cd "${REPO_ROOT}"
echo "Bootstrap starting in ${REPO_ROOT}"
echo "Logs streaming to ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

step "0/9  Verifying environment + GPU"
require_env WANDB_API_KEY
if [[ -z "${KAGGLE_API_TOKEN:-}" ]] && { [[ -z "${KAGGLE_USERNAME:-}" ]] || [[ -z "${KAGGLE_KEY:-}" ]]; }; then
    echo "ERROR: set either KAGGLE_API_TOKEN (bearer) OR (KAGGLE_USERNAME + KAGGLE_KEY)."
    exit 2
fi
nvidia-smi || { echo "ERROR: nvidia-smi failed -- no GPU?"; exit 2; }
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(f'torch={torch.__version__}  cuda={torch.version.cuda}  device={torch.cuda.get_device_name(0)}')"

step "1/9  Installing Python dependencies (~5 min)"
pip install --upgrade pip
pip install -r requirements.txt
echo "Pip install complete."

step "2/9  Configuring credentials"
if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
    echo "Using KAGGLE_API_TOKEN (bearer); the kaggle CLI picks this up automatically."
else
    mkdir -p ~/.kaggle
    cat > ~/.kaggle/kaggle.json <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
    chmod 600 ~/.kaggle/kaggle.json
    echo "Legacy kaggle.json written to ~/.kaggle/kaggle.json"
fi
wandb login --relogin "${WANDB_API_KEY}"
echo "W&B logged in."

step "3/9  Downloading Kaggle data files (~30 sec, ~9 MB)"
mkdir -p data/raw
if [[ ! -f data/raw/train_dataset.json ]] || [[ ! -f data/raw/test_dataset.json ]]; then
    cd data/raw
    kaggle competitions download -c fathomnet-2026 -p . --force
    unzip -o fathomnet-2026.zip
    cd "${REPO_ROOT}"
    echo "Kaggle data files unzipped."
else
    echo "Kaggle JSON files already present; skipping download."
fi

step "4/9  Downloading FathomNet images (~15-25 min, ~45 GB)"
mkdir -p data/raw/images/train data/raw/images/test
n_train=$(ls data/raw/images/train 2>/dev/null | wc -l)
n_test=$(ls data/raw/images/test 2>/dev/null | wc -l)
echo "Currently have: ${n_train} train images, ${n_test} test images"
DOWNLOADER="data/raw/download.py"
[[ -f "${DOWNLOADER}" ]] || DOWNLOADER="data/raw/kaggle/download.py"
if [[ "${n_train}" -lt 6000 ]]; then
    echo "Downloading train images..."
    python "${DOWNLOADER}" data/raw/train_dataset.json -o data/raw/images/train
fi
if [[ "${n_test}" -lt 1400 ]]; then
    echo "Downloading test images..."
    python "${DOWNLOADER}" data/raw/test_dataset.json -o data/raw/images/test
fi
echo "Image download complete: $(ls data/raw/images/train | wc -l) train, $(ls data/raw/images/test | wc -l) test"

step "5/9  Downloading BioCLIP2 weights (~3 min, ~3.3 GB)"
if [[ ! -d weights/bioclip2 ]] || [[ -z "$(ls weights/bioclip2 2>/dev/null)" ]]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('imageomics/bioclip-2', local_dir='weights/bioclip2', token='${HUGGINGFACE_HUB_TOKEN:-}')
print('BioCLIP2 download complete.')
"
else
    echo "BioCLIP2 weights already present; skipping."
fi

step "6/9  Downloading marine pretrained models (~2 min, ~1 GB)"
python scripts/download_marine_models.py

step "7/9  Building YOLO labels from COCO JSON (~30 sec, CPU)"
n_labels=$(ls data/raw/labels/train 2>/dev/null | wc -l)
if [[ "${n_labels}" -lt 6000 ]]; then
    python scripts/coco_to_yolo.py
    echo "YOLO labels built."
else
    echo "YOLO labels already built."
fi

step "8/9  Building 5-fold YAMLs (~10 sec, CPU)"
if [[ ! -f data/folds/fold0.yaml ]]; then
    python scripts/make_yolo_fold.py
    echo "Fold YAMLs built."
else
    echo "Fold YAMLs present (regenerating to ensure pod-correct paths)..."
    python scripts/make_yolo_fold.py
fi
head -8 data/folds/fold0.yaml

step "9/9  Pre-flight summary"
echo "Image counts:"
echo "  train: $(ls data/raw/images/train | wc -l)  (expected 6,463)"
echo "  test:  $(ls data/raw/images/test | wc -l)  (expected 1,425)"
echo ""
echo "YOLO labels: $(ls data/raw/labels/train | wc -l) .txt files"
echo ""
echo "Fold YAMLs:  $(ls data/folds/*.yaml 2>/dev/null | wc -l)"
echo ""
echo "Pretrained weights:"
for d in weights/bioclip2 weights/marine_models/megalodon weights/marine_models/mbari_315k weights/marine_models/megafishdetector; do
    if [[ -d "$d" ]]; then
        size=$(du -sh "$d" 2>/dev/null | cut -f1)
        echo "  ${d}: ${size}"
    else
        echo "  ${d}: MISSING"
    fi
done
echo ""
echo "============================================================================"
echo "BOOTSTRAP COMPLETE.  Next:"
echo ""
echo "  # 1. Smoke test (10 min, ~\$0.10)"
echo "  python scripts/train_phase2_baseline.py \\"
echo "      --init yolo11n.pt --epochs 10 --imgsz 640 --fold 0 --name p2_smoke_yolo11n"
echo ""
echo "  # 2. Vanilla baseline (~2.5 hr, ~\$1.50) -- this is the FIRST KAGGLE SUBMISSION"
echo "  python scripts/train_phase2_baseline.py \\"
echo "      --init yolo11m.pt --epochs 50 --imgsz 640 --fold 0 --name p2_baseline_yolo11m_coco"
echo ""
echo "  # 3. Generate vanilla baseline CSV"
echo "  python scripts/predict_test_set.py \\"
echo "      --weights weights/runs/p2_baseline_yolo11m_coco/weights/best.pt \\"
echo "      --out submissions/p2_baseline.csv"
echo ""
echo "  # 4. Submit to Kaggle (uses 1 of 10 daily slots)"
echo "  kaggle competitions submit -c fathomnet-2026 \\"
echo "      -f submissions/p2_baseline.csv \\"
echo "      -m 'Phase 2 vanilla yolo11m+COCO baseline (anchoring)'"
echo ""
echo "  # 5. THE differentiator (5 hr, ~\$3) -- 5-way bake-off"
echo "  python scripts/eval_pretrained_inits.py --epochs 10 --fold 0 --imgsz 640"
echo ""
echo "  # See notes/phase2_init_bakeoff.md for the ranking + decision."
echo "============================================================================"
