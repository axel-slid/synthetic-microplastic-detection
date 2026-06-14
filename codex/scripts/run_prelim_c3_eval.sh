#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/shared/dils/anaconda3/bin/python}"
GPU="${GPU:-1}"
CONFIG="${CONFIG:-configs/prelim_c3_eval.yaml}"
RUN_ID="${RUN_ID:-prelim_c2_sdxl_inpaint__smp_unet_resnet34__seed13}"
EPOCHS="${EPOCHS:-8}"
LOG_DIR="${LOG_DIR:-results/prelim_c3_eval/logs/prelim_c3_sdxl_unet_seed13_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="results/prelim_c3_eval/runs/${RUN_ID}"
MANIFEST="results/prelim_c3_eval/manifests/c3_clean.csv"

mkdir -p "$LOG_DIR"
echo "$$" > "$LOG_DIR/pipeline.pid"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_DIR/status.log"
}

log "preliminary training start run_id=${RUN_ID} gpu=${GPU} epochs=${EPOCHS}"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON" scripts/train_segmenter.py \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --epochs "$EPOCHS" \
  >"$LOG_DIR/train.log" 2>&1

log "c3 evaluation start"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON" scripts/evaluate_checkpoint.py \
  --config "$CONFIG" \
  --checkpoint "$OUT_DIR/best.pt" \
  --manifest "$MANIFEST" \
  --out "$OUT_DIR/c3_clean_metrics.csv" \
  >"$LOG_DIR/evaluate_c3.log" 2>&1

log "preliminary evaluation done"
