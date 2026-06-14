#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/mnt/shared/dils/anaconda3/bin/python}"
EVAL_GPU="${EVAL_GPU:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
LOG_DIR="$ROOT/results/logs/c2_refresh"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/refresh_${STAMP}.log"

mkdir -p "$LOG_DIR"
cd "$ROOT"

{
  echo "refresh started $(date -Is)"
  echo "python=$PYTHON eval_gpu=$EVAL_GPU eval_device=$EVAL_DEVICE"

  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/evaluate_registry.py \
    --config configs/c2_inpainting_5x5.yaml \
    --registry results/reports/c2_inpainting_5x5/trained_checkpoints.csv \
    --manifest results/manifests/c2_inpainting_5x5/c3_clean.csv \
    --device "$EVAL_DEVICE" \
    --skip-existing

  "$PYTHON" scripts/make_current_performance_report.py
  "$PYTHON" scripts/sync_c2_results_to_manuscript.py

  (cd "$ROOT/../overleaf_microplastic_project" && latexmk -pdf -interaction=nonstopmode main.tex)
  "$PYTHON" scripts/verify_paper.py
  echo "refresh finished $(date -Is)"
} | tee "$RUN_LOG"

echo "$RUN_LOG"
