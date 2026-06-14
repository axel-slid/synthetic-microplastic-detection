#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/mnt/shared/dils/anaconda3/bin/python}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
TARGET_RUNS="${TARGET_RUNS:-75}"
EVAL_GPU="${EVAL_GPU:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
LOG_DIR="$ROOT/results/logs/c2_refresh_watch"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/watch_${STAMP}.log"

mkdir -p "$LOG_DIR"
cd "$ROOT"

count_registry() {
  "$PYTHON" - <<'PY'
import pandas as pd
from pathlib import Path
p = Path("results/reports/c2_inpainting_5x5/trained_checkpoints.csv")
print(0 if not p.exists() else len(pd.read_csv(p).drop_duplicates("run_id", keep="last")))
PY
}

count_summaries() {
  find results/runs/c2_inpainting_5x5 -maxdepth 2 -name c3_clean_metrics.summary.json | wc -l
}

{
  echo "watch started $(date -Is)"
  echo "target_runs=$TARGET_RUNS interval_seconds=$INTERVAL_SECONDS eval_gpu=$EVAL_GPU eval_device=$EVAL_DEVICE"
  while true; do
    registry="$(count_registry)"
    summaries="$(count_summaries)"
    echo "status $(date -Is) registry=$registry summaries=$summaries"

    EVAL_GPU="$EVAL_GPU" EVAL_DEVICE="$EVAL_DEVICE" "$ROOT/scripts/refresh_c2_outputs.sh"

    registry="$(count_registry)"
    summaries="$(count_summaries)"
    echo "after-refresh $(date -Is) registry=$registry summaries=$summaries"
    if [[ "$registry" -ge "$TARGET_RUNS" && "$summaries" -ge "$TARGET_RUNS" ]]; then
      echo "watch complete $(date -Is)"
      break
    fi
    sleep "$INTERVAL_SECONDS"
  done
} >> "$LOG" 2>&1

echo "$LOG"
