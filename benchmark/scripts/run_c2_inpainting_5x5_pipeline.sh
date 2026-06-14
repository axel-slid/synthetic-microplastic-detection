#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/mnt/shared/dils/anaconda3/bin/python}"
CONFIG="${CONFIG:-configs/c2_inpainting_5x5.yaml}"
COUNT="${COUNT:-10000}"
SEED="${SEED:-13}"
TRAIN_GPUS="${TRAIN_GPUS:-1:6,2:3,0:1}"
OVERWRITE_GENERATION="${OVERWRITE_GENERATION:-0}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-results/logs/c2_inpainting_5x5_${STAMP}}"
RUN_MATRIX="${RUN_MATRIX:-results/manifests/c2_inpainting_5x5/run_matrix.csv}"
RUN_COMMANDS="${RUN_COMMANDS:-results/manifests/c2_inpainting_5x5/run_matrix.sh}"
REGISTRY="${REGISTRY:-results/reports/c2_inpainting_5x5/trained_checkpoints.csv}"
C3_MANIFEST="${C3_MANIFEST:-results/manifests/c2_inpainting_5x5/c3_clean.csv}"

mkdir -p "$LOG_DIR"
echo "$$" > "$LOG_DIR/pipeline.pid"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_DIR/pipeline.log"
}

run_generation() {
  local method="$1"
  local gpu="$2"
  local log_path="$LOG_DIR/generate_${method}.log"
  local overwrite_args=()
  if [[ "$OVERWRITE_GENERATION" == "1" ]]; then
    overwrite_args+=(--overwrite)
  fi
  log "generation start method=${method} gpu=${gpu} count=${COUNT}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON" scripts/generate_synthetic.py \
    --config "$CONFIG" \
    --method "$method" \
    --count "$COUNT" \
    --seed "$SEED" \
    --device cuda:0 \
    "${overwrite_args[@]}" \
    >"$log_path" 2>&1
  log "generation finished method=${method}"
}

wait_for_group() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    log "one or more generation jobs failed"
    return 1
  fi
}

log "pipeline start config=${CONFIG}"
log "logs=${LOG_DIR}"

run_generation c2_sdxl_inpaint 1 &
pid_sdxl=$!
run_generation c2_flux_inpaint 2 &
pid_flux=$!
run_generation c2_sd2_fiber_inpaint 0 &
pid_sd2_fiber=$!
wait_for_group "$pid_sdxl" "$pid_flux" "$pid_sd2_fiber"

run_generation c2_sd2_inpaint 0 &
pid_sd2=$!
run_generation c2_sdxl_texture_inpaint 1 &
pid_sdxl_texture=$!
wait_for_group "$pid_sd2" "$pid_sdxl_texture"

log "writing manifests"
"$PYTHON" scripts/prepare_manifests.py --config "$CONFIG" --require-generated \
  >"$LOG_DIR/prepare_manifests.log" 2>&1

log "planning training runs"
"$PYTHON" scripts/plan_runs.py \
  --config "$CONFIG" \
  --out "$RUN_MATRIX" \
  --commands "$RUN_COMMANDS" \
  --available-only \
  >"$LOG_DIR/plan_runs.log" 2>&1

log "training start gpus=${TRAIN_GPUS}"
PYTHONUNBUFFERED=1 "$PYTHON" scripts/launch_benchmark.py \
  --config "$CONFIG" \
  --run-matrix "$RUN_MATRIX" \
  --gpus "$TRAIN_GPUS" \
  --family semantic \
  >"$LOG_DIR/train_launcher.log" 2>&1
log "training finished"

log "c3 evaluation start"
PYTHONUNBUFFERED=1 "$PYTHON" scripts/evaluate_registry.py \
  --config "$CONFIG" \
  --registry "$REGISTRY" \
  --manifest "$C3_MANIFEST" \
  >"$LOG_DIR/evaluate_c3.log" 2>&1
log "c3 evaluation finished"

log "pipeline complete"
