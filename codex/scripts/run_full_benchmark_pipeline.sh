#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

stamp="$(date +%Y%m%d_%H%M%S)"
log_root="results/logs/full_benchmark_${stamp}"
mkdir -p "${log_root}"

echo "Full benchmark started at $(date -Is)" | tee "${log_root}/pipeline.log"
echo "Semantic jobs: GPU 1, 3 workers" | tee -a "${log_root}/pipeline.log"
echo "YOLO jobs: GPU 0, 1 worker" | tee -a "${log_root}/pipeline.log"

python scripts/launch_benchmark.py --family semantic --gpus 1:3 \
  > "${log_root}/semantic_launcher.log" 2>&1 &
semantic_pid=$!

python scripts/launch_benchmark.py --family yolo --gpus 0:1 \
  > "${log_root}/yolo_launcher.log" 2>&1 &
yolo_pid=$!

wait "${semantic_pid}"
wait "${yolo_pid}"

python - <<'PY'
import pandas as pd
from pathlib import Path

registry = Path("results/reports/trained_checkpoints.csv")
if not registry.exists():
    raise SystemExit("No trained checkpoint registry was produced.")
frame = pd.read_csv(registry).drop_duplicates("run_id", keep="last")
if len(frame) < 105:
    raise SystemExit(f"Expected 105 completed runs before paper update, found {len(frame)}")
print(f"Completed runs: {len(frame)}")
PY

CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_registry.py --config configs/benchmark.yaml \
  > "${log_root}/evaluate_registry.log" 2>&1
python scripts/aggregate_results.py > "${log_root}/aggregate_results.log" 2>&1
python scripts/update_paper_with_results.py --expected-runs 105 \
  > "${log_root}/update_paper_with_results.log" 2>&1
python scripts/make_paper_assets.py > "${log_root}/make_paper_assets.log" 2>&1

(cd .. && latexmk -pdf -interaction=nonstopmode paper.tex) \
  > "${log_root}/latexmk.log" 2>&1
python scripts/verify_paper.py > "${log_root}/verify_paper.log" 2>&1

echo "Full benchmark and paper build completed at $(date -Is)" | tee -a "${log_root}/pipeline.log"
