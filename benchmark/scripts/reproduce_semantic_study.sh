#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/benchmark.yaml}"
RUN_MATRIX="${2:-results/manifests/run_matrix.csv}"

python scripts/validate_data.py --config "$CONFIG"
python scripts/prepare_manifests.py --config "$CONFIG"
python scripts/plan_runs.py --config "$CONFIG" --out "$RUN_MATRIX" --available-only
python scripts/launch_benchmark.py \
  --config "$CONFIG" \
  --run-matrix "$RUN_MATRIX" \
  --family semantic
python scripts/evaluate_registry.py --config "$CONFIG"
python scripts/aggregate_results.py
python scripts/make_paper_assets.py
python scripts/make_journal_paper_assets.py
