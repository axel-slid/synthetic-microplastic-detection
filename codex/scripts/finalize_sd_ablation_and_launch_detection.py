#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def log(message: str, path: Path) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def registry_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(pd.read_csv(path).drop_duplicates("run_id", keep="last"))


def missing_run_ids(registry_path: Path, matrix_path: Path) -> set[str]:
    planned = set(pd.read_csv(matrix_path)["run_id"].astype(str))
    if not registry_path.exists():
        return planned
    registered = set(pd.read_csv(registry_path).drop_duplicates("run_id", keep="last")["run_id"].astype(str))
    return planned - registered


def run_command(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        result = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for SD ablation segmentation, evaluate it, then launch detection ablation."
    )
    parser.add_argument("--expected-runs", type=int, default=105)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--detection-gpus", default="1:1")
    parser.add_argument("--detection-epochs", type=int, default=80)
    parser.add_argument("--detection-batch-size", type=int, default=8)
    parser.add_argument("--run-matrix", default="results/manifests/sd_ablation/run_matrix.csv")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "results" / "logs" / f"sd_ablation_finalize_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    main_log = log_dir / "finalize.log"
    registry = ROOT / "results" / "reports" / "sd_ablation" / "trained_checkpoints.csv"
    run_matrix = ROOT / args.run_matrix

    while True:
        count = registry_count(registry)
        missing = missing_run_ids(registry, run_matrix)
        log(f"sd_ablation_registered={count}/{args.expected_runs} missing_planned={len(missing)}", main_log)
        if count >= args.expected_runs and not missing:
            break
        time.sleep(args.poll_seconds)

    run_command(
        [
            "python",
            "scripts/evaluate_registry.py",
            "--config",
            "configs/sd_ablation.yaml",
            "--registry",
            "results/reports/sd_ablation/trained_checkpoints.csv",
            "--manifest",
            "results/manifests/c3_clean.csv",
            "--skip-existing",
        ],
        log_dir / "evaluate_sd_ablation.log",
    )
    run_command(
        [
            "python",
            "scripts/aggregate_results.py",
            "--results-dir",
            "results/runs/sd_ablation",
            "--out",
            "results/reports/sd_ablation/aggregate_c3_clean.csv",
        ],
        log_dir / "aggregate_sd_ablation.log",
    )
    run_command(
        [
            "python",
            "scripts/launch_detection_ablation.py",
            "--run-matrix",
            "results/manifests/detection_ablation/run_matrix.csv",
            "--gpus",
            args.detection_gpus,
            "--epochs",
            str(args.detection_epochs),
            "--batch-size",
            str(args.detection_batch_size),
        ],
        log_dir / "launch_detection_ablation.log",
    )
    log("complete", main_log)


if __name__ == "__main__":
    main()
