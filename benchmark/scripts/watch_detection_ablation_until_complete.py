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


def count_registry(path: Path) -> int:
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
    parser = argparse.ArgumentParser(description="Wait for detection ablation, then evaluate all detectors.")
    parser.add_argument("--expected-runs", type=int, default=15)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--run-matrix", default="results/manifests/detection_ablation/run_matrix.csv")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "results" / "logs" / f"detection_ablation_finalize_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    main_log = log_dir / "finalize.log"
    registry = ROOT / "results" / "reports" / "detection_ablation" / "trained_checkpoints.csv"
    run_matrix = ROOT / args.run_matrix

    while True:
        count = count_registry(registry)
        missing = missing_run_ids(registry, run_matrix)
        log(f"detection_registered={count}/{args.expected_runs} missing_planned={len(missing)}", main_log)
        if count >= args.expected_runs and not missing:
            break
        time.sleep(args.poll_seconds)

    run_command(
        [
            "python",
            "scripts/evaluate_detection_ablation_registry.py",
            "--registry",
            "results/reports/detection_ablation/trained_checkpoints.csv",
            "--data",
            "results/detection_eval/c3_clean/dataset.yaml",
            "--skip-existing",
        ],
        log_dir / "evaluate_detection.log",
    )
    run_command(
        [
            "python",
            "scripts/update_paper_with_results.py",
            "--aggregate",
            "results/reports/aggregate_c3_clean.csv",
            "--expected-runs",
            "180",
            "--sd-aggregate",
            "results/reports/sd_ablation/aggregate_c3_clean.csv",
            "--sd-expected-runs",
            "105",
            "--detection-aggregate",
            "results/reports/detection_ablation/aggregate_c3_clean.csv",
            "--detection-expected-runs",
            "15",
        ],
        log_dir / "update_paper.log",
    )
    log("complete", main_log)


if __name__ == "__main__":
    main()
