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


def registry_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path).drop_duplicates("run_id", keep="last")["run_id"].astype(str))


def missing_run_ids(registry_path: Path, matrix_path: Path) -> set[str]:
    planned = set(pd.read_csv(matrix_path)["run_id"].astype(str))
    return planned - registry_run_ids(registry_path)


def run_command(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        result = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for gpu1_sd_gan segmentation, evaluate final checkpoints, and summarize verified-generator results."
    )
    parser.add_argument("--expected-runs", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--run-matrix", default="results/manifests/gpu1_sd_gan/run_matrix.csv")
    parser.add_argument("--registry", default="results/reports/gpu1_sd_gan/trained_checkpoints.csv")
    parser.add_argument("--manifest", default="results/manifests/gpu1_sd_gan/c3_clean.csv")
    parser.add_argument("--config", default="configs/gpu1_sd_gan.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "results" / "logs" / f"gpu1_sd_gan_finalize_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    main_log = log_dir / "finalize.log"
    registry = ROOT / args.registry
    run_matrix = ROOT / args.run_matrix

    while True:
        registered = registry_run_ids(registry)
        missing = missing_run_ids(registry, run_matrix)
        log(
            f"gpu1_sd_gan_registered={len(registered)}/{args.expected_runs} missing_planned={len(missing)}",
            main_log,
        )
        if len(registered) >= args.expected_runs and not missing:
            break
        time.sleep(args.poll_seconds)

    run_command(
        [
            "python",
            "scripts/evaluate_registry.py",
            "--config",
            args.config,
            "--registry",
            args.registry,
            "--manifest",
            args.manifest,
            "--device",
            args.device,
        ],
        log_dir / "evaluate_gpu1_sd_gan.log",
    )
    run_command(
        [
            "python",
            "scripts/summarize_gpu1_sd_gan_results.py",
            "--run-root",
            "results/runs/gpu1_sd_gan",
            "--out-dir",
            "results/reports/gpu1_sd_gan",
        ],
        log_dir / "summarize_gpu1_sd_gan.log",
    )
    warning = ROOT / "results" / "reports" / "gpu1_sd_gan" / "INTERIM_METRICS_NOT_FOR_MANUSCRIPT.md"
    if warning.exists():
        warning.rename(ROOT / "results" / "reports" / "gpu1_sd_gan" / "INTERIM_METRICS_SUPERSEDED_BY_FINAL.md")
    log("complete", main_log)


if __name__ == "__main__":
    main()
