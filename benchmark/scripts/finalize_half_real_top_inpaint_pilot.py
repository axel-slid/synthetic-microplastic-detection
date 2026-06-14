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


def registered(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path).drop_duplicates("run_id", keep="last")["run_id"].astype(str))


def planned(path: Path) -> set[str]:
    return set(pd.read_csv(path)["run_id"].astype(str))


def run_command(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        result = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--config", default="configs/half_real_top_inpaint_pilot.yaml")
    parser.add_argument("--run-matrix", default="results/manifests/half_real_top_inpaint_pilot/run_matrix.csv")
    parser.add_argument("--registry", default="results/reports/half_real_top_inpaint_pilot/trained_checkpoints.csv")
    parser.add_argument("--manifest", default="results/manifests/half_real_top_inpaint_pilot/c3_clean.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "results" / "logs" / f"half_real_top_inpaint_pilot_finalize_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    main_log = log_dir / "finalize.log"
    registry = ROOT / args.registry
    matrix = ROOT / args.run_matrix
    target = planned(matrix)
    while True:
        have = registered(registry)
        missing = target - have
        log(f"registered={len(have)}/{args.expected_runs} missing={len(missing)}", main_log)
        if len(have) >= args.expected_runs and not missing:
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
        log_dir / "evaluate.log",
    )
    run_command(
        [
            "python",
            "scripts/summarize_half_real_top_inpaint_pilot.py",
            "--run-root",
            "results/runs/half_real_top_inpaint_pilot",
            "--out-dir",
            "results/reports/half_real_top_inpaint_pilot",
        ],
        log_dir / "summarize.log",
    )
    log("complete", main_log)


if __name__ == "__main__":
    main()
