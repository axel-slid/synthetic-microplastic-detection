#!/usr/bin/env python
from __future__ import annotations

import argparse
import fcntl
import re
import subprocess
import sys
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


def registered_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path).drop_duplicates("run_id", keep="last")["run_id"].astype(str))


def missing_run_ids(registry_path: Path, matrix_path: Path) -> set[str]:
    matrix = pd.read_csv(matrix_path)
    planned = set(matrix[matrix["family"].astype(str) == "semantic"]["run_id"].astype(str))
    return planned - registered_run_ids(registry_path)


def process_lines() -> list[str]:
    try:
        return subprocess.check_output(["ps", "-eo", "args"], text=True).splitlines()
    except subprocess.SubprocessError:
        return []


def active_train_run_ids(lines: list[str]) -> set[str]:
    pattern = re.compile(r"scripts/train_segmenter\.py .*?--config configs/gpu1_sd_gan\.yaml .*?--run-id ([^ ]+)")
    active: set[str] = set()
    for line in lines:
        match = pattern.search(line)
        if match:
            active.add(match.group(1))
    return active


def active_launcher_count(lines: list[str]) -> int:
    return sum(
        1
        for line in lines
        if "scripts/launch_benchmark.py" in line
        and "--config configs/gpu1_sd_gan.yaml" in line
        and "--family semantic" in line
    )


def launch(args: argparse.Namespace, log_dir: Path, main_log: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/launch_benchmark.py",
        "--config",
        args.config,
        "--run-matrix",
        args.run_matrix,
        "--registry",
        args.registry,
        "--gpus",
        args.gpus,
        "--family",
        "semantic",
    ]
    child_log = log_dir / f"resume_semantic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.out"
    with child_log.open("w", encoding="utf-8") as out:
        out.write("$ " + " ".join(cmd) + "\n\n")
        out.flush()
        subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    log(f"launched_semantic log={child_log.relative_to(ROOT)}", main_log)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relaunch gpu1_sd_gan semantic benchmark if queue exits early.")
    parser.add_argument("--config", default="configs/gpu1_sd_gan.yaml")
    parser.add_argument("--run-matrix", default="results/manifests/gpu1_sd_gan/run_matrix.csv")
    parser.add_argument("--registry", default="results/reports/gpu1_sd_gan/trained_checkpoints.csv")
    parser.add_argument("--expected-runs", type=int, default=30)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--gpus", default="1:3")
    args = parser.parse_args()

    lock_dir = ROOT / "results" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "gpu1_sd_gan_launcher_watchdog.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another gpu1_sd_gan launcher watchdog is already running.")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = ROOT / "results" / "logs" / f"gpu1_sd_gan_launch_watchdog_{stamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        main_log = log_dir / "watchdog.log"
        registry = ROOT / args.registry
        matrix = ROOT / args.run_matrix

        while True:
            missing = missing_run_ids(registry, matrix)
            registered = args.expected_runs - len(missing)
            lines = process_lines()
            active_train = active_train_run_ids(lines)
            launchers = active_launcher_count(lines)
            log(
                f"gpu1_sd_gan_registered={registered}/{args.expected_runs} "
                f"missing_planned={len(missing)} active_train={len(active_train)} active_launchers={launchers}",
                main_log,
            )
            if not missing:
                log("complete", main_log)
                return
            if launchers == 0 and not active_train:
                launch(args, log_dir, main_log)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
