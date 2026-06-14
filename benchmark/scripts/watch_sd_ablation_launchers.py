#!/usr/bin/env python
from __future__ import annotations

import argparse
import fcntl
import os
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


def missing_run_ids(registry_path: Path, matrix_path: Path) -> set[str]:
    planned = set(pd.read_csv(matrix_path)["run_id"].astype(str))
    if not registry_path.exists():
        return planned
    registered = set(pd.read_csv(registry_path).drop_duplicates("run_id", keep="last")["run_id"].astype(str))
    return planned - registered


def process_lines() -> list[str]:
    try:
        output = subprocess.check_output(["ps", "-eo", "args"], text=True)
    except subprocess.SubprocessError:
        return []
    return output.splitlines()


def active_sd_train_counts(lines: list[str]) -> dict[str, int]:
    run_ids: dict[str, set[str]] = {"semantic": set(), "yolo": set()}
    patterns = [
        ("semantic", re.compile(r"scripts/train_segmenter\.py .*?--run-id ([^ ]+)")),
        ("yolo", re.compile(r"scripts/train_yolo_segmenter\.py .*?--run-id ([^ ]+)")),
    ]
    for line in lines:
        if "configs/sd_ablation.yaml" not in line:
            continue
        for family, pattern in patterns:
            match = pattern.search(line)
            if match:
                run_ids[family].add(match.group(1))
                break
    return {family: len(ids) for family, ids in run_ids.items()}


def active_sd_launcher_counts(lines: list[str]) -> dict[str, int]:
    counts = {"semantic": 0, "yolo": 0}
    for line in lines:
        if "scripts/launch_benchmark.py" not in line or "configs/sd_ablation.yaml" not in line:
            continue
        if "--family semantic" in line:
            counts["semantic"] += 1
        elif "--family yolo" in line:
            counts["yolo"] += 1
    return counts


def missing_by_family(registry_path: Path, matrix_path: Path) -> dict[str, set[str]]:
    matrix = pd.read_csv(matrix_path)
    registered: set[str] = set()
    if registry_path.exists():
        registered = set(pd.read_csv(registry_path).drop_duplicates("run_id", keep="last")["run_id"].astype(str))
    missing = matrix[~matrix["run_id"].astype(str).isin(registered)]
    return {
        "semantic": set(missing[missing["family"].astype(str) == "semantic"]["run_id"].astype(str)),
        "yolo": set(missing[missing["family"].astype(str) == "yolo"]["run_id"].astype(str)),
    }


def launch_family(family: str, args: argparse.Namespace, log_dir: Path, main_log: Path) -> None:
    if family == "semantic":
        cmd = [
            sys.executable,
            "scripts/launch_benchmark.py",
            "--config",
            args.config,
            "--run-matrix",
            args.run_matrix,
            "--family",
            "semantic",
            "--gpus",
            args.semantic_gpus,
        ]
    elif family == "yolo":
        cmd = [
            sys.executable,
            "scripts/launch_benchmark.py",
            "--config",
            args.config,
            "--run-matrix",
            args.run_matrix,
            "--family",
            "yolo",
            "--gpus",
            args.yolo_gpus,
            "--yolo-weights",
            args.yolo_weights,
        ]
    else:
        raise ValueError(f"Unknown family: {family}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    child_log = log_dir / f"resume_{family}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.out"
    with child_log.open("w", encoding="utf-8") as out:
        out.write("$ " + " ".join(cmd) + "\n\n")
        out.flush()
        subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    log(f"launched_{family} log={child_log.relative_to(ROOT)}", main_log)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relaunch SD ablation benchmark launchers if the queue stops early.")
    parser.add_argument("--config", default="configs/sd_ablation.yaml")
    parser.add_argument("--run-matrix", default="results/manifests/sd_ablation/run_matrix.csv")
    parser.add_argument("--registry", default="results/reports/sd_ablation/trained_checkpoints.csv")
    parser.add_argument("--expected-runs", type=int, default=105)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--semantic-gpus", default="1:3,2:3")
    parser.add_argument("--yolo-gpus", default="0:1")
    parser.add_argument("--yolo-weights", default="yolo11m-seg.pt")
    args = parser.parse_args()

    lock_dir = ROOT / "results" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "sd_ablation_launcher_watchdog.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another SD ablation launcher watchdog is already running.")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = ROOT / "results" / "logs" / f"sd_ablation_launch_watchdog_{stamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        main_log = log_dir / "watchdog.log"
        registry = ROOT / args.registry
        run_matrix = ROOT / args.run_matrix

        while True:
            missing = missing_run_ids(registry, run_matrix)
            missing_families = missing_by_family(registry, run_matrix)
            registered = args.expected_runs - len(missing)
            lines = process_lines()
            train_counts = active_sd_train_counts(lines)
            launcher_counts = active_sd_launcher_counts(lines)
            log(
                f"sd_ablation_registered={registered}/{args.expected_runs} "
                f"missing_planned={len(missing)} "
                f"missing_semantic={len(missing_families['semantic'])} missing_yolo={len(missing_families['yolo'])} "
                f"active_train_semantic={train_counts['semantic']} active_train_yolo={train_counts['yolo']} "
                f"active_launchers_semantic={launcher_counts['semantic']} active_launchers_yolo={launcher_counts['yolo']}",
                main_log,
            )
            if not missing:
                log("complete", main_log)
                return
            for family in ("semantic", "yolo"):
                if missing_families[family] and train_counts[family] == 0 and launcher_counts[family] == 0:
                    launch_family(family, args, log_dir, main_log)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
