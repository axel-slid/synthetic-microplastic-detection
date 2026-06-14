#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import fcntl
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_gpu_workers(spec: str) -> list[str]:
    workers: list[str] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        gpu, _, count_text = item.partition(":")
        count = int(count_text or "1")
        workers.extend([gpu] * count)
    if not workers:
        raise SystemExit("No GPU workers requested.")
    return workers


def checkpoint_exists(row: pd.Series) -> bool:
    out_dir = Path(row.output_dir)
    if row.family == "yolo":
        return (out_dir / "train" / "weights" / "best.pt").exists()
    return (out_dir / "best.pt").exists()


def registry_path(args: argparse.Namespace) -> Path:
    if args.registry:
        return ROOT / args.registry
    run_matrix = Path(args.run_matrix)
    if "sd_ablation" in run_matrix.parts:
        return ROOT / "results" / "reports" / "sd_ablation" / "trained_checkpoints.csv"
    return ROOT / "results" / "reports" / "trained_checkpoints.csv"


def registered_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path).drop_duplicates("run_id", keep="last")
    return set(frame["run_id"].astype(str))


def active_run_ids() -> set[str]:
    patterns = [
        re.compile(r"scripts/train_segmenter\.py .*?--run-id ([^ ]+)"),
        re.compile(r"scripts/train_yolo_segmenter\.py .*?--run-id ([^ ]+)"),
    ]
    try:
        output = subprocess.check_output(["ps", "-eo", "args"], text=True)
    except subprocess.SubprocessError:
        return set()
    active: set[str] = set()
    for line in output.splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                active.add(match.group(1))
                break
    return active


def command_for(row: pd.Series, args: argparse.Namespace) -> list[str]:
    if row.family == "yolo":
        cmd = [
            sys.executable,
            "scripts/train_yolo_segmenter.py",
            "--config",
            args.config,
            "--run-id",
            row.run_id,
            "--weights",
            args.yolo_weights,
        ]
    else:
        cmd = [
            sys.executable,
            "scripts/train_segmenter.py",
            "--config",
            args.config,
            "--run-id",
            row.run_id,
        ]
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    return cmd


def append_status(path: Path, lock: threading.Lock, row: dict[str, str | int | float]) -> None:
    with lock:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def run_worker(
    worker_name: str,
    gpu: str,
    jobs: queue.Queue[pd.Series],
    args: argparse.Namespace,
    log_dir: Path,
    status_path: Path,
    status_lock: threading.Lock,
) -> None:
    while True:
        try:
            row = jobs.get_nowait()
        except queue.Empty:
            return

        lock_dir = ROOT / "results" / "locks" / "benchmark"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{row.run_id}.lock"
        log_path = log_dir / f"{row.run_id}.log"
        with lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                append_status(
                    status_path,
                    status_lock,
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "worker": worker_name,
                        "gpu": gpu,
                        "run_id": row.run_id,
                        "status": "skipped_locked",
                        "returncode": "",
                        "seconds": "",
                        "log": str(log_path),
                    },
                )
                jobs.task_done()
                continue
            registry = registry_path(args)
            if not args.include_completed and row.run_id in registered_run_ids(registry):
                append_status(
                    status_path,
                    status_lock,
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "worker": worker_name,
                        "gpu": gpu,
                        "run_id": row.run_id,
                        "status": "skipped_registered",
                        "returncode": "",
                        "seconds": "",
                        "log": str(log_path),
                    },
                )
                jobs.task_done()
                continue
            if not args.include_completed and row.run_id in active_run_ids():
                append_status(
                    status_path,
                    status_lock,
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "worker": worker_name,
                        "gpu": gpu,
                        "run_id": row.run_id,
                        "status": "skipped_active",
                        "returncode": "",
                        "seconds": "",
                        "log": str(log_path),
                    },
                )
                jobs.task_done()
                continue

            started = datetime.now().isoformat(timespec="seconds")
            cmd = command_for(row, args)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("OMP_NUM_THREADS", "4")

            append_status(
                status_path,
                status_lock,
                {
                    "time": started,
                    "worker": worker_name,
                    "gpu": gpu,
                    "run_id": row.run_id,
                    "status": "started",
                    "returncode": "",
                    "seconds": "",
                    "log": str(log_path),
                },
            )
            begin = time.time()
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"started={started}\nworker={worker_name}\ngpu={gpu}\ncmd={' '.join(cmd)}\n\n")
                log.flush()
                proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            elapsed = round(time.time() - begin, 2)
            append_status(
                status_path,
                status_lock,
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "worker": worker_name,
                    "gpu": gpu,
                    "run_id": row.run_id,
                    "status": "finished" if proc.returncode == 0 else "failed",
                    "returncode": proc.returncode,
                    "seconds": elapsed,
                    "log": str(log_path),
                },
            )
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        jobs.task_done()


def postprocess(args: argparse.Namespace, log_dir: Path) -> None:
    if not args.evaluate_on_finish:
        return
    commands = [
        [sys.executable, "scripts/evaluate_registry.py", "--config", args.config],
        [sys.executable, "scripts/aggregate_results.py"],
        [sys.executable, "scripts/make_paper_assets.py"],
    ]
    if args.compile_paper:
        commands.append(["latexmk", "-pdf", "-interaction=nonstopmode", "paper.tex"])
    log_path = log_dir / "postprocess.log"
    with log_path.open("w", encoding="utf-8") as log:
        for cmd in commands:
            log.write(f"\n$ {' '.join(cmd)}\n")
            log.flush()
            cwd = ROOT.parent if cmd[0] == "latexmk" else ROOT
            result = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                raise SystemExit(f"Postprocess command failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--run-matrix", default="results/manifests/run_matrix.csv")
    parser.add_argument("--gpus", default="1:3,0:1", help="Comma-separated GPU:worker_count list.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--yolo-weights", default="yolo11m-seg.pt")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--family", action="append", choices=["semantic", "yolo"])
    parser.add_argument("--model", action="append")
    parser.add_argument("--condition", action="append")
    parser.add_argument("--include-completed", action="store_true")
    parser.add_argument("--registry")
    parser.add_argument("--evaluate-on-finish", action="store_true")
    parser.add_argument("--compile-paper", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(ROOT / args.run_matrix)
    if args.family:
        frame = frame[frame["family"].isin(args.family)].reset_index(drop=True)
    if args.model:
        frame = frame[frame["model"].isin(args.model)].reset_index(drop=True)
    if args.condition:
        frame = frame[frame["condition"].isin(args.condition)].reset_index(drop=True)
    if not args.include_completed:
        registered = registered_run_ids(registry_path(args))
        active = active_run_ids()
        frame = frame[
            ~frame["run_id"].astype(str).isin(registered | active)
        ].reset_index(drop=True)
    if args.limit is not None:
        frame = frame.head(args.limit).reset_index(drop=True)
    if frame.empty:
        print("No runs to launch.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "results" / "logs" / f"benchmark_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.csv"

    jobs: queue.Queue[pd.Series] = queue.Queue()
    for row in frame.itertuples(index=False):
        jobs.put(pd.Series(row._asdict()))

    workers = parse_gpu_workers(args.gpus)
    print(f"Launching {len(frame)} runs across {len(workers)} workers: {args.gpus}")
    print(f"Logs: {log_dir}")

    status_lock = threading.Lock()
    threads = []
    for idx, gpu in enumerate(workers, start=1):
        thread = threading.Thread(
            target=run_worker,
            args=(f"worker{idx}", gpu, jobs, args, log_dir, status_path, status_lock),
            daemon=False,
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    postprocess(args, log_dir)
    print("Benchmark launcher finished.")


if __name__ == "__main__":
    main()
