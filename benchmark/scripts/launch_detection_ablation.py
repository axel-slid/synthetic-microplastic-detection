#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import queue
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
        workers.extend([gpu] * int(count_text or "1"))
    if not workers:
        raise SystemExit("No GPU workers requested.")
    return workers


def checkpoint_exists(row: pd.Series) -> bool:
    return (Path(row.output_dir) / "train" / "weights" / "best.pt").exists()


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

        log_path = log_dir / f"{row.run_id}.log"
        started = datetime.now().isoformat(timespec="seconds")
        cmd = [
            sys.executable,
            "scripts/train_yolo_detector.py",
            "--run-matrix",
            args.run_matrix,
            "--run-id",
            row.run_id,
            "--weights",
            row.weights,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--registry",
            args.registry,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
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
                "seconds": round(time.time() - begin, 2),
                "log": str(log_path),
            },
        )
        jobs.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch YOLO detection ablation runs.")
    parser.add_argument("--run-matrix", default="results/manifests/detection_ablation/run_matrix.csv")
    parser.add_argument("--gpus", default="0:1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--registry", default="results/reports/detection_ablation/trained_checkpoints.csv")
    parser.add_argument("--include-completed", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(ROOT / args.run_matrix)
    if not args.include_completed:
        frame = frame[~frame.apply(checkpoint_exists, axis=1)].reset_index(drop=True)
    if frame.empty:
        print("No detection runs to launch.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "results" / "logs" / f"detection_ablation_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.csv"
    jobs: queue.Queue[pd.Series] = queue.Queue()
    for row in frame.itertuples(index=False):
        jobs.put(pd.Series(row._asdict()))

    workers = parse_gpu_workers(args.gpus)
    print(f"Launching {len(frame)} detection runs across {len(workers)} workers: {args.gpus}")
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
    print("Detection launcher finished.")


if __name__ == "__main__":
    main()
