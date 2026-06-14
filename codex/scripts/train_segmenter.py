#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import fcntl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microplastic_benchmark.config import load_config
from microplastic_benchmark.manifests import planned_runs
from microplastic_benchmark.training import train_semantic


def find_run(cfg, run_id: str) -> dict:
    for run in planned_runs(cfg):
        if run["run_id"] == run_id:
            return run
    raise SystemExit(f"Unknown run_id: {run_id}")


def find_model(cfg, name: str) -> dict:
    for model in cfg["segmentation_models"]:
        if model["name"] == name:
            return model
    raise SystemExit(f"Unknown model: {name}")


def append_registry(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = find_run(cfg, args.run_id)
    model = find_model(cfg, run["model"])
    if model["family"] != "semantic":
        raise SystemExit(f"{run['model']} is not a semantic PyTorch model. Use train_yolo_segmenter.py.")

    train_cfg = cfg["training"]
    best = train_semantic(
        run["manifest"],
        model,
        run["output_dir"],
        seed=int(run["seed"]),
        image_size=int(cfg["project"]["image_size"]),
        batch_size=args.batch_size or int(train_cfg["batch_size"]),
        epochs=args.epochs or int(train_cfg["epochs"]),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        patience=int(train_cfg["patience"]),
        threshold=float(train_cfg["threshold"]),
        device_name=cfg["project"]["device"],
        num_workers=int(cfg["project"]["num_workers"]),
        amp=bool(train_cfg["amp"]),
    )
    append_registry(
        Path(cfg["paths"]["reports"]) / "trained_checkpoints.csv",
        {
            **run,
            "checkpoint": str(best),
        },
    )
    print(best)


if __name__ == "__main__":
    main()
