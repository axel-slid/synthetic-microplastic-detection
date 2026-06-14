#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import fcntl
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


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
    parser = argparse.ArgumentParser(description="Train one YOLO detection run from a detection ablation matrix.")
    parser.add_argument("--run-matrix", default="results/manifests/detection_ablation/run_matrix.csv")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--registry", default="results/reports/detection_ablation/trained_checkpoints.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.run_matrix)
    matches = frame[frame["run_id"] == args.run_id]
    if matches.empty:
        raise SystemExit(f"Unknown run_id: {args.run_id}")
    row = matches.iloc[0].to_dict()
    out_dir = Path(row["output_dir"])
    weights = args.weights or row.get("weights") or "yolo26n.pt"
    model = YOLO(str(weights))
    result = model.train(
        data=str(row["dataset_yaml"]),
        imgsz=args.image_size,
        epochs=args.epochs,
        batch=args.batch_size,
        project=str(out_dir),
        name="train",
        seed=int(row["seed"]),
        task="detect",
    )
    checkpoint = str(Path(result.save_dir) / "weights" / "best.pt")
    append_registry(
        Path(args.registry),
        {
            **row,
            "checkpoint": checkpoint,
        },
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
