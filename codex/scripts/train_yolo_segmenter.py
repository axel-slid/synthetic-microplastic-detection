#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import fcntl
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from PIL import Image
from ultralytics import YOLO

from microplastic_benchmark.config import load_config
from microplastic_benchmark.data import mask_to_polygons, read_manifest
from microplastic_benchmark.manifests import planned_runs


def find_run(cfg, run_id: str) -> dict:
    for run in planned_runs(cfg):
        if run["run_id"] == run_id:
            return run
    raise SystemExit(f"Unknown run_id: {run_id}")


def link_or_copy(src: str | Path, dst: str | Path) -> None:
    src = Path(src).resolve()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_yolo_label(mask_path: str | Path, label_path: str | Path, image_size: tuple[int, int]) -> None:
    width, height = image_size
    lines = []
    for poly in mask_to_polygons(mask_path):
        coords = []
        for x, y in poly:
            coords.append(f"{max(0.0, min(1.0, x / width)):.6f}")
            coords.append(f"{max(0.0, min(1.0, y / height)):.6f}")
        if len(coords) >= 6:
            lines.append("0 " + " ".join(coords))
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_yolo_dataset(frame: pd.DataFrame, out_dir: Path) -> Path:
    for split in ("train", "val"):
        split_df = frame[frame.split == split]
        for row in split_df.itertuples(index=False):
            image_path = Path(row.image_path)
            out_img = out_dir / "images" / split / image_path.name
            out_label = out_dir / "labels" / split / f"{image_path.stem}.txt"
            link_or_copy(image_path, out_img)
            with Image.open(image_path) as im:
                write_yolo_label(row.mask_path, out_label, im.size)
    data_yaml = out_dir / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "microplastic"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


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
    parser.add_argument("--weights", default="yolo11m-seg.pt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = find_run(cfg, args.run_id)
    frame = read_manifest(run["manifest"])
    out_dir = Path(run["output_dir"])
    yolo_data = export_yolo_dataset(frame, out_dir / "yolo_dataset")
    model = YOLO(args.weights)
    result = model.train(
        data=str(yolo_data),
        imgsz=int(cfg["project"]["image_size"]),
        epochs=args.epochs or int(cfg["training"]["epochs"]),
        batch=args.batch_size or int(cfg["training"]["batch_size"]),
        project=str(out_dir),
        name="train",
        seed=int(run["seed"]),
        task="segment",
    )
    append_registry(
        Path(cfg["paths"]["reports"]) / "trained_checkpoints.csv",
        {**run, "checkpoint": str(Path(result.save_dir) / "weights" / "best.pt")},
    )
    print(Path(result.save_dir) / "weights" / "best.pt")


if __name__ == "__main__":
    main()
