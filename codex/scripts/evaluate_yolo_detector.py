#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO detector on the C3-clean detection set.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="results/detection_eval/c3_clean/dataset.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    model = YOLO(args.checkpoint)
    metrics = model.val(data=args.data, imgsz=args.image_size, split="test", task="detect")
    box = getattr(metrics, "box", None)
    summary = {
        "checkpoint": args.checkpoint,
        "data": args.data,
        "box_map50": float(getattr(box, "map50", 0.0)) if box is not None else 0.0,
        "box_map50_95": float(getattr(box, "map", 0.0)) if box is not None else 0.0,
        "box_precision_mean": float(getattr(box, "mp", 0.0)) if box is not None else 0.0,
        "box_recall_mean": float(getattr(box, "mr", 0.0)) if box is not None else 0.0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
