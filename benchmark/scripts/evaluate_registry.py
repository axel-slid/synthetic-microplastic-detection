#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from microplastic_benchmark.config import load_config
from microplastic_benchmark.evaluation import evaluate_checkpoint, evaluate_yolo_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--registry", default="results/reports/trained_checkpoints.csv")
    parser.add_argument("--manifest", default="results/manifests/c3_clean.csv")
    parser.add_argument("--device")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    frame = pd.read_csv(args.registry).drop_duplicates("run_id", keep="last")
    for row in frame.itertuples(index=False):
        out = Path(row.output_dir) / "c3_clean_metrics.csv"
        summary_path = out.with_suffix(".summary.json")
        if args.skip_existing and out.exists() and summary_path.exists():
            print(row.run_id, "skip-existing")
            continue
        if getattr(row, "family", "") == "yolo":
            summary = evaluate_yolo_checkpoint(row.checkpoint, args.manifest, out, split="test")
        else:
            summary = evaluate_checkpoint(
                row.checkpoint,
                args.manifest,
                out,
                threshold=float(cfg["training"]["threshold"]),
                device_name=args.device or cfg["project"]["device"],
                split="test",
            )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(row.run_id, summary.get("dice_mean"), summary.get("iou_mean"))


if __name__ == "__main__":
    main()
