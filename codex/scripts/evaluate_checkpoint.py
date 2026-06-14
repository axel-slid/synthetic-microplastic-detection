#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microplastic_benchmark.config import load_config
from microplastic_benchmark.evaluation import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="results/manifests/c3_clean.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    summary = evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        args.out,
        threshold=float(cfg["training"]["threshold"]),
        device_name=cfg["project"]["device"],
        split=args.split,
    )
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
