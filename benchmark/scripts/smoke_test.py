#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from microplastic_benchmark.config import load_config
from microplastic_benchmark.evaluation import evaluate_checkpoint
from microplastic_benchmark.training import train_semantic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    base = pd.read_csv(Path(cfg["paths"]["manifests"]) / "baseline_c1.csv")
    smoke = pd.concat(
        [base[base.split == "train"].head(8), base[base.split == "val"].head(4)],
        ignore_index=True,
    )
    smoke_path = Path(cfg["paths"]["manifests"]) / "smoke.csv"
    smoke.to_csv(smoke_path, index=False)
    out_dir = Path(cfg["paths"]["runs"]) / "smoke_tiny_unet"
    checkpoint = train_semantic(
        smoke_path,
        {"name": "tiny_unet", "family": "semantic", "library": "builtin"},
        out_dir,
        seed=13,
        image_size=128,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        patience=1,
        threshold=0.5,
        device_name=cfg["project"]["device"],
        num_workers=0,
        amp=False,
    )
    summary = evaluate_checkpoint(
        checkpoint,
        Path(cfg["paths"]["manifests"]) / "c3_clean.csv",
        out_dir / "c3_clean_metrics.csv",
        threshold=0.5,
        device_name=cfg["project"]["device"],
        split="test",
    )
    print(f"checkpoint={checkpoint}")
    print(summary)


if __name__ == "__main__":
    main()
