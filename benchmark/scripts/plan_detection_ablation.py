#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


TRACKS = [
    {
        "condition": "pure_real",
        "summary": "results/detection_sd_ablation/summary.csv",
        "track": "no_synthetic",
    },
    {
        "condition": "real_plus_synthetic",
        "summary": "results/detection_sd_ablation/summary.csv",
        "track": "synthetic",
    },
    {
        "condition": "pure_synthetic",
        "summary": "results/detection_sd_ablation/summary.csv",
        "track": "full_synthetic",
    },
    {
        "condition": "real_plus_inpainting",
        "summary": "results/detection_inpaint_ablation/summary.csv",
        "track": "synthetic",
    },
    {
        "condition": "pure_inpainting",
        "summary": "results/detection_inpaint_ablation/summary.csv",
        "track": "full_synthetic",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan YOLO detection runs for the SD ablation datasets.")
    parser.add_argument("--out", default="results/manifests/detection_ablation/run_matrix.csv")
    parser.add_argument("--commands", default="results/manifests/detection_ablation/run_matrix.sh")
    parser.add_argument("--runs-root", default="results/runs/detection_ablation")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--seeds", default="13,37,101")
    args = parser.parse_args()

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    rows = []
    for track in TRACKS:
        summary = pd.read_csv(track["summary"])
        match = summary[summary["track"] == track["track"]]
        if match.empty:
            raise SystemExit(f"Missing track {track['track']} in {track['summary']}")
        dataset_yaml = str(match.iloc[0]["dataset_yaml"])
        for seed in seeds:
            run_id = f"{track['condition']}__yolo26n_det__seed{seed}"
            rows.append(
                {
                    "run_id": run_id,
                    "condition": track["condition"],
                    "model": "yolo26n_det",
                    "family": "detection",
                    "seed": seed,
                    "dataset_yaml": dataset_yaml,
                    "output_dir": str(Path(args.runs_root) / run_id),
                    "weights": args.weights,
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    commands = Path(args.commands)
    commands.parent.mkdir(parents=True, exist_ok=True)
    commands.write_text(
        "\n".join(
            (
                "python scripts/train_yolo_detector.py "
                f"--run-matrix {out} --run-id {row['run_id']} --weights {row['weights']}"
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    print(f"Wrote {commands}")


if __name__ == "__main__":
    main()
