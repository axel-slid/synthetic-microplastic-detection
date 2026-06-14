#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all trained detection ablation checkpoints.")
    parser.add_argument("--registry", default="results/reports/detection_ablation/trained_checkpoints.csv")
    parser.add_argument("--data", default="results/detection_eval/c3_clean/dataset.yaml")
    parser.add_argument("--out-dir", default="results/reports/detection_ablation/evaluations")
    parser.add_argument("--aggregate", default="results/reports/detection_ablation/aggregate_c3_clean.csv")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(ROOT / args.registry).drop_duplicates("run_id", keep="last")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in frame.itertuples(index=False):
        out = out_dir / f"{row.run_id}.summary.json"
        if not (args.skip_existing and out.exists()):
            subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_yolo_detector.py",
                    "--checkpoint",
                    row.checkpoint,
                    "--data",
                    args.data,
                    "--out",
                    str(out.relative_to(ROOT)),
                ],
                cwd=ROOT,
                check=True,
            )
        payload = json.loads(out.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": row.run_id,
                "condition": row.condition,
                "model": row.model,
                "seed": int(row.seed),
                **payload,
            }
        )

    aggregate = ROOT / args.aggregate
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(aggregate, index=False)
    print(aggregate)


if __name__ == "__main__":
    main()
