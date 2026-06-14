#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/reports/aggregate_c3_clean.csv")
    args = parser.parse_args()
    root = Path(args.results_dir)
    rows = []
    summary_paths = list(root.glob("runs/**/*.summary.json"))
    if not summary_paths:
        summary_paths = list(root.glob("**/*.summary.json"))
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        run_id = summary_path.parent.name
        bits = run_id.split("__")
        row = {"run_id": run_id}
        if len(bits) == 3:
            row.update({"condition": bits[0], "model": bits[1], "seed": bits[2].replace("seed", "")})
        row.update(payload)
        rows.append(row)
    if not rows:
        raise SystemExit(f"No summary files found under {root}/runs")
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
