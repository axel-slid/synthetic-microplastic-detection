#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microplastic_benchmark.config import ensure_dirs, load_config
from microplastic_benchmark.manifests import planned_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--out", default="results/manifests/run_matrix.csv")
    parser.add_argument("--commands", default="results/manifests/run_matrix.sh")
    parser.add_argument("--available-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    runs = planned_runs(cfg, require_manifest=args.available_only)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)

    by_model = {m["name"]: m for m in cfg["segmentation_models"]}
    with Path(args.commands).open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for run in runs:
            family = run["family"]
            if family == "semantic":
                f.write(
                    "python scripts/train_segmenter.py "
                    f"--config {args.config} --run-id {run['run_id']}\n"
                )
            elif family == "yolo":
                weights = by_model[run["model"]].get("weights", "yolo11m-seg.pt")
                f.write(
                    "python scripts/train_yolo_segmenter.py "
                    f"--config {args.config} --run-id {run['run_id']} --weights {weights}\n"
                )
    print(f"Wrote {out}")
    print(f"Wrote {args.commands}")


if __name__ == "__main__":
    main()
