#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "runs" / "gpu1_sd_gan"
OUT = ROOT / "results" / "reports" / "gpu1_sd_gan" / "interim_c3_smoke_summary.csv"


def parse_run_id(run_id: str) -> tuple[str, str, int]:
    condition, model, seed_text = run_id.rsplit("__", 2)
    return condition, model, int(seed_text.replace("seed", ""))


def main() -> None:
    rows = []
    paths = list(RUN_ROOT.glob("*/interim_latest_c3_clean_metrics.summary.json"))
    source = "interim_latest_best_pt_smoke"
    if not paths:
        paths = list(RUN_ROOT.glob("*/interim_c3_clean_metrics.summary.json"))
        source = "interim_best_pt_smoke"
    for path in sorted(paths):
        run_id = path.parent.name
        condition, model, seed = parse_run_id(run_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": run_id,
                "condition": condition,
                "model": model,
                "seed": seed,
                "dice_mean": data.get("dice_mean"),
                "iou_mean": data.get("iou_mean"),
                "boundary_f1_mean": data.get("boundary_f1_mean"),
                "precision_mean": data.get("precision_mean"),
                "recall_mean": data.get("recall_mean"),
                "source": source,
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False)
    if frame.empty:
        print(f"No interim C3 smoke summaries found under {RUN_ROOT}")
        return
    print(frame.sort_values("dice_mean", ascending=False).to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
