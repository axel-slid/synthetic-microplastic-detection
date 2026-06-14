#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "smp_deeplabv3plus_resnet50": "DeepLabV3+",
    "smp_unet_resnet34": "U-Net",
    "smp_unetpp_effb4": "U-Net++",
}


def parse_run_id(run_id: str) -> tuple[str, str, int]:
    condition, model, seed_text = run_id.rsplit("__", 2)
    return condition, model, int(seed_text.replace("seed", ""))


def load_best_val(path: Path) -> float | None:
    if not path.exists():
        return None
    vals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            vals.append(float(json.loads(line)["dice_mean"]))
    return max(vals) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="results/runs/half_real_top_inpaint_pilot")
    parser.add_argument("--out-dir", default="results/reports/half_real_top_inpaint_pilot")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary_path in sorted(run_root.glob("*/c3_clean_metrics.summary.json")):
        run_id = summary_path.parent.name
        condition, model, seed = parse_run_id(run_id)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": run_id,
                "condition": condition,
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "seed": seed,
                "best_val_dice": load_best_val(summary_path.parent / "history.jsonl"),
                **payload,
            }
        )
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise SystemExit("No C3-clean summaries found.")
    runs = runs.sort_values("dice_mean", ascending=False)
    runs.to_csv(out_dir / "pilot_c3_run_metrics.csv", index=False)
    by_model = runs[
        [
            "model_label",
            "seed",
            "dice_mean",
            "iou_mean",
            "boundary_f1_mean",
            "precision_mean",
            "recall_mean",
            "best_val_dice",
        ]
    ].sort_values("dice_mean", ascending=False)
    by_model.to_csv(out_dir / "pilot_c3_by_model.csv", index=False)
    print(by_model.to_string(index=False))


if __name__ == "__main__":
    main()
