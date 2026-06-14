#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def quality_score(frame: pd.DataFrame) -> pd.Series:
    cols = [
        "qc_masked_mad_vs_background",
        "qc_changed_px_frac_vs_background",
        "qc_masked_mad_vs_seeded",
        "qc_changed_px_frac_vs_seeded",
    ]
    score = pd.Series(0.0, index=frame.index)
    for col in cols:
        lo = frame[col].quantile(0.01)
        hi = frame[col].quantile(0.99)
        score += ((frame[col].clip(lo, hi) - lo) / max(hi - lo, 1e-9)).fillna(0.0)
    # Prefer visible but not extreme masks; very large pasted masks are less realistic.
    center = frame["qc_mask_frac"].median()
    spread = max(frame["qc_mask_frac"].quantile(0.95) - frame["qc_mask_frac"].quantile(0.05), 1e-9)
    score -= 0.25 * ((frame["qc_mask_frac"] - center).abs() / spread)
    return score


def select_diverse(frame: pd.DataFrame, n: int, background_cap: int, source_cap: int) -> pd.DataFrame:
    selected = []
    bg_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in frame.sort_values("quality_score", ascending=False).itertuples(index=False):
        bg = str(row.background_path)
        src = str(row.source_mask_path)
        if bg_counts.get(bg, 0) >= background_cap:
            continue
        if source_counts.get(src, 0) >= source_cap:
            continue
        selected.append(row._asdict())
        bg_counts[bg] = bg_counts.get(bg, 0) + 1
        source_counts[src] = source_counts.get(src, 0) + 1
        if len(selected) == n:
            break
    if len(selected) < n:
        raise SystemExit(f"Only selected {len(selected)} diverse rows, need {n}. Relax caps.")
    return pd.DataFrame(selected)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", default="results/manifests/gpu1_sd_gan/gpu1_sd_inpaint.csv")
    parser.add_argument("--generation-log", default="data/c2/gpu1_sd_inpaint/generation_log.csv")
    parser.add_argument("--c3-clean", default="results/manifests/gpu1_sd_gan/c3_clean.csv")
    parser.add_argument("--out-dir", default="results/manifests/half_real_top_inpaint_pilot")
    parser.add_argument("--condition", default="half_real_top_inpaint")
    parser.add_argument("--background-cap", type=int, default=1)
    parser.add_argument("--source-cap", type=int, default=2)
    args = parser.parse_args()

    base = pd.read_csv(ROOT / args.base_manifest)
    real_train = base[(base["source"] == "c1") & (base["split"] == "train")].copy()
    real_val = base[(base["source"] == "c1") & (base["split"] == "val")].copy()
    if real_train.empty or real_val.empty:
        raise SystemExit("Base manifest must contain real train and val rows.")

    need = len(real_train) + len(real_val)
    log = pd.read_csv(ROOT / args.generation_log)
    log["quality_score"] = quality_score(log)
    selected = select_diverse(log, need, args.background_cap, args.source_cap)
    selected = selected.sort_values("quality_score", ascending=False).reset_index(drop=True)

    synth_rows = []
    for i, row in selected.iterrows():
        split = "train" if i < len(real_train) else "val"
        synth_rows.append(
            {
                "name": row["generated_name"],
                "image_path": str(Path("data/c2/gpu1_sd_inpaint") / row["generated_name"]),
                "mask_path": str(row["output_mask_path"]),
                "split": split,
                "source": "synthetic_top_inpaint",
                "condition": args.condition,
            }
        )

    real_rows = []
    for frame in (real_train, real_val):
        tmp = frame.copy()
        tmp["condition"] = args.condition
        real_rows.extend(tmp[["name", "image_path", "mask_path", "split", "source", "condition"]].to_dict("records"))

    rows = real_rows + synth_rows
    out_dir = ROOT / args.out_dir
    manifest_path = out_dir / f"{args.condition}.csv"
    write_rows(manifest_path, rows)
    c3 = pd.read_csv(ROOT / args.c3_clean)
    write_rows(out_dir / "c3_clean.csv", c3.to_dict("records"))

    selected_report = selected[
        [
            "generated_name",
            "background_path",
            "source_mask_path",
            "quality_score",
            "qc_masked_mad_vs_background",
            "qc_changed_px_frac_vs_background",
            "qc_masked_mad_vs_seeded",
            "qc_changed_px_frac_vs_seeded",
            "qc_mask_frac",
        ]
    ]
    selected_report.to_csv(out_dir / "selected_top_inpaint_qc.csv", index=False)
    summary = {
        "condition": args.condition,
        "real_train": int(len(real_train)),
        "synthetic_train": int(len(real_train)),
        "real_val": int(len(real_val)),
        "synthetic_val": int(len(real_val)),
        "selected_total": int(len(selected)),
        "background_cap": args.background_cap,
        "source_cap": args.source_cap,
        "quality_score_min": float(selected["quality_score"].min()),
        "quality_score_median": float(selected["quality_score"].median()),
        "quality_score_max": float(selected["quality_score"].max()),
        "qc_masked_mad_vs_background_min": float(selected["qc_masked_mad_vs_background"].min()),
        "qc_changed_px_frac_vs_background_min": float(selected["qc_changed_px_frac_vs_background"].min()),
        "qc_masked_mad_vs_seeded_min": float(selected["qc_masked_mad_vs_seeded"].min()),
        "qc_changed_px_frac_vs_seeded_min": float(selected["qc_changed_px_frac_vs_seeded"].min()),
        "qc_mask_frac_min": float(selected["qc_mask_frac"].min()),
        "qc_mask_frac_max": float(selected["qc_mask_frac"].max()),
    }
    (out_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(manifest_path)


if __name__ == "__main__":
    main()
