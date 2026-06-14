#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microplastic_benchmark.data import load_mask_l, load_rgb, read_manifest
from microplastic_benchmark.training import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = PROJECT / "overleaf_microplastic_project" / "images"
RUN_ROOT = ROOT / "results" / "runs" / "sd_ablation"
MODEL_LABELS = {
    "monai_unet": "MONAI U-Net",
    "segformer_b2": "SegFormer-B2",
    "smp_deeplabv3plus_resnet50": "DeepLabV3+",
    "smp_fpn_effb3": "FPN",
    "smp_unet_resnet34": "U-Net",
    "smp_unetpp_effb4": "U-Net++",
}
COND_LABELS = {
    "pure_real": "Real only",
    "real_plus_inpainting": "Real + inpainting",
}


def parse_run_id(run_id: str) -> tuple[str, str, int]:
    condition, model, seed_text = run_id.rsplit("__", 2)
    return condition, model, int(seed_text.replace("seed", ""))


def load_summaries() -> pd.DataFrame:
    rows = []
    for summary_path in sorted(RUN_ROOT.glob("*__*/c3_clean_metrics.summary.json")):
        run_id = summary_path.parent.name
        condition, model, seed = parse_run_id(run_id)
        if condition not in COND_LABELS or model not in MODEL_LABELS:
            continue
        data = json.loads(summary_path.read_text())
        rows.append(
            {
                "run_id": run_id,
                "condition": condition,
                "condition_label": COND_LABELS[condition],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "seed": seed,
                **data,
            }
        )
    if not rows:
        raise SystemExit("No C3 summary files found. Run evaluation first.")
    return pd.DataFrame(rows)


def add_validation(df: pd.DataFrame) -> pd.DataFrame:
    vals = []
    for row in df.itertuples(index=False):
        hist = RUN_ROOT / row.run_id / "history.jsonl"
        if not hist.exists():
            vals.append(np.nan)
            continue
        records = [json.loads(line) for line in hist.read_text().splitlines() if line.strip()]
        vals.append(max((r.get("dice_mean", np.nan) for r in records), default=np.nan))
    df = df.copy()
    df["best_val_dice"] = vals
    df["generalization_gap"] = df["best_val_dice"] - df["dice_mean"]
    return df


def save_metric_tables(df: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.sort_values(["condition", "model", "seed"]).to_csv(OUT / "journal_c3_run_metrics.csv", index=False)
    by_model = (
        df.groupby(["condition_label", "model_label"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            dice_mean=("dice_mean", "mean"),
            dice_sd=("dice_mean", "std"),
            iou_mean=("iou_mean", "mean"),
            iou_sd=("iou_mean", "std"),
            boundary_f1_mean=("boundary_f1_mean", "mean"),
            val_dice_mean=("best_val_dice", "mean"),
            gap_mean=("generalization_gap", "mean"),
        )
        .sort_values(["condition_label", "dice_mean"], ascending=[True, False])
    )
    by_model.to_csv(OUT / "journal_c3_by_model.csv", index=False)
    by_cond = (
        df.groupby("condition_label", as_index=False)
        .agg(
            runs=("run_id", "count"),
            dice_mean=("dice_mean", "mean"),
            dice_sd=("dice_mean", "std"),
            iou_mean=("iou_mean", "mean"),
            iou_sd=("iou_mean", "std"),
            boundary_f1_mean=("boundary_f1_mean", "mean"),
            val_dice_mean=("best_val_dice", "mean"),
            gap_mean=("generalization_gap", "mean"),
        )
        .sort_values("dice_mean", ascending=False)
    )
    by_cond.to_csv(OUT / "journal_c3_by_condition.csv", index=False)
    top = df.sort_values("dice_mean", ascending=False).head(8)
    top.to_csv(OUT / "journal_c3_top_runs.csv", index=False)


def save_barplots(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    colors = {"Real only": "#226E9C", "Real + inpainting": "#D95F02"}

    by_model = (
        df.groupby(["condition_label", "model_label"], as_index=False)
        .agg(dice=("dice_mean", "mean"), dice_sd=("dice_mean", "std"), iou=("iou_mean", "mean"))
    )
    order = (
        by_model[by_model.condition_label == "Real only"]
        .sort_values("dice", ascending=False)["model_label"]
        .tolist()
    )
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for offset, condition in [(-width / 2, "Real only"), (width / 2, "Real + inpainting")]:
        sub = by_model[by_model.condition_label == condition].set_index("model_label").reindex(order)
        ax.bar(x + offset, sub["dice"], width, yerr=sub["dice_sd"], capsize=2.5, label=condition, color=colors[condition])
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Cohort 3 Dice")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_journal_c3_model_dice.pdf")
    fig.savefig(OUT / "fig_journal_c3_model_dice.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for condition, marker in [("Real only", "o"), ("Real + inpainting", "s")]:
        sub = df[df.condition_label == condition]
        ax.scatter(
            sub["best_val_dice"],
            sub["dice_mean"],
            s=44,
            alpha=0.78,
            label=condition,
            color=colors[condition],
            marker=marker,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.plot([0, 1], [0, 1], color="#777777", linewidth=1, linestyle="--")
    ax.set_xlim(0.35, 1.0)
    ax.set_ylim(0.0, 0.85)
    ax.set_xlabel("Best validation Dice")
    ax.set_ylabel("Held-out Cohort 3 Dice")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color="#e0e0e0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_journal_generalization_gap.pdf")
    fig.savefig(OUT / "fig_journal_generalization_gap.png")
    plt.close(fig)


@torch.no_grad()
def save_qualitative_panel() -> None:
    checkpoint = RUN_ROOT / "pure_real__segformer_b2__seed37" / "best.pt"
    model, checkpoint_data, device = load_checkpoint(checkpoint, "cpu")
    image_size = int(checkpoint_data["image_size"])
    frame = read_manifest(ROOT / "results" / "manifests" / "c3_clean.csv")
    frame = frame[frame.split == "test"].reset_index(drop=True)
    metrics = pd.read_csv(RUN_ROOT / "pure_real__segformer_b2__seed37" / "c3_clean_metrics.csv")
    chosen = metrics.sort_values("dice", ascending=False).iloc[[2, 12, 32, 55]].name.tolist()
    selected = frame[frame["name"].isin(chosen)].head(4)

    fig, axes = plt.subplots(len(selected), 3, figsize=(6.6, 6.9))
    for row_idx, row in enumerate(selected.itertuples(index=False)):
        image = load_rgb(row.image_path)
        mask = np.asarray(load_mask_l(row.mask_path)) > 0
        arr = np.asarray(image.resize((image_size, image_size))).copy().transpose(2, 0, 1)
        tensor = torch.from_numpy(arr).float().div(255.0)[None].to(device)
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_img = Image.fromarray((prob * 255).astype(np.uint8)).resize(image.size, Image.Resampling.BILINEAR)
        pred = np.asarray(pred_img) >= 128

        rgb = np.asarray(image).copy()
        gt_overlay = rgb.copy()
        pred_overlay = rgb.copy()
        gt_overlay[mask] = (0.35 * gt_overlay[mask] + 0.65 * np.array([34, 110, 156])).astype(np.uint8)
        pred_overlay[pred] = (0.35 * pred_overlay[pred] + 0.65 * np.array([217, 95, 2])).astype(np.uint8)

        panels = [rgb, gt_overlay, pred_overlay]
        titles = ["Image", "Manual mask", "Prediction"] if row_idx == 0 else ["", "", ""]
        for col, panel in enumerate(panels):
            axes[row_idx, col].imshow(panel)
            axes[row_idx, col].set_title(titles[col], fontsize=9)
            axes[row_idx, col].axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_journal_qualitative_masks.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_validation(load_summaries())
    save_metric_tables(df)
    save_barplots(df)
    save_qualitative_panel()
    print("Wrote journal paper assets to", OUT)


if __name__ == "__main__":
    main()
