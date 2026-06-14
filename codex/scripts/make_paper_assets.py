#!/usr/bin/env python
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
CODEX = ROOT / "codex"
DATA = CODEX / "data"
RESULTS = CODEX / "results"
OUT = ROOT / "images"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)


def pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"))


def mask_coverage(mask_dir: Path, max_items: int | None = None) -> pd.DataFrame:
    files = pngs(mask_dir)
    if max_items and len(files) > max_items:
        idx = np.linspace(0, len(files) - 1, max_items).astype(int)
        files = [files[i] for i in idx]
    rows = []
    for p in files:
        arr = np.asarray(Image.open(p).convert("L"))
        rows.append({"name": p.name, "coverage": float((arr > 0).mean() * 100)})
    return pd.DataFrame(rows)


def make_dataset_counts() -> None:
    rows = [
        ("C1 labeled lab", len(pngs(DATA / "c1" / "imgs")), "real"),
        ("C2 ecological backgrounds", len(pngs(DATA / "c2" / "imgs")), "background"),
        ("C3 all test", len(pngs(DATA / "c3" / "imgs")), "test"),
        ("C3 clean primary", len(pd.read_csv(RESULTS / "manifests" / "c3_clean.csv")), "test"),
        ("Legacy SD", len(pngs(DATA / "c2" / "gen_sd")), "synthetic"),
        ("Legacy GAN", len(pngs(DATA / "c2" / "gen_gan")), "synthetic"),
        ("Legacy gen1", len(pngs(DATA / "c2" / "gen_imgs_1")), "synthetic"),
        ("Legacy c2_gen", len(pngs(DATA / "c2" / "c2_gen")), "synthetic"),
    ]
    df = pd.DataFrame(rows, columns=["cohort", "count", "type"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    palette = {"real": "#4C78A8", "background": "#72B7B2", "test": "#F58518", "synthetic": "#54A24B"}
    sns.barplot(df, x="count", y="cohort", hue="type", dodge=False, palette=palette, ax=ax)
    ax.set_xlabel("Images or image-mask pairs")
    ax.set_ylabel("")
    ax.set_xscale("log")
    ax.legend(title="")
    for patch in ax.patches:
        width = patch.get_width()
        if width > 0:
            ax.text(width * 1.05, patch.get_y() + patch.get_height() / 2, f"{int(width):,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dataset_counts.pdf")
    fig.savefig(OUT / "fig_dataset_counts.png", dpi=220)
    plt.close(fig)


def make_mask_coverage() -> None:
    frames = []
    for label, rel, n in [
        ("C1 masks", "c1/masks", None),
        ("C1 dilated", "c1/masks_dilated", None),
        ("C3 clean", "c3/masks", None),
        ("Legacy SD", "c2/gen_sd_masks", 1000),
        ("Legacy GAN", "c2/gen_gan_masks", 1000),
        ("Legacy gen1", "c2/gen_masks_1", 1000),
        ("Legacy c2_gen", "c2/c2_gen_mask", 1000),
    ]:
        df = mask_coverage(DATA / rel, n)
        if label == "C3 clean":
            clean_names = set(pd.read_csv(RESULTS / "manifests" / "c3_clean.csv")["name"])
            df = df[df["name"].isin(clean_names)]
        df["set"] = label
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    sns.violinplot(all_df, x="set", y="coverage", inner="quartile", cut=0, color="#BFD7EA", ax=ax)
    sns.stripplot(all_df, x="set", y="coverage", color="#1f2933", alpha=0.25, size=1.7, ax=ax)
    ax.set_ylabel("Foreground mask area (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mask_coverage.pdf")
    fig.savefig(OUT / "fig_mask_coverage.png", dpi=220)
    all_df.groupby("set")["coverage"].describe().round(4).to_csv(OUT / "table_mask_coverage.csv")
    plt.close(fig)


def thumb(path: Path, size=(170, 120)) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def overlay(img_path: Path, mask_path: Path, size=(170, 120)) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    mk = Image.open(mask_path).convert("L")
    if im.size == mk.size:
        rgba = im.convert("RGBA")
        red = Image.new("RGBA", im.size, (220, 20, 60, 0))
        red.putalpha(mk.point(lambda v: 120 if v > 0 else 0))
        im = Image.alpha_composite(rgba, red).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def make_qualitative_grid() -> None:
    items = [
        ("C1 lab + mask", overlay(DATA / "c1/imgs/001.png", DATA / "c1/masks/001.png")),
        ("C1 lab + mask", overlay(DATA / "c1/imgs/120.png", DATA / "c1/masks/120.png")),
        ("C2 background", thumb(DATA / "c2/imgs/0100.png")),
        ("C2 background", thumb(DATA / "c2/imgs/0600.png")),
        ("Legacy SD", overlay(DATA / "c2/gen_sd/generated_00025.png", DATA / "c2/gen_sd_masks/generated_00025.png")),
        ("Legacy GAN", overlay(DATA / "c2/gen_gan/generated_00025.png", DATA / "c2/gen_gan_masks/generated_00025.png")),
        ("Legacy gen1", overlay(DATA / "c2/gen_imgs_1/generated_00100.png", DATA / "c2/gen_masks_1/generated_00100.png")),
        ("Legacy c2_gen", overlay(DATA / "c2/c2_gen/generated_00100.png", DATA / "c2/c2_gen_mask/generated_00100.png")),
        ("C3 clean + mask", overlay(DATA / "c3/imgs/001.png", DATA / "c3/masks/001.png")),
        ("C3 clean + mask", overlay(DATA / "c3/imgs/100.png", DATA / "c3/masks/100.png")),
    ]
    cols, tw, th, label_h, pad = 5, 170, 120, 18, 12
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad, rows * (th + label_h) + (rows + 1) * pad), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (label, im) in enumerate(items):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (th + label_h + pad)
        sheet.paste(im, (x, y))
        draw.text((x, y + th + 3), label, fill=(0, 0, 0), font=font)
    sheet.save(OUT / "fig_qualitative_grid.png")


def make_run_matrix() -> None:
    df = pd.read_csv(RESULTS / "manifests/run_matrix.csv")
    pivot = df.pivot_table(index="condition", columns="model", values="seed", aggfunc="count").fillna(0)
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "seeds"}, ax=ax)
    ax.set_xlabel("Segmentation backbone/type")
    ax.set_ylabel("Training condition")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(OUT / "fig_run_matrix.pdf")
    fig.savefig(OUT / "fig_run_matrix.png", dpi=220)
    plt.close(fig)


def make_pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.axis("off")
    boxes = [
        (0.03, 0.58, "C1 labelled\nlab images\n+ masks", "#4C78A8"),
        (0.03, 0.17, "C2 ecological\nbackgrounds", "#72B7B2"),
        (0.27, 0.58, "Mask dilation\n+ affine\nplacement", "#B279A2"),
        (0.50, 0.58, "5+ generation\nconditions\nGAN / SD / SDXL /\nFLUX / gen variants", "#54A24B"),
        (0.73, 0.58, "5+ segmentation\nmodels\nCNN / transformer /\ninstance", "#ECA82C"),
        (0.73, 0.17, "C3-clean\n97-image locked\nprimary test", "#F58518"),
        (0.50, 0.17, "Metrics + CI\nDice, IoU,\nprecision/recall,\nboundary F1", "#E45756"),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), 0.19, 0.25, facecolor=color, edgecolor="#1f2933", linewidth=1.0, alpha=0.92)
        )
        ax.text(x + 0.095, y + 0.125, text, ha="center", va="center", fontsize=9, color="white", weight="bold")

    arrows = [
        ((0.22, 0.705), (0.27, 0.705)),
        ((0.22, 0.295), (0.37, 0.58)),
        ((0.46, 0.705), (0.50, 0.705)),
        ((0.69, 0.705), (0.73, 0.705)),
        ((0.825, 0.58), (0.825, 0.42)),
        ((0.73, 0.295), (0.69, 0.295)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5, color="#1f2933"))
    ax.text(
        0.50,
        0.04,
        "C3-clean is never used for generation, validation, threshold selection, or model selection.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#1f2933",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline.pdf")
    fig.savefig(OUT / "fig_pipeline.png", dpi=220)
    plt.close(fig)


def make_pilot_results() -> None:
    metrics = pd.read_csv(RESULTS / "runs/smoke_tiny_unet/c3_clean_metrics.csv")
    summary = metrics[["dice", "iou", "precision", "recall"]].mean().reset_index()
    summary.columns = ["metric", "value"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    sns.barplot(summary, x="metric", y="value", color="#E45756", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("One-epoch tiny U-Net smoke test (not benchmark result)")
    for patch in ax.patches:
        h = patch.get_height()
        ax.text(patch.get_x() + patch.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pilot_smoke_metrics.pdf")
    fig.savefig(OUT / "fig_pilot_smoke_metrics.png", dpi=220)
    plt.close(fig)


def main() -> None:
    make_dataset_counts()
    make_mask_coverage()
    make_qualitative_grid()
    make_run_matrix()
    make_pipeline_figure()
    make_pilot_results()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
