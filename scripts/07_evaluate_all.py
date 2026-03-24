#!/usr/bin/env python3
"""
Step 7 — Comprehensive evaluation of all trained segmentation models.

Auto-discovers checkpoints at <ckpt_dir>/seg_*/best_model.pth and evaluates
every available model against the test set in one pass, then produces a full
suite of charts and a self-contained HTML report.

Figures produced
----------------
  fig01_metrics_overview.png       Grouped bar chart: all metrics × all models
  fig02_iou_distribution.png       Per-image IoU box-plots + strip overlay
  fig03_roc_curves.png             ROC curves (pixel-level) for all models
  fig04_pr_curves.png              Precision-Recall curves for all models
  fig05_confusion_matrices.png     Normalised confusion matrices side-by-side
  fig06_qualitative_comparison.png Image / GT / predictions grid
  fig07_radar_chart.png            Spider chart — all metrics, all models
  fig08_confidence_heatmaps.png    Probability-map overlays on test images
  fig09_improvement_over_baseline.png  Delta vs baseline for each metric
  fig10_per_image_scatter.png      IoU vs Dice scatter coloured by model
  fig11_summary_table.png          Formatted metrics table
  fig12_error_analysis.png         Best / worst predictions per model

  metrics.json                     All numeric results in machine-readable form
  report.html                      Self-contained HTML report (figures base64-embedded)

Usage
-----
    python scripts/07_evaluate_all.py \\
        --data_root  data/c3 \\
        --ckpt_dir   checkpoints \\
        --output_dir outputs/evaluation \\
        --device     cuda
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

try:
    from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[Warning] scikit-learn not found — ROC/PR curves will be skipped. "
          "Install with: pip install scikit-learn")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "baseline": "#6c757d",
    "gan":      "#1f77b4",
    "sd":       "#9467bd",
    "lama":     "#ff7f0e",
    "mat":      "#2ca02c",
}
MODEL_LABELS = {
    "baseline": "Baseline",
    "gan":      "GAN",
    "sd":       "Stable Diffusion",
    "lama":     "LaMa",
    "mat":      "MAT",
}
METRICS_DISPLAY = {
    "pixel_acc": "Pixel Acc.",
    "iou":       "IoU",
    "dice":      "Dice",
    "precision": "Precision",
    "recall":    "Recall",
    "f1":        "F1",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.prop_cycle": plt.cycler(color=list(MODEL_COLORS.values())),
})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 512):
        self.image_dir = Path(root_dir) / "imgs"
        self.mask_dir  = Path(root_dir) / "masks"
        self.img_paths = sorted(self.image_dir.glob("*"))
        if not self.img_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")
        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img  = Image.open(p).convert("RGB")
        mask = Image.open(self.mask_dir / p.name)
        return self.img_tf(img), (self.mask_tf(mask) > 0).long().squeeze(0), p.name


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_seg_model(ckpt_path: str, device: torch.device) -> nn.Module:
    model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    model.classifier[-1] = nn.Conv2d(256, 2, 1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, 2, 1)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def discover_models(ckpt_dir: str) -> Dict[str, str]:
    """Return {model_name: ckpt_path} for every seg checkpoint found."""
    found = {}
    for p in sorted(Path(ckpt_dir).glob("seg_*/best_model.pth")):
        name = p.parent.name.replace("seg_", "")
        found[name] = str(p)
    return found


# ---------------------------------------------------------------------------
# Per-image metric helpers
# ---------------------------------------------------------------------------

def compute_image_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    tp = ((pred == 1) & (gt == 1)).sum().item()
    fp = ((pred == 1) & (gt == 0)).sum().item()
    fn = ((pred == 0) & (gt == 1)).sum().item()
    tn = ((pred == 0) & (gt == 0)).sum().item()

    pixel_acc = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice      = f1  # equivalent for binary segmentation
    return dict(pixel_acc=pixel_acc, precision=precision, recall=recall,
                f1=f1, iou=iou, dice=dice, tp=tp, fp=fp, fn=fn, tn=tn)


def denorm(t: torch.Tensor) -> np.ndarray:
    """ImageNet denormalisation -> HxWx3 float32 [0,1]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t.cpu() * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()


# ---------------------------------------------------------------------------
# Inference — collect all results for one model
# ---------------------------------------------------------------------------

def run_inference(model_name: str, ckpt_path: str, loader: DataLoader,
                  device: torch.device, n_viz_samples: int = 16,
                  prob_stride: int = 4) -> dict:
    """Run full test-set inference for one model. Returns results dict."""
    model = load_seg_model(ckpt_path, device)

    per_image   : List[dict] = []
    all_probs   : List[np.ndarray] = []   # downsampled, for ROC/PR
    all_gts_flat: List[np.ndarray] = []
    conf_pred   : List[np.ndarray] = []
    conf_gt     : List[np.ndarray] = []
    viz_samples : List[dict] = []         # first n_viz_samples images

    with torch.no_grad():
        for imgs, masks, names in tqdm(loader, desc=f"  Inferring [{model_name}]", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)["out"]                    # (B, 2, H, W)
            probs  = logits.softmax(1)[:, 1]               # (B, H, W)  prob of class 1
            preds  = logits.argmax(1)                      # (B, H, W)

            for i in range(imgs.size(0)):
                m = compute_image_metrics(preds[i].cpu(), masks[i].cpu())
                m["name"] = names[i]
                per_image.append(m)

                # Flatten (downsampled) for ROC/PR
                p_sub = probs[i].cpu().numpy()[::prob_stride, ::prob_stride].ravel()
                g_sub = masks[i].cpu().numpy()[::prob_stride, ::prob_stride].ravel()
                all_probs.append(p_sub)
                all_gts_flat.append(g_sub)

                conf_pred.append(preds[i].cpu().numpy().ravel())
                conf_gt.append(masks[i].cpu().numpy().ravel())

                if len(viz_samples) < n_viz_samples:
                    viz_samples.append({
                        "image": denorm(imgs[i]),
                        "gt":    masks[i].cpu().numpy().astype(np.uint8),
                        "pred":  preds[i].cpu().numpy().astype(np.uint8),
                        "prob":  probs[i].cpu().numpy(),
                        "name":  names[i],
                    })

    # Aggregate metrics
    keys = ["pixel_acc", "iou", "dice", "precision", "recall", "f1"]
    agg  = {k: float(np.mean([r[k] for r in per_image])) for k in keys}

    # Confusion matrix
    cp = np.concatenate(conf_pred)
    cg = np.concatenate(conf_gt)
    if HAS_SKLEARN:
        cm = confusion_matrix(cg, cp, labels=[0, 1])
    else:
        tn = ((cp == 0) & (cg == 0)).sum()
        fp = ((cp == 1) & (cg == 0)).sum()
        fn = ((cp == 0) & (cg == 1)).sum()
        tp = ((cp == 1) & (cg == 1)).sum()
        cm = np.array([[tn, fp], [fn, tp]])

    # ROC / PR
    roc_data = pr_data = None
    if HAS_SKLEARN:
        probs_cat = np.concatenate(all_probs)
        gts_cat   = np.concatenate(all_gts_flat)
        if gts_cat.sum() > 0:
            fpr, tpr, _ = roc_curve(gts_cat, probs_cat)
            roc_auc     = auc(fpr, tpr)
            pre, rec, _ = precision_recall_curve(gts_cat, probs_cat)
            pr_auc      = auc(rec, pre)
            roc_data    = dict(fpr=fpr.tolist(), tpr=tpr.tolist(), auc=roc_auc)
            pr_data     = dict(precision=pre.tolist(), recall=rec.tolist(), auc=pr_auc)
            agg["roc_auc"] = roc_auc
            agg["pr_auc"]  = pr_auc

    return dict(
        name=model_name,
        metrics=agg,
        per_image=per_image,
        confusion_matrix=cm.tolist(),
        roc=roc_data,
        pr=pr_data,
        viz=viz_samples,
    )


# ===========================================================================
# Figure functions
# ===========================================================================

def _savefig(fig, path: str):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {Path(path).name}")


# ---------------------------------------------------------------------------
# Fig 01 — Metrics overview bar chart
# ---------------------------------------------------------------------------
def fig01_metrics_overview(results: dict, out: str):
    models  = list(results.keys())
    metrics = list(METRICS_DISPLAY.keys())
    n_m, n_g = len(metrics), len(models)
    x   = np.arange(n_m)
    w   = 0.8 / n_g
    fig, ax = plt.subplots(figsize=(14, 6))

    for gi, model in enumerate(models):
        vals  = [results[model]["metrics"].get(k, 0.0) for k in metrics]
        color = MODEL_COLORS.get(model, "#333333")
        bars  = ax.bar(x + gi * w - (n_g - 1) * w / 2, vals, w * 0.9,
                       label=MODEL_LABELS.get(model, model), color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([METRICS_DISPLAY[k] for k in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Segmentation Metrics — All Models", fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Fig 02 — Per-image IoU box plots
# ---------------------------------------------------------------------------
def fig02_iou_distribution(results: dict, out: str):
    models = list(results.keys())
    data   = [[r["iou"] for r in results[m]["per_image"]] for m in models]
    colors = [MODEL_COLORS.get(m, "#333") for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for i, (d, color) in enumerate(zip(data, colors)):
        jitter = np.random.uniform(-0.15, 0.15, len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, s=15, alpha=0.4,
                   color=color, zorder=3)

    ax.set_xticklabels(labels)
    ax.set_ylabel("IoU (per image)")
    ax.set_title("Per-Image IoU Distribution", fontweight="bold", pad=12)
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Fig 03 — ROC curves
# ---------------------------------------------------------------------------
def fig03_roc_curves(results: dict, out: str):
    if not HAS_SKLEARN:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Random")
    for model, res in results.items():
        roc = res.get("roc")
        if roc is None:
            continue
        ax.plot(roc["fpr"], roc["tpr"],
                color=MODEL_COLORS.get(model, "#333"),
                lw=2, label=f"{MODEL_LABELS.get(model, model)}  AUC={roc['auc']:.3f}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (pixel-level)", fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Fig 04 — PR curves
# ---------------------------------------------------------------------------
def fig04_pr_curves(results: dict, out: str):
    if not HAS_SKLEARN:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for model, res in results.items():
        pr = res.get("pr")
        if pr is None:
            continue
        ax.plot(pr["recall"], pr["precision"],
                color=MODEL_COLORS.get(model, "#333"),
                lw=2, label=f"{MODEL_LABELS.get(model, model)}  AUC={pr['auc']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (pixel-level)", fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Fig 05 — Confusion matrices
# ---------------------------------------------------------------------------
def fig05_confusion_matrices(results: dict, out: str):
    models = list(results.keys())
    n = len(models)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        cm = np.array(results[model]["confusion_matrix"], dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({int(cm[i,j])})",
                        ha="center", va="center", fontsize=10,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred BG", "Pred MP"]); ax.set_yticklabels(["GT BG", "GT MP"])
        ax.set_title(MODEL_LABELS.get(model, model), fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Confusion Matrices (row-normalised)", fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Fig 06 — Qualitative comparison grid
# ---------------------------------------------------------------------------
def fig06_qualitative_comparison(results: dict, out: str, n_rows: int = 8):
    models  = list(results.keys())
    n_cols  = 2 + len(models)   # image + GT + one pred per model
    samples = results[models[0]]["viz"]
    n_rows  = min(n_rows, len(samples))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    col_titles = ["Image", "Ground Truth"] + [MODEL_LABELS.get(m, m) for m in models]

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontweight="bold", fontsize=9)

    for r in range(n_rows):
        # image
        axes[r, 0].imshow(results[models[0]]["viz"][r]["image"])
        axes[r, 0].axis("off")
        # GT
        axes[r, 1].imshow(results[models[0]]["viz"][r]["gt"], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].axis("off")
        # each model's prediction
        for ci, model in enumerate(models):
            viz = results[model]["viz"]
            if r >= len(viz):
                axes[r, 2 + ci].axis("off")
                continue
            pred = viz[r]["pred"]
            iou  = compute_image_metrics(
                torch.tensor(pred), torch.tensor(viz[r]["gt"])
            )["iou"]
            axes[r, 2 + ci].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[r, 2 + ci].set_title(f"IoU={iou:.3f}", fontsize=7, pad=2)
            axes[r, 2 + ci].axis("off")

    fig.suptitle("Qualitative Comparison — Predictions vs Ground Truth",
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Fig 07 — Radar / spider chart
# ---------------------------------------------------------------------------
def fig07_radar_chart(results: dict, out: str):
    metric_keys   = ["pixel_acc", "iou", "dice", "precision", "recall", "f1"]
    metric_labels = [METRICS_DISPLAY[k] for k in metric_keys]
    n = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, alpha=0.6)
    ax.grid(alpha=0.4)

    handles = []
    for model, res in results.items():
        vals = [res["metrics"].get(k, 0.0) for k in metric_keys]
        vals += vals[:1]
        color = MODEL_COLORS.get(model, "#333")
        ax.plot(angles, vals, "o-", lw=2, color=color, markersize=4)
        ax.fill(angles, vals, alpha=0.08, color=color)
        handles.append(mpatches.Patch(color=color, label=MODEL_LABELS.get(model, model)))

    ax.legend(handles=handles, loc="upper right",
              bbox_to_anchor=(1.35, 1.15), framealpha=0.9)
    ax.set_title("Model Comparison — Radar Chart", fontweight="bold", pad=20)
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Fig 08 — Confidence heatmaps
# ---------------------------------------------------------------------------
def fig08_confidence_heatmaps(results: dict, out: str, n_rows: int = 6):
    models   = list(results.keys())
    n_cols   = 2 + len(models)   # image + GT + one heatmap per model
    cmap_hot = LinearSegmentedColormap.from_list("mp", ["#000000", "#ff4444", "#ffff00"])

    n_rows = min(n_rows, len(results[models[0]]["viz"]))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    col_titles = ["Image", "Ground Truth"] + [f"{MODEL_LABELS.get(m,m)}\nP(microplastic)"
                                               for m in models]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontweight="bold", fontsize=8)

    for r in range(n_rows):
        base = results[models[0]]["viz"][r]
        axes[r, 0].imshow(base["image"]); axes[r, 0].axis("off")
        axes[r, 1].imshow(base["gt"], cmap="gray", vmin=0, vmax=1); axes[r, 1].axis("off")
        for ci, model in enumerate(models):
            viz = results[model]["viz"]
            if r >= len(viz):
                axes[r, 2 + ci].axis("off")
                continue
            prob = viz[r]["prob"]
            img  = viz[r]["image"]
            # overlay: blend image with probability heatmap
            hmap = cmap_hot(prob)[..., :3]
            overlay = 0.55 * img + 0.45 * hmap
            axes[r, 2 + ci].imshow(overlay.clip(0, 1))
            axes[r, 2 + ci].axis("off")

    fig.suptitle("Confidence Heatmaps — P(microplastic) overlaid on image",
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Fig 09 — Improvement over baseline
# ---------------------------------------------------------------------------
def fig09_improvement_over_baseline(results: dict, out: str):
    if "baseline" not in results:
        print("  [Skip fig09] No baseline model found.")
        return

    metrics = list(METRICS_DISPLAY.keys())
    other   = [m for m in results if m != "baseline"]
    if not other:
        return

    n_g = len(other)
    x   = np.arange(len(metrics))
    w   = 0.8 / n_g
    baseline_vals = {k: results["baseline"]["metrics"].get(k, 0.0) for k in metrics}

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhline(0, color="black", lw=1.2)

    for gi, model in enumerate(other):
        deltas = [results[model]["metrics"].get(k, 0.0) - baseline_vals[k] for k in metrics]
        color  = MODEL_COLORS.get(model, "#333")
        bars   = ax.bar(x + gi * w - (n_g - 1) * w / 2, deltas, w * 0.9,
                        label=MODEL_LABELS.get(model, model), color=color, alpha=0.85)
        for bar, d in zip(bars, deltas):
            va = "bottom" if d >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.002 if d >= 0 else -0.002),
                    f"{d:+.3f}", ha="center", va=va, fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([METRICS_DISPLAY[k] for k in metrics])
    ax.set_ylabel("Δ vs Baseline")
    ax.set_title("Improvement Over Baseline (higher = better)", fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Fig 10 — Per-image IoU vs Dice scatter
# ---------------------------------------------------------------------------
def fig10_per_image_scatter(results: dict, out: str):
    fig, ax = plt.subplots(figsize=(8, 7))
    for model, res in results.items():
        ious  = [r["iou"]  for r in res["per_image"]]
        dices = [r["dice"] for r in res["per_image"]]
        ax.scatter(ious, dices, s=18, alpha=0.45,
                   color=MODEL_COLORS.get(model, "#333"),
                   label=MODEL_LABELS.get(model, model))

    # identity line
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1)
    ax.set_xlabel("IoU"); ax.set_ylabel("Dice (F1)")
    ax.set_title("Per-Image IoU vs Dice", fontweight="bold", pad=12)
    ax.legend(framealpha=0.9)
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Fig 11 — Summary table
# ---------------------------------------------------------------------------
def fig11_summary_table(results: dict, out: str):
    metric_keys = list(METRICS_DISPLAY.keys())
    # add AUC columns if available
    if any("roc_auc" in r["metrics"] for r in results.values()):
        metric_keys += ["roc_auc", "pr_auc"]
    col_labels = [METRICS_DISPLAY.get(k, k.upper()) for k in metric_keys]
    row_labels  = [MODEL_LABELS.get(m, m) for m in results]
    cell_data   = [[f"{results[m]['metrics'].get(k, float('nan')):.4f}"
                    for k in metric_keys]
                   for m in results]

    # find best value per column (max is better for all these metrics)
    best_per_col = []
    for ci, k in enumerate(metric_keys):
        vals = [results[m]["metrics"].get(k, float("-inf")) for m in results]
        best_per_col.append(max(vals))

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(metric_keys)), 1.5 + 0.55 * len(results)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    # colour header row
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # colour row labels and highlight best cells
    for ri, model in enumerate(results):
        tbl[ri + 1, -1].set_facecolor(MODEL_COLORS.get(model, "#eee"))
        tbl[ri + 1, -1].set_alpha(0.35)
        for ci, k in enumerate(metric_keys):
            val = results[model]["metrics"].get(k, float("-inf"))
            if abs(val - best_per_col[ci]) < 1e-9:
                tbl[ri + 1, ci].set_facecolor("#d5f5e3")
                tbl[ri + 1, ci].set_text_props(fontweight="bold")

    ax.set_title("Metrics Summary  (green = best per column)",
                 fontweight="bold", pad=12, fontsize=12)
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Fig 12 — Error analysis: best and worst per model
# ---------------------------------------------------------------------------
def fig12_error_analysis(results: dict, out: str, n_each: int = 4):
    models  = list(results.keys())
    n_rows  = len(models)
    n_cols  = 1 + 2 * n_each  # label + best N + worst N

    fig = plt.figure(figsize=(3.5 * n_cols, 3.5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.15)

    for ri, model in enumerate(models):
        pi = sorted(results[model]["per_image"], key=lambda x: x["iou"])
        worst_names = [x["name"] for x in pi[:n_each]]
        best_names  = [x["name"] for x in pi[-n_each:][::-1]]
        viz_dict    = {v["name"]: v for v in results[model]["viz"]}

        # Model name label axis
        ax_lbl = fig.add_subplot(gs[ri, 0])
        ax_lbl.axis("off")
        ax_lbl.text(0.5, 0.5, MODEL_LABELS.get(model, model),
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    rotation=90, transform=ax_lbl.transAxes)

        for ci_offset, (names, tag, cmap) in enumerate([
            (best_names,  "BEST",  "Greens"),
            (worst_names, "WORST", "Reds"),
        ]):
            col_start = 1 + ci_offset * n_each
            for ni, name in enumerate(names):
                col = col_start + ni
                ax  = fig.add_subplot(gs[ri, col])
                if name in viz_dict:
                    v   = viz_dict[name]
                    iou = next(x["iou"] for x in results[model]["per_image"] if x["name"] == name)
                    # blend pred onto image
                    img  = v["image"]
                    pred = v["pred"]
                    overlay = img.copy()
                    overlay[pred == 1] = overlay[pred == 1] * 0.4 + np.array([0.2, 0.8, 0.2]) * 0.6
                    ax.imshow(overlay.clip(0, 1))
                    ax.set_title(f"IoU={iou:.3f}", fontsize=7,
                                 color="green" if tag == "BEST" else "red", pad=2)
                else:
                    ax.text(0.5, 0.5, "no sample", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color="gray")
                ax.axis("off")
                if ri == 0:
                    col_title = f"{tag} #{ni+1}"
                    ax.set_title(col_title + f"\n(IoU={iou:.3f})" if name in viz_dict else col_title,
                                 fontsize=7, color="green" if tag == "BEST" else "red", pad=2)

    fig.suptitle("Error Analysis — Best and Worst Predictions per Model (green overlay = predicted MP)",
                 fontweight="bold", y=1.01, fontsize=11)
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_html_report(results: dict, fig_paths: dict, metrics_json: str, out_html: str):
    metric_keys = list(METRICS_DISPLAY.keys())
    if any("roc_auc" in r["metrics"] for r in results.values()):
        metric_keys += ["roc_auc", "pr_auc"]

    # Build metrics table HTML
    header_cells = "".join(f"<th>{METRICS_DISPLAY.get(k, k.upper())}</th>" for k in metric_keys)
    rows_html = ""
    best_per_col = {}
    for k in metric_keys:
        vals = {m: results[m]["metrics"].get(k, float("-inf")) for m in results}
        best_per_col[k] = max(vals.values())

    for model, res in results.items():
        color = MODEL_COLORS.get(model, "#ccc")
        cells = ""
        for k in metric_keys:
            v = res["metrics"].get(k, float("nan"))
            is_best = abs(v - best_per_col[k]) < 1e-9
            style   = 'style="background:#d5f5e3;font-weight:bold;"' if is_best else ""
            cells += f"<td {style}>{v:.4f}</td>"
        rows_html += f"""
        <tr>
          <td style="background:{color}33;font-weight:bold;">{MODEL_LABELS.get(model, model)}</td>
          {cells}
        </tr>"""

    # Build figure sections
    sections = ""
    fig_meta = [
        ("fig01", "Fig 1 — Metrics Overview",        "Grouped bar chart of all metrics for all models."),
        ("fig02", "Fig 2 — IoU Distribution",         "Per-image IoU box-plots showing score spread."),
        ("fig03", "Fig 3 — ROC Curves",               "Pixel-level receiver operating characteristic curves."),
        ("fig04", "Fig 4 — Precision-Recall Curves",  "Pixel-level precision-recall curves."),
        ("fig05", "Fig 5 — Confusion Matrices",       "Row-normalised confusion matrices for each model."),
        ("fig06", "Fig 6 — Qualitative Comparison",   "Sample test images with ground-truth and predictions."),
        ("fig07", "Fig 7 — Radar Chart",              "Spider chart comparing all metrics simultaneously."),
        ("fig08", "Fig 8 — Confidence Heatmaps",      "P(microplastic) overlaid as heatmap on test images."),
        ("fig09", "Fig 9 — Improvement Over Baseline","Delta metrics vs baseline for each augmentation model."),
        ("fig10", "Fig 10 — IoU vs Dice Scatter",     "Per-image IoU vs Dice coloured by model."),
        ("fig11", "Fig 11 — Summary Table",           "Formatted metrics table (green = best per column)."),
        ("fig12", "Fig 12 — Error Analysis",          "Best and worst predictions per model."),
    ]
    for key, title, caption in fig_meta:
        if key not in fig_paths or not Path(fig_paths[key]).exists():
            continue
        b64 = _b64(fig_paths[key])
        sections += f"""
        <section>
          <h2>{title}</h2>
          <p class="caption">{caption}</p>
          <img src="data:image/png;base64,{b64}" alt="{title}">
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Microplastic Segmentation — Evaluation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f8f9fa; color: #212529; line-height: 1.6; }}
  header {{ background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white; padding: 2.5rem 2rem; text-align: center; }}
  header h1 {{ font-size: 2rem; margin-bottom: 0.4rem; }}
  header p  {{ opacity: 0.75; font-size: 0.95rem; }}
  main {{ max-width: 1400px; margin: 2rem auto; padding: 0 1.5rem; }}
  section {{ background: white; border-radius: 10px; padding: 1.8rem;
             margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  section h2 {{ font-size: 1.2rem; color: #1a1a2e; border-bottom: 2px solid #e9ecef;
                padding-bottom: 0.5rem; margin-bottom: 0.8rem; }}
  .caption {{ color: #6c757d; font-size: 0.88rem; margin-bottom: 1rem; }}
  img {{ max-width: 100%; border-radius: 6px; display: block; margin: 0 auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin-top: 0.5rem; }}
  th, td {{ padding: 0.6rem 0.8rem; text-align: center; border: 1px solid #dee2e6; }}
  th {{ background: #2c3e50; color: white; }}
  tr:hover {{ background: #f1f3f5; }}
  .models-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                  gap: 1rem; margin-top: 0.8rem; }}
  .model-card {{ padding: 1rem; border-radius: 8px; text-align: center;
                 border: 2px solid #dee2e6; }}
  .model-card .name {{ font-weight: bold; margin-bottom: 0.3rem; }}
  .model-card .iou  {{ font-size: 1.5rem; font-weight: bold; }}
  .model-card .sub  {{ font-size: 0.8rem; color: #6c757d; }}
  footer {{ text-align: center; padding: 2rem; color: #6c757d; font-size: 0.85rem; }}
</style>
</head>
<body>
<header>
  <h1>Microplastic Segmentation — Evaluation Report</h1>
  <p>Comprehensive comparison of {len(results)} segmentation models on the held-out test cohort (c3)</p>
</header>

<main>

<!-- ── Quick summary ── -->
<section>
  <h2>Quick Summary</h2>
  <p class="caption">Mean IoU on the test set for each model.</p>
  <div class="models-grid">
{"".join(f'''    <div class="model-card" style="border-color:{MODEL_COLORS.get(m,'#ccc')}">
      <div class="name">{MODEL_LABELS.get(m,m)}</div>
      <div class="iou" style="color:{MODEL_COLORS.get(m,'#333')}">{results[m]['metrics'].get('iou',0):.3f}</div>
      <div class="sub">IoU &nbsp;|&nbsp; Dice {results[m]['metrics'].get('dice',0):.3f}</div>
    </div>''' for m in results)}
  </div>
</section>

<!-- ── Metrics table ── -->
<section>
  <h2>Full Metrics Table</h2>
  <p class="caption">Green highlights indicate the best-performing model per metric.</p>
  <table>
    <thead><tr><th>Model</th>{header_cells}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</section>

{sections}

</main>
<footer>
  Generated by <code>scripts/07_evaluate_all.py</code> &nbsp;|&nbsp;
  Metrics saved to <code>metrics.json</code>
</footer>
</body>
</html>
"""
    with open(out_html, "w") as f:
        f.write(html)
    print(f"  Saved: report.html  ({Path(out_html).stat().st_size // 1024} KB)")


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="Comprehensive evaluation of all segmentation models")
    p.add_argument("--data_root",  type=str, default="data/c3")
    p.add_argument("--ckpt_dir",   type=str, default="checkpoints")
    p.add_argument("--output_dir", type=str, default="outputs/evaluation")
    p.add_argument("--img_size",   type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--n_viz",      type=int, default=16,
                   help="Number of test images to use for visualisation figures")
    p.add_argument("--prob_stride", type=int, default=4,
                   help="Spatial stride for downsampling pixels when computing ROC/PR "
                        "(1=full resolution, 4=use 1/16 of pixels — much faster)")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n[Eval] Device : {device}")

    # Discover models
    available = discover_models(args.ckpt_dir)
    if not available:
        print(f"[Eval] No checkpoints found under {args.ckpt_dir}/seg_*/best_model.pth")
        return
    print(f"[Eval] Found {len(available)} model(s): {list(available.keys())}")

    # Build dataloader
    dataset = TestDataset(args.data_root, args.img_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
    print(f"[Eval] Test set : {len(dataset)} images in {args.data_root}")

    # Output dirs
    out_dir  = Path(args.output_dir)
    fig_dir  = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Run inference for each model
    # ---------------------------------------------------------------------------
    results = {}
    for name, ckpt in available.items():
        print(f"\n[Eval] Evaluating: {name}  ({ckpt})")
        results[name] = run_inference(name, ckpt, loader, device,
                                      n_viz_samples=args.n_viz,
                                      prob_stride=args.prob_stride)

    # Sort so baseline appears first
    order   = ["baseline"] + [k for k in results if k != "baseline"]
    results = {k: results[k] for k in order if k in results}

    # ---------------------------------------------------------------------------
    # Save metrics JSON
    # ---------------------------------------------------------------------------
    metrics_out = str(out_dir / "metrics.json")
    serialisable = {
        m: {
            "metrics":          res["metrics"],
            "per_image_mean":   {k: float(np.mean([r[k] for r in res["per_image"]]))
                                 for k in ["pixel_acc","iou","dice","precision","recall","f1"]},
            "per_image_std":    {k: float(np.std([r[k] for r in res["per_image"]]))
                                 for k in ["pixel_acc","iou","dice","precision","recall","f1"]},
        }
        for m, res in results.items()
    }
    with open(metrics_out, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n[Eval] Metrics JSON saved to {metrics_out}")

    # ---------------------------------------------------------------------------
    # Generate all figures
    # ---------------------------------------------------------------------------
    print("\n[Eval] Generating figures...")
    def fp(name): return str(fig_dir / name)

    fig_paths = {
        "fig01": fp("fig01_metrics_overview.png"),
        "fig02": fp("fig02_iou_distribution.png"),
        "fig03": fp("fig03_roc_curves.png"),
        "fig04": fp("fig04_pr_curves.png"),
        "fig05": fp("fig05_confusion_matrices.png"),
        "fig06": fp("fig06_qualitative_comparison.png"),
        "fig07": fp("fig07_radar_chart.png"),
        "fig08": fp("fig08_confidence_heatmaps.png"),
        "fig09": fp("fig09_improvement_over_baseline.png"),
        "fig10": fp("fig10_per_image_scatter.png"),
        "fig11": fp("fig11_summary_table.png"),
        "fig12": fp("fig12_error_analysis.png"),
    }

    fig01_metrics_overview(results,          fig_paths["fig01"])
    fig02_iou_distribution(results,          fig_paths["fig02"])
    fig03_roc_curves(results,                fig_paths["fig03"])
    fig04_pr_curves(results,                 fig_paths["fig04"])
    fig05_confusion_matrices(results,        fig_paths["fig05"])
    fig06_qualitative_comparison(results,    fig_paths["fig06"])
    fig07_radar_chart(results,               fig_paths["fig07"])
    fig08_confidence_heatmaps(results,       fig_paths["fig08"])
    fig09_improvement_over_baseline(results, fig_paths["fig09"])
    fig10_per_image_scatter(results,         fig_paths["fig10"])
    fig11_summary_table(results,             fig_paths["fig11"])
    fig12_error_analysis(results,            fig_paths["fig12"])

    # ---------------------------------------------------------------------------
    # HTML report
    # ---------------------------------------------------------------------------
    print("\n[Eval] Generating HTML report...")
    generate_html_report(results, fig_paths, metrics_out, str(out_dir / "report.html"))

    # ---------------------------------------------------------------------------
    # Print final table to console
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  {'Model':<22}", end="")
    for k in ["pixel_acc", "iou", "dice", "f1"]:
        print(f"  {METRICS_DISPLAY[k]:>10}", end="")
    print()
    print("-" * 70)
    for model, res in results.items():
        m = res["metrics"]
        print(f"  {MODEL_LABELS.get(model, model):<22}", end="")
        for k in ["pixel_acc", "iou", "dice", "f1"]:
            print(f"  {m.get(k,0):>10.4f}", end="")
        print()
    print("=" * 70)
    print(f"\n[Eval] All outputs in: {out_dir.resolve()}")
    print(f"[Eval] Open report  : {(out_dir / 'report.html').resolve()}\n")


if __name__ == "__main__":
    main()
