from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .data import load_mask_l, load_rgb, read_manifest
from .metrics import bootstrap_ci, compute_binary_metrics, summarize_metric_rows
from .training import load_checkpoint


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_csv: str | Path,
    *,
    threshold: float,
    device_name: str,
    split: str = "test",
) -> dict[str, float]:
    model, checkpoint, device = load_checkpoint(checkpoint_path, device_name)
    image_size = int(checkpoint["image_size"])
    frame = read_manifest(manifest_path)
    frame = frame[frame.split == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows for split={split} in {manifest_path}")

    rows = []
    for row in frame.itertuples(index=False):
        image = load_rgb(row.image_path)
        original_size = image.size
        arr = np.asarray(image.resize((image_size, image_size))).copy().transpose(2, 0, 1)
        tensor = torch.from_numpy(arr).float().div(255.0)[None].to(device)
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        prob_img = Image.fromarray((prob * 255).astype(np.uint8)).resize(original_size, Image.Resampling.BILINEAR)
        pred = np.asarray(prob_img) >= int(threshold * 255)
        target = np.asarray(load_mask_l(row.mask_path)) > 0
        metrics = compute_binary_metrics(pred, target).__dict__
        rows.append({"name": getattr(row, "name", Path(row.image_path).name), **metrics})

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    summary = summarize_metric_rows(rows)
    for key in ("dice", "iou"):
        lo, hi = bootstrap_ci([r[key] for r in rows])
        summary[f"{key}_ci_low"] = lo
        summary[f"{key}_ci_high"] = hi
    return summary


def evaluate_yolo_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_csv: str | Path,
    *,
    split: str = "test",
    conf: float = 0.25,
) -> dict[str, float]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint_path))
    frame = read_manifest(manifest_path)
    frame = frame[frame.split == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows for split={split} in {manifest_path}")

    rows = []
    for row in frame.itertuples(index=False):
        image = load_rgb(row.image_path)
        target = np.asarray(load_mask_l(row.mask_path)) > 0
        result = model.predict(source=str(row.image_path), conf=conf, verbose=False)[0]
        pred = np.zeros(target.shape, dtype=bool)
        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()
            for mask in masks:
                mask_img = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
                if mask_img.size != image.size:
                    mask_img = mask_img.resize(image.size, Image.Resampling.NEAREST)
                pred |= np.asarray(mask_img) > 0
        metrics = compute_binary_metrics(pred, target).__dict__
        rows.append({"name": getattr(row, "name", Path(row.image_path).name), **metrics})

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    summary = summarize_metric_rows(rows)
    for key in ("dice", "iou"):
        lo, hi = bootstrap_ci([r[key] for r in rows])
        summary[f"{key}_ci_low"] = lo
        summary[f"{key}_ci_high"] = hi
    return summary
