from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BinaryMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    boundary_f1: float
    area_error: float


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def boundary(mask: np.ndarray, dilation: int = 2) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=dilation)
    eroded = cv2.erode(mask, kernel, iterations=dilation)
    return (dilated - eroded) > 0


def compute_binary_metrics(pred: np.ndarray, target: np.ndarray) -> BinaryMetrics:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    iou = _safe_div(tp, union)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)

    pb = boundary(pred)
    tb = boundary(target)
    btp = int(np.logical_and(pb, tb).sum())
    bfp = int(np.logical_and(pb, ~tb).sum())
    bfn = int(np.logical_and(~pb, tb).sum())
    bp = _safe_div(btp, btp + bfp)
    br = _safe_div(btp, btp + bfn)
    bf1 = _safe_div(2 * bp * br, bp + br)
    area_error = abs(float(pred.mean()) - float(target.mean()))
    return BinaryMetrics(dice, iou, precision, recall, bf1, area_error)


def summarize_metric_rows(rows: list[dict]) -> dict[str, float]:
    keys = ["dice", "iou", "precision", "recall", "boundary_f1", "area_error"]
    summary = {}
    for key in keys:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(vals.mean()) if len(vals) else 0.0
        summary[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return summary


def bootstrap_ci(values: list[float], samples: int = 5000, seed: int = 13) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(samples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
