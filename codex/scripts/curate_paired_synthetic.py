#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def binary_components(mask: np.ndarray) -> list[int]:
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[int] = []
    h, w = mask.shape
    for y, x in zip(*np.nonzero(mask & ~seen), strict=False):
        if seen[y, x]:
            continue
        q: deque[tuple[int, int]] = deque([(int(y), int(x))])
        seen[y, x] = True
        size = 0
        while q:
            cy, cx = q.popleft()
            size += 1
            for ny in (cy - 1, cy, cy + 1):
                for nx in (cx - 1, cx, cx + 1):
                    if ny == cy and nx == cx:
                        continue
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        components.append(size)
    return components


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant")
        out = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
    return out


def score_pair(image_path: Path, mask_path: Path) -> dict[str, float | int | str | bool]:
    with Image.open(image_path) as image_raw, Image.open(mask_path) as mask_raw:
        image = np.asarray(image_raw.convert("RGB"), dtype=np.float32)
        mask = np.asarray(mask_raw.convert("L")) > 0
    area = int(mask.sum())
    total = int(mask.size)
    area_frac = area / max(1, total)
    components = binary_components(mask)
    largest = max(components) if components else 0
    largest_share = largest / max(1, area)
    ys, xs = np.nonzero(mask)
    if area:
        bbox_area = int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
        bbox_fill = area / max(1, bbox_area)
    else:
        bbox_fill = 0.0
    ring = dilate(mask, 5) & ~mask
    if area and ring.any():
        inside = image[mask].mean(axis=0)
        around = image[ring].mean(axis=0)
        contrast = float(np.abs(inside - around).mean())
    else:
        contrast = 0.0
    return {
        "name": image_path.name,
        "area_frac": area_frac,
        "bbox_fill": bbox_fill,
        "components": len(components),
        "largest_component_share": largest_share,
        "local_contrast": contrast,
    }


def link_pair(src_img: Path, src_mask: Path, dst_img: Path, dst_mask: Path) -> None:
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_mask.parent.mkdir(parents=True, exist_ok=True)
    for src, dst in ((src_img, dst_img), (src_mask, dst_mask)):
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(os.path.relpath(src.resolve(), dst.parent.resolve()), dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--out-imgs", required=True)
    parser.add_argument("--out-masks", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-area-frac", type=float, default=0.0015)
    parser.add_argument("--max-area-frac", type=float, default=0.18)
    parser.add_argument("--max-bbox-fill", type=float, default=1.0)
    parser.add_argument("--max-components", type=int, default=6)
    parser.add_argument("--min-largest-share", type=float, default=0.45)
    parser.add_argument("--min-local-contrast", type=float, default=8.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    img_dir = Path(args.imgs)
    mask_dir = Path(args.masks)
    out_imgs = Path(args.out_imgs)
    out_masks = Path(args.out_masks)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    accepted = 0
    for img_path in sorted(img_dir.glob("*.png")):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue
        row = score_pair(img_path, mask_path)
        keep = (
            float(row["area_frac"]) >= args.min_area_frac
            and float(row["area_frac"]) <= args.max_area_frac
            and float(row["bbox_fill"]) <= args.max_bbox_fill
            and int(row["components"]) <= args.max_components
            and float(row["largest_component_share"]) >= args.min_largest_share
            and float(row["local_contrast"]) >= args.min_local_contrast
        )
        row["accepted"] = keep
        rows.append(row)
        if keep and (args.limit is None or accepted < args.limit):
            link_pair(img_path, mask_path, out_imgs / img_path.name, out_masks / img_path.name)
            accepted += 1

    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "accepted",
                "area_frac",
                "bbox_fill",
                "components",
                "largest_component_share",
                "local_contrast",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"scored={len(rows)} accepted={accepted} report={report}")


if __name__ == "__main__":
    main()
