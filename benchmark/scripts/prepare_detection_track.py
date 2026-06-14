#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microplastic_benchmark.config import load_config
from microplastic_benchmark.data import load_mask_l, paired_by_filename
from microplastic_benchmark.manifests import baseline_c1_rows, synthetic_rows


def link_or_copy(src: str | Path, dst: str | Path) -> None:
    src = Path(src).resolve()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mask_to_box_line(mask_path: str | Path, image_size: tuple[int, int]) -> str:
    width, height = image_size
    bbox = load_mask_l(mask_path).getbbox()
    if bbox is None:
        return ""
    x0, y0, x1, y1 = bbox
    xc = ((x0 + x1) / 2) / width
    yc = ((y0 + y1) / 2) / height
    bw = (x1 - x0) / width
    bh = (y1 - y0) / height
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def write_yolo_detection_dataset(rows: list[dict], out_dir: Path) -> dict[str, int]:
    frame = pd.DataFrame(rows)
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        split_df = frame[frame.split == split]
        counts[split] = len(split_df)
        for row in split_df.itertuples(index=False):
            image_path = Path(row.image_path)
            out_img = out_dir / "images" / split / image_path.name
            out_label = out_dir / "labels" / split / f"{image_path.stem}.txt"
            link_or_copy(image_path, out_img)
            with Image.open(image_path) as im:
                label = mask_to_box_line(row.mask_path, im.size)
            out_label.parent.mkdir(parents=True, exist_ok=True)
            out_label.write_text(label, encoding="utf-8")

    data_yaml = out_dir / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "microplastic"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return counts


def available_synthetic_rows(cfg: dict, condition_names: list[str]) -> dict[str, list[dict]]:
    by_name = {c["name"]: c for c in cfg["generation_conditions"]}
    rows_by_condition: dict[str, list[dict]] = {}
    for name in condition_names:
        condition = by_name.get(name)
        if not condition or condition.get("kind") not in {"paired_folder", "generated_folder"}:
            continue
        if not Path(condition["imgs"]).exists() or not Path(condition["masks"]).exists():
            continue
        pairs, errors = paired_by_filename(condition["imgs"], condition["masks"])
        if errors or not pairs:
            continue
        rows_by_condition[name] = synthetic_rows(
            condition["imgs"],
            condition["masks"],
            name,
            train_fraction=float(cfg["training"]["synthetic_train_fraction"]),
        )
    return rows_by_condition


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO detection datasets for synthetic-data ablations.")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--out-root", default="results/detection_track")
    parser.add_argument(
        "--preferred-synthetic",
        default="c2_sdxl_inpaint,legacy_sd,new_sdxl_inpaint",
        help="Comma-separated priority list for the single-synthetic and full-synthetic tracks.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    condition_names = [c["name"] for c in cfg["generation_conditions"]]
    synthetic_by_condition = available_synthetic_rows(cfg, condition_names)
    if not synthetic_by_condition:
        raise SystemExit("No mask-labeled synthetic folders are available.")

    preferred = [name.strip() for name in args.preferred_synthetic.split(",") if name.strip()]
    selected = next((name for name in preferred if name in synthetic_by_condition), sorted(synthetic_by_condition)[0])
    real_rows = baseline_c1_rows(cfg, "detection_no_synthetic")
    selected_rows = synthetic_by_condition[selected]
    c2_rows = [
        row
        for name, rows in synthetic_by_condition.items()
        if name.startswith("c2_") or name.startswith("new_")
        for row in rows
    ]
    all_synth_rows = [row for rows in synthetic_by_condition.values() for row in rows]

    tracks = {
        "no_synthetic": real_rows,
        "synthetic": real_rows + selected_rows,
        "full_synthetic": selected_rows,
        "both": real_rows + c2_rows,
        "all_together": real_rows + all_synth_rows,
    }

    summary_rows = []
    for name, rows in tracks.items():
        counts = write_yolo_detection_dataset(rows, out_root / name)
        summary_rows.append(
            {
                "track": name,
                "selected_single_synthetic": selected if name in {"synthetic", "full_synthetic"} else "",
                "train_images": counts.get("train", 0),
                "val_images": counts.get("val", 0),
                "dataset_yaml": str((out_root / name / "dataset.yaml").resolve()),
            }
        )

    summary_path = out_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(summary_path)


if __name__ == "__main__":
    main()
