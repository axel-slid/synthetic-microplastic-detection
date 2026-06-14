#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from prepare_detection_track import mask_to_box_line


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare C3-clean YOLO detection evaluation data.")
    parser.add_argument("--manifest", default="results/manifests/c3_clean.csv")
    parser.add_argument("--out-dir", default="results/detection_eval/c3_clean")
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    out_dir = Path(args.out_dir)
    for row in frame.itertuples(index=False):
        image_path = Path(row.image_path)
        out_img = out_dir / "images" / "test" / image_path.name
        out_label = out_dir / "labels" / "test" / f"{image_path.stem}.txt"
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
                "train": "images/test",
                "val": "images/test",
                "test": "images/test",
                "names": {0: "microplastic"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(data_yaml)


if __name__ == "__main__":
    main()
