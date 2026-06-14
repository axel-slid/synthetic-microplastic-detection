#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microplastic_benchmark.config import load_config
from microplastic_benchmark.data import average_hash, hamming, paired_by_filename, pixel_sha256, validate_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--out", default="results/reports/data_validation.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)

    checks = [
        ("c1", cfg["paths"]["c1_imgs"], cfg["paths"]["c1_masks"]),
        ("c1_dilated", cfg["paths"]["c1_imgs"], cfg["paths"]["c1_masks_dilated"]),
        ("c3", cfg["paths"]["c3_imgs"], cfg["paths"]["c3_masks"]),
    ]
    frames = []
    for dataset, imgs, masks in checks:
        pairs, errors = paired_by_filename(imgs, masks)
        frame = validate_pairs(pairs)
        frame.insert(0, "dataset", dataset)
        frame["pairing_error_count"] = len(errors)
        frames.append(frame)
        if errors:
            print(f"{dataset}: {len(errors)} pairing errors")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)

    c2 = sorted(Path(cfg["paths"]["c2_imgs"]).glob("*.png"))
    c3 = sorted(Path(cfg["paths"]["c3_imgs"]).glob("*.png"))
    c2_pixel = {pixel_sha256(p): p.name for p in c2}
    exact = [(p.name, c2_pixel[pixel_sha256(p)]) for p in c3 if pixel_sha256(p) in c2_pixel]
    c2_hashes = [(p.name, average_hash(p)) for p in c2]
    near = []
    for p in c3:
        ah = average_hash(p)
        best = min((hamming(ah, c2h), name) for name, c2h in c2_hashes)
        if best[0] <= 8:
            near.append((p.name, best[1], best[0]))

    print(f"Wrote {out}")
    print(f"Exact C3/C2 duplicates: {exact}")
    print(f"Near C3/C2 duplicates, ahash<=8: {near}")


if __name__ == "__main__":
    main()
