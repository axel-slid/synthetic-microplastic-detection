#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microplastic_benchmark.config import load_config
from microplastic_benchmark.generation import generate_inpaint_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--method", required=True)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    condition = next((c for c in cfg["generation_conditions"] if c["name"] == args.method), None)
    if condition is None:
        choices = ", ".join(c["name"] for c in cfg["generation_conditions"])
        raise SystemExit(f"Unknown generation condition: {args.method}. Available: {choices}")
    if args.method not in cfg["generation_methods"]:
        choices = ", ".join(sorted(cfg["generation_methods"]))
        raise SystemExit(f"Missing generation method config for {args.method}. Available: {choices}")
    method_cfg = cfg["generation_methods"][args.method]
    generate_inpaint_set(
        method_cfg=method_cfg,
        backgrounds_dir=cfg["paths"]["c2_imgs"],
        source_images_dir=cfg["paths"]["c1_imgs"],
        source_masks_dir=cfg["paths"].get("generation_source_masks", cfg["paths"]["c1_masks"]),
        output_imgs=condition["imgs"],
        output_masks=condition["masks"],
        count=args.count,
        image_size=int(cfg["project"]["image_size"]),
        seed=args.seed,
        device=args.device or cfg["project"]["device"],
        exclude_backgrounds=set(cfg["c3_clean"]["leaked_c2_backgrounds"]),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
