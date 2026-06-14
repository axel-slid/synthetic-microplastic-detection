#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microplastic_benchmark.config import load_config
from microplastic_benchmark.generation import generate_text2image_set


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fully novel, unlabeled microplastic microscopy images with Stable Diffusion."
    )
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--method", default="sd_new_microplastic_text2image")
    parser.add_argument("--out", default=None)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.method not in cfg["generation_methods"]:
        choices = ", ".join(sorted(cfg["generation_methods"]))
        raise SystemExit(f"Missing generation method config for {args.method}. Available: {choices}")
    method_cfg = cfg["generation_methods"][args.method]
    output = args.out or method_cfg.get("output_imgs")
    if not output:
        raise SystemExit("--out is required unless the method config defines output_imgs")

    generate_text2image_set(
        method_cfg=method_cfg,
        output_imgs=output,
        count=args.count,
        image_size=int(cfg["project"]["image_size"]),
        seed=args.seed,
        device=args.device or cfg["project"]["device"],
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
