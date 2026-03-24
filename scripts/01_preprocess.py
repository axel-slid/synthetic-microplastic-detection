#!/usr/bin/env python3
"""
Step 1 - Preprocess masks: dilate binary masks to create wider inpainting regions.

Dilation makes masks slightly larger than the actual annotation, ensuring the
generative model inpaints over the full microplastic extent.

Usage:
    python scripts/01_preprocess.py \
        --src data/c1/masks \
        --dst data/c1/masks_dilated \
        --kernel_size 3 \
        --iterations 4
"""

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def dilate_masks(src_folder: str, dst_folder: str, kernel_size: int = 3, iterations: int = 4):
    os.makedirs(dst_folder, exist_ok=True)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    filenames = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    skipped = 0

    for filename in tqdm(filenames, desc="Dilating masks"):
        src_path = os.path.join(src_folder, filename)
        mask = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  Skipping {filename}: not a valid image.")
            skipped += 1
            continue
        dilated = cv2.dilate(mask, kernel, iterations=iterations)
        cv2.imwrite(os.path.join(dst_folder, filename), dilated)

    total = len(filenames) - skipped
    print(f"Done. Dilated {total} masks -> {dst_folder}  (skipped {skipped})")


def main():
    p = argparse.ArgumentParser(description="Dilate binary segmentation masks")
    p.add_argument("--src", type=str, required=True, help="Source mask directory")
    p.add_argument("--dst", type=str, required=True, help="Destination directory for dilated masks")
    p.add_argument("--kernel_size", type=int, default=3, help="Dilation kernel size (default: 3)")
    p.add_argument("--iterations", type=int, default=4, help="Dilation iterations (default: 4)")
    args = p.parse_args()

    if not os.path.isdir(args.src):
        raise FileNotFoundError(f"Source directory not found: {args.src}")

    print(f"Dilating masks: {args.src} -> {args.dst}")
    print(f"  kernel={args.kernel_size}x{args.kernel_size}, iterations={args.iterations}")
    dilate_masks(args.src, args.dst, args.kernel_size, args.iterations)


if __name__ == "__main__":
    main()
