#!/usr/bin/env python3
"""
Step 4 - Prepare train/val data splits for segmentation training.

Combines real (c1) image-mask pairs with synthetic generated pairs, shuffles,
and creates an 80/20 train/val split.  Pass --real_only to create a baseline
split from real data only (no synthetic augmentation).

The output directory will have the layout:
    <output_dir>/
        train/
            imgs/
            masks/
        val/
            imgs/
            masks/

Usage:
    # Baseline split (real data only)
    python scripts/04_prepare_data.py \
        --real_imgs data/c1/imgs --real_masks data/c1/masks \
        --output_dir data/splits/baseline \
        --real_only

    # Augmented split (real + GAN-generated)
    python scripts/04_prepare_data.py \
        --real_imgs data/c1/imgs --real_masks data/c1/masks \
        --gen_imgs data/c2/gen_gan --gen_masks data/c2/gen_gan_masks \
        --output_dir data/splits/gan

    # Augmented split (real + SD-generated)
    python scripts/04_prepare_data.py \
        --real_imgs data/c1/imgs --real_masks data/c1/masks \
        --gen_imgs data/c2/gen_sd --gen_masks data/c2/gen_sd_masks \
        --output_dir data/splits/sd

    # Augmented split (real + LaMa-generated)
    python scripts/04_prepare_data.py \
        --real_imgs data/c1/imgs --real_masks data/c1/masks \
        --gen_imgs data/c2/gen_lama --gen_masks data/c2/gen_lama_masks \
        --output_dir data/splits/lama

    # Augmented split (real + MAT-generated)
    python scripts/04_prepare_data.py \
        --real_imgs data/c1/imgs --real_masks data/c1/masks \
        --gen_imgs data/c2/gen_mat --gen_masks data/c2/gen_mat_masks \
        --output_dir data/splits/mat
"""

import argparse
import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm


def collect_pairs(img_dir: Path, mask_dir: Path):
    pairs = []
    for img_file in sorted(img_dir.iterdir()):
        if not img_file.is_file():
            continue
        mask_file = mask_dir / img_file.name
        if mask_file.exists():
            pairs.append((img_file, mask_file))
        else:
            print(f"  Warning: no mask for {img_file.name}, skipping.")
    return pairs


def copy_pairs(pairs, img_out: Path, mask_out: Path, desc: str = ""):
    for img_path, mask_path in tqdm(pairs, desc=desc):
        shutil.copy(img_path, img_out / img_path.name)
        shutil.copy(mask_path, mask_out / mask_path.name)


def main():
    p = argparse.ArgumentParser(
        description="Build train/val splits for segmentation training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--real_imgs", type=str, required=True, help="Real image directory (c1/imgs)")
    p.add_argument("--real_masks", type=str, required=True, help="Real mask directory (c1/masks)")
    p.add_argument("--gen_imgs", type=str, default=None,
                   help="Generated image directory (from step 3); omit with --real_only")
    p.add_argument("--gen_masks", type=str, default=None,
                   help="Generated mask directory (from step 3); omit with --real_only")
    p.add_argument("--output_dir", type=str, required=True, help="Output split directory")
    p.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of data held out for val")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--real_only", action="store_true",
                   help="Only use real data (baseline split, no generated images)")
    args = p.parse_args()

    random.seed(args.seed)

    base = Path(args.output_dir)
    for subset in ["train", "val"]:
        (base / subset / "imgs").mkdir(parents=True, exist_ok=True)
        (base / subset / "masks").mkdir(parents=True, exist_ok=True)

    print("Collecting real pairs...")
    real_pairs = collect_pairs(Path(args.real_imgs), Path(args.real_masks))
    print(f"  Found {len(real_pairs)} real pairs")

    all_pairs = list(real_pairs)

    if not args.real_only:
        if args.gen_imgs is None or args.gen_masks is None:
            p.error("--gen_imgs and --gen_masks are required unless --real_only is set")
        print("Collecting generated pairs...")
        gen_pairs = collect_pairs(Path(args.gen_imgs), Path(args.gen_masks))
        print(f"  Found {len(gen_pairs)} generated pairs")
        all_pairs += gen_pairs

    random.shuffle(all_pairs)
    split_idx = int((1.0 - args.val_fraction) * len(all_pairs))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"\nTotal: {len(all_pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    copy_pairs(train_pairs, base / "train" / "imgs", base / "train" / "masks", desc="Copying train")
    copy_pairs(val_pairs,   base / "val"   / "imgs", base / "val"   / "masks", desc="Copying val  ")

    print(f"\nDone. Split saved to {base}")


if __name__ == "__main__":
    main()
