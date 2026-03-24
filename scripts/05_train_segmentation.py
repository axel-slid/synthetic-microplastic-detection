#!/usr/bin/env python3
"""
Step 5 - Train DeepLabV3 segmentation model.

Train a binary segmentation model (background vs microplastic) on a prepared
data split.  Run once for the baseline (real data only) and once per
generative model to measure the effect of data augmentation.

Usage:
    # Baseline (real data only)
    python scripts/05_train_segmentation.py \
        --data_root data/splits/baseline \
        --output_dir checkpoints/seg_baseline \
        --samples_per_epoch 10000

    # GAN-augmented
    python scripts/05_train_segmentation.py \
        --data_root data/splits/gan \
        --output_dir checkpoints/seg_gan \
        --samples_per_epoch 10000

    # SD-augmented
    python scripts/05_train_segmentation.py \
        --data_root data/splits/sd \
        --output_dir checkpoints/seg_sd \
        --samples_per_epoch 10000

    # LaMa-augmented
    python scripts/05_train_segmentation.py \
        --data_root data/splits/lama \
        --output_dir checkpoints/seg_lama \
        --samples_per_epoch 10000

    # MAT-augmented
    python scripts/05_train_segmentation.py \
        --data_root data/splits/mat \
        --output_dir checkpoints/seg_mat \
        --samples_per_epoch 10000
"""

import sys
import os

# Allow running as a top-level script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.segmentation.model import train, build_parser

if __name__ == "__main__":
    train(build_parser().parse_args())
