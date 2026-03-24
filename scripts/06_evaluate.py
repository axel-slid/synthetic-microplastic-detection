#!/usr/bin/env python3
"""
Step 6 - Evaluate and compare segmentation models on the held-out test set (c3).

Compares a baseline model (trained on real data only) against a model trained
with synthetic data augmentation.  Prints pixel accuracy and IoU, and saves
side-by-side visualisation images.

Usage:
    # Compare baseline vs GAN-augmented
    python scripts/06_evaluate.py \
        --model_a checkpoints/seg_baseline/best_model.pth \
        --model_b checkpoints/seg_gan/best_model.pth \
        --label_a "Baseline" --label_b "GAN" \
        --data_root data/c3 \
        --output_dir outputs/comparison_gan

    # Compare baseline vs SD-augmented
    python scripts/06_evaluate.py \
        --model_a checkpoints/seg_baseline/best_model.pth \
        --model_b checkpoints/seg_sd/best_model.pth \
        --label_a "Baseline" --label_b "Stable Diffusion" \
        --data_root data/c3 \
        --output_dir outputs/comparison_sd

    # Compare baseline vs LaMa-augmented
    python scripts/06_evaluate.py \
        --model_a checkpoints/seg_baseline/best_model.pth \
        --model_b checkpoints/seg_lama/best_model.pth \
        --label_a "Baseline" --label_b "LaMa" \
        --data_root data/c3 \
        --output_dir outputs/comparison_lama

    # Compare baseline vs MAT-augmented
    python scripts/06_evaluate.py \
        --model_a checkpoints/seg_baseline/best_model.pth \
        --model_b checkpoints/seg_mat/best_model.pth \
        --label_a "Baseline" --label_b "MAT" \
        --data_root data/c3 \
        --output_dir outputs/comparison_mat
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.segmentation.metrics import main, build_parser

if __name__ == "__main__":
    main()
