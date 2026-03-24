#!/usr/bin/env python3
"""
Step 2 - Train a generative inpainting model.

Dispatches to one of four model implementations:
  gan  - U-Net generator + PatchGAN discriminator (BCE adversarial loss)
  sd   - Stable Diffusion 2 inpainting fine-tuned on c1 data
  lama - Large Mask Inpainting with Fourier Convolutions (ICLR 2022)
  mat  - Mask-Aware Transformer inpainting (CVPR 2022)

All models share the same training interface:
  --image_dir  path to RGB training images
  --mask_dir   path to corresponding binary masks (dilated)
  --output_dir where to save checkpoints and training figures

Usage:
    # GAN
    python scripts/02_train_generator.py --model gan \
        --image_dir data/c1/imgs --mask_dir data/c1/masks_dilated \
        --output_dir checkpoints/gan --epochs 500

    # Stable Diffusion
    python scripts/02_train_generator.py --model sd \
        --image_dir data/c1/imgs --mask_dir data/c1/masks_dilated \
        --output_dir checkpoints/sd --epochs 100

    # LaMa
    python scripts/02_train_generator.py --model lama \
        --image_dir data/c1/imgs --mask_dir data/c1/masks_dilated \
        --output_dir checkpoints/lama --epochs 200

    # MAT
    python scripts/02_train_generator.py --model mat \
        --image_dir data/c1/imgs --mask_dir data/c1/masks_dilated \
        --output_dir checkpoints/mat --epochs 200
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_parser():
    p = argparse.ArgumentParser(
        description="Train a generative inpainting model for microplastic synthesis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["gan", "sd", "lama", "mat"], required=True,
                   help="Which generative model to train")
    p.add_argument("--image_dir", type=str, default="data/c1/imgs")
    p.add_argument("--mask_dir", type=str, default="data/c1/masks_dilated")
    p.add_argument("--output_dir", type=str, help="Checkpoint output directory (defaults per model)")
    p.add_argument("--epochs", type=int, help="Training epochs (defaults per model)")
    p.add_argument("--batch_size", type=int, help="Batch size (defaults per model)")
    p.add_argument("--lr", type=float, help="Learning rate (defaults per model)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")

    # GAN-specific
    p.add_argument("--gan_epochs", type=int, default=500)

    # SD-specific
    p.add_argument("--sd_epochs", type=int, default=100)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--resolution", type=int, default=512)

    # LaMa-specific
    p.add_argument("--lama_epochs", type=int, default=200)
    p.add_argument("--ffc_blocks", type=int, default=9)
    p.add_argument("--lambda_rec", type=float, default=10.0)

    # MAT-specific
    p.add_argument("--mat_epochs", type=int, default=200)
    p.add_argument("--style_dim", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=6)

    return p


def main():
    args = build_parser().parse_args()

    if args.model == "gan":
        from src.generators.gan import train, build_parser as bp
        sub = bp().parse_args([
            "--mode", "train",
            "--image_dir", args.image_dir,
            "--mask_dir", args.mask_dir,
            "--output_dir", args.output_dir or "checkpoints/gan",
            "--epochs", str(args.epochs or args.gan_epochs),
            "--batch_size", str(args.batch_size or 16),
            "--lr", str(args.lr or 2e-4),
            "--workers", str(args.workers),
            "--device", args.device,
        ])
        train(sub)

    elif args.model == "sd":
        from src.generators.stable_diffusion import train, build_parser as bp
        sub = bp().parse_args([
            "--mode", "train",
            "--image_dir", args.image_dir,
            "--mask_dir", args.mask_dir,
            "--output_dir", args.output_dir or "checkpoints/sd",
            "--epochs", str(args.epochs or args.sd_epochs),
            "--batch_size", str(args.batch_size or 1),
            "--lr", str(args.lr or 1e-5),
            "--grad_accum", str(args.grad_accum),
            "--mixed_precision", args.mixed_precision,
            "--resolution", str(args.resolution),
            "--workers", str(args.workers),
            "--device", args.device,
        ])
        train(sub)

    elif args.model == "lama":
        from src.generators.lama import train, build_parser as bp
        sub = bp().parse_args([
            "--mode", "train",
            "--image_dir", args.image_dir,
            "--mask_dir", args.mask_dir,
            "--output_dir", args.output_dir or "checkpoints/lama",
            "--epochs", str(args.epochs or args.lama_epochs),
            "--batch_size", str(args.batch_size or 8),
            "--lr", str(args.lr or 1e-4),
            "--workers", str(args.workers),
            "--ffc_blocks", str(args.ffc_blocks),
            "--lambda_rec", str(args.lambda_rec),
            "--device", args.device,
        ])
        train(sub)

    elif args.model == "mat":
        from src.generators.mat import train, build_parser as bp
        sub = bp().parse_args([
            "--mode", "train",
            "--image_dir", args.image_dir,
            "--mask_dir", args.mask_dir,
            "--output_dir", args.output_dir or "checkpoints/mat",
            "--epochs", str(args.epochs or args.mat_epochs),
            "--batch_size", str(args.batch_size or 4),
            "--lr", str(args.lr or 1e-4),
            "--workers", str(args.workers),
            "--style_dim", str(args.style_dim),
            "--embed_dim", str(args.embed_dim),
            "--num_heads", str(args.num_heads),
            "--depth", str(args.depth),
            "--lambda_rec", str(args.lambda_rec),
            "--device", args.device,
        ])
        train(sub)


if __name__ == "__main__":
    main()
