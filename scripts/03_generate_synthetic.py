#!/usr/bin/env python3
"""
Step 3 - Generate synthetic microplastic images using a trained generative model.

Each model takes images from --image_dir (c2 unlabelled images) and randomly
selected masks from --mask_dir (c1 dilated masks), composites synthetic
microplastics into those images, and saves:
  <output_dir>/            generated images  (generated_00000.png ...)
  <output_dir>_masks/      corresponding mask for each generated image

Usage:
    # GAN
    python scripts/03_generate_synthetic.py --model gan \
        --checkpoint checkpoints/gan/generator.pth \
        --image_dir data/c2/imgs --mask_dir data/c1/masks_dilated \
        --output_dir data/c2/gen_gan --num_images 10000

    # Stable Diffusion
    python scripts/03_generate_synthetic.py --model sd \
        --checkpoint checkpoints/sd/unet_final \
        --image_dir data/c2/imgs --mask_dir data/c1/masks_dilated \
        --output_dir data/c2/gen_sd --num_images 10000

    # LaMa
    python scripts/03_generate_synthetic.py --model lama \
        --checkpoint checkpoints/lama/generator.pth \
        --image_dir data/c2/imgs --mask_dir data/c1/masks_dilated \
        --output_dir data/c2/gen_lama --num_images 10000

    # MAT
    python scripts/03_generate_synthetic.py --model mat \
        --checkpoint checkpoints/mat/generator.pth \
        --image_dir data/c2/imgs --mask_dir data/c1/masks_dilated \
        --output_dir data/c2/gen_mat --num_images 10000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_parser():
    p = argparse.ArgumentParser(
        description="Generate synthetic microplastic images with a trained inpainting model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["gan", "sd", "lama", "mat"], required=True)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained generator weights (.pth) or directory (SD)")
    p.add_argument("--image_dir", type=str, default="data/c2/imgs",
                   help="Background images to inpaint into")
    p.add_argument("--mask_dir", type=str, default="data/c1/masks_dilated",
                   help="Mask pool (randomly sampled per generation)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for generated images")
    p.add_argument("--num_images", type=int, default=10000)
    p.add_argument("--device", type=str, default="cuda")
    # SD-specific
    p.add_argument("--inference_steps", type=int, default=50,
                   help="DDPM denoising steps (SD only)")
    return p


def main():
    args = build_parser().parse_args()

    if args.model == "gan":
        from src.generators.gan import generate, build_parser as bp
        sub = bp().parse_args([
            "--mode", "generate",
            "--checkpoint", args.checkpoint,
            "--image_dir", args.image_dir,
            "--mask_dir", args.mask_dir,
            "--output_dir", args.output_dir,
            "--num_images", str(args.num_images),
            "--device", args.device,
        ])
        generate(sub)

    elif args.model == "sd":
        from src.generators.stable_diffusion import generate, build_parser as bp
        sub = bp().parse_args([
            "--mode", "generate",
            "--checkpoint", args.checkpoint,
            "--image_dir", args.image_dir,
            "--mask_dir", args.mask_dir,
            "--output_dir", args.output_dir,
            "--num_images", str(args.num_images),
            "--inference_steps", str(args.inference_steps),
            "--device", args.device,
        ])
        generate(sub)

    elif args.model == "lama":
        from src.generators.lama import generate, build_parser as bp
        sub = bp().parse_args([
            "--mode", "generate",
            "--checkpoint", args.checkpoint,
            "--image_dir", args.image_dir,
            "--mask_dir", args.mask_dir,
            "--output_dir", args.output_dir,
            "--num_images", str(args.num_images),
            "--device", args.device,
        ])
        generate(sub)

    elif args.model == "mat":
        from src.generators.mat import generate, build_parser as bp
        sub = bp().parse_args([
            "--mode", "generate",
            "--checkpoint", args.checkpoint,
            "--image_dir", args.image_dir,
            "--mask_dir", args.mask_dir,
            "--output_dir", args.output_dir,
            "--num_images", str(args.num_images),
            "--device", args.device,
        ])
        generate(sub)


if __name__ == "__main__":
    main()
