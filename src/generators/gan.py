"""
GAN-based inpainting for microplastic synthetic data generation.

Architecture: U-Net generator + 70x70 PatchGAN discriminator.
Trained to inpaint microplastic regions defined by binary masks.

Usage:
    # Train
    python -m src.generators.gan --mode train \
        --image_dir data/c1/imgs --mask_dir data/c1/masks_dilated \
        --output_dir checkpoints/gan --epochs 500

    # Generate
    python -m src.generators.gan --mode generate \
        --checkpoint checkpoints/gan/generator.pth \
        --image_dir data/c2/imgs --mask_dir data/c1/masks_dilated \
        --output_dir data/c2/gen_gan --num_images 10000
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class JointTransform:
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.resize = transforms.Resize(image_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = self.resize(image)
        mask = self.resize(mask)
        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        if random.random() < 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            image = transforms.functional.affine(image, angle=0, translate=[0, 0], scale=scale, shear=0,
                                                  fill=0, interpolation=InterpolationMode.BILINEAR)
            mask = transforms.functional.affine(mask, angle=0, translate=[0, 0], scale=scale, shear=0,
                                                 fill=0, interpolation=InterpolationMode.NEAREST)
        image = self.normalize(self.to_tensor(image))
        mask = (self.to_tensor(mask) > 0.5).float()
        return image, mask


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """U-Net style encoder-decoder for inpainting masked regions."""

    def __init__(self) -> None:
        super().__init__()

        def conv(in_c, out_c):
            return nn.Sequential(nn.Conv2d(in_c, out_c, 4, 2, 1), nn.ReLU(inplace=True))

        def deconv(in_c, out_c):
            return nn.Sequential(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(conv(4, 64), conv(64, 128), conv(128, 256), conv(256, 512))
        self.decoder = nn.Sequential(
            deconv(512, 256), deconv(256, 128), deconv(128, 64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, original: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        recon = self.decoder(self.encoder(x))
        return recon * mask + original * (1.0 - mask)


class Discriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(self) -> None:
        super().__init__()

        def block(in_c, out_c, stride=2):
            return nn.Sequential(nn.Conv2d(in_c, out_c, 4, stride, 1), nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(
            block(3, 64), block(64, 128), block(128, 256), block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InpaintingDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, image_size: Tuple[int, int] = (512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.masks = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        assert len(self.images) == len(self.masks), \
            f"Image/mask count mismatch: {len(self.images)} images vs {len(self.masks)} masks"
        self.transform = JointTransform(image_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("L")
        image, mask = self.transform(image, mask)
        return image, mask, self.images[idx], self.masks[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[GAN] Using device: {device}")

    dataset = InpaintingDataset(args.image_dir, args.mask_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    opt_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    for epoch in range(args.epochs):
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=False)
        for images, masks, *_ in loop:
            images, masks = images.to(device), masks.to(device)
            masked = images * (1.0 - masks)
            gen_in = torch.cat([masked, masks], dim=1)

            # --- Generator step ---
            opt_G.zero_grad()
            generated = generator(gen_in, images, masks)
            g_loss = criterion(discriminator(generated), torch.ones_like(discriminator(generated)))
            g_loss.backward()
            opt_G.step()

            # --- Discriminator step ---
            opt_D.zero_grad()
            real_loss = criterion(discriminator(images), torch.ones_like(discriminator(images)))
            fake_loss = criterion(discriminator(generated.detach()), torch.zeros_like(discriminator(generated.detach())))
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            opt_D.step()

            loop.set_postfix(G=f"{g_loss.item():.4f}", D=f"{d_loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            _save_preview(generator, loader, device, epoch + 1, figures_dir)

    torch.save(generator.state_dict(), os.path.join(args.output_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(args.output_dir, "discriminator.pth"))
    print(f"[GAN] Training complete. Weights saved to {args.output_dir}")


def _save_preview(generator, loader, device, epoch, output_dir, n=6):
    generator.eval()
    with torch.no_grad():
        images, masks, img_names, _ = next(iter(loader))
        images, masks = images.to(device), masks.to(device)
        masked = images * (1.0 - masks)
        gen_in = torch.cat([masked, masks], dim=1)
        generated = generator(gen_in, images, masks).cpu()

    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        axes[0, i].set_title(img_names[i], fontsize=7); axes[0, i].axis("off")
        axes[1, i].imshow(masks[i].squeeze().cpu().numpy(), cmap="gray")
        axes[1, i].set_title("Mask"); axes[1, i].axis("off")
        axes[2, i].imshow(generated[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[2, i].set_title("Generated"); axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch:03d}.png"), dpi=150)
    plt.close(fig)
    generator.train()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[GAN] Generating {args.num_images} images on {device}")

    tf_image = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tf_mask = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t > 0.5).float()),
    ])

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    generator.eval()

    all_images = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    all_masks = sorted([os.path.join(args.mask_dir, f) for f in os.listdir(args.mask_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    os.makedirs(args.output_dir, exist_ok=True)
    mask_out_dir = args.output_dir.rstrip("/") + "_masks"
    os.makedirs(mask_out_dir, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(args.num_images), desc="GAN generation"):
            img_path = random.choice(all_images)
            mask_path = random.choice(all_masks)

            image = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")

            img_t = tf_image(image).unsqueeze(0).to(device)
            mask_t = tf_mask(mask_pil).unsqueeze(0).to(device)

            masked = img_t * (1.0 - mask_t)
            gen_in = torch.cat([masked, mask_t], dim=1)
            output = generator(gen_in, img_t, mask_t)

            out_np = (output.squeeze().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            out_pil = Image.fromarray((out_np * 255).astype("uint8"))
            fname = f"generated_{i:05d}.png"
            out_pil.save(os.path.join(args.output_dir, fname))
            mask_pil.resize((512, 512), Image.NEAREST).save(os.path.join(mask_out_dir, fname))

    print(f"[GAN] Saved {args.num_images} images to {args.output_dir}")
    print(f"[GAN] Saved corresponding masks to {mask_out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="GAN inpainting for microplastic synthesis")
    p.add_argument("--mode", choices=["train", "generate"], required=True)

    # shared
    p.add_argument("--image_dir", type=str, default="data/c1/imgs")
    p.add_argument("--mask_dir", type=str, default="data/c1/masks_dilated")
    p.add_argument("--device", type=str, default="cuda")

    # train
    p.add_argument("--output_dir", type=str, default="checkpoints/gan")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=4)

    # generate
    p.add_argument("--checkpoint", type=str, default="checkpoints/gan/generator.pth")
    p.add_argument("--num_images", type=int, default=10000)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        generate(args)
