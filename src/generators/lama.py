"""
LaMa (Large Mask Inpainting with Fourier Convolutions) for microplastic synthesis.

Reference: Suvorov et al., "Resolution-robust Large Mask Inpainting with Fourier
Convolutions", ICLR 2022. https://arxiv.org/abs/2109.07161

Key ideas implemented here:
  - Fast Fourier Convolution (FFC): splits feature maps into a local (spatial)
    path and a global (spectral) path, enabling large effective receptive fields.
  - FFCResBlock: residual block built from two FFC layers.
  - Generator: encoder (stride-2 convs) -> FFC bottleneck -> decoder (bilinear up + conv).
  - Discriminator: multi-scale PatchGAN with spectral normalisation.
  - Losses: adversarial (hinge) + L1 reconstruction.

Usage:
    # Train
    python -m src.generators.lama --mode train \
        --image_dir data/c1/imgs --mask_dir data/c1/masks_dilated \
        --output_dir checkpoints/lama --epochs 200

    # Generate
    python -m src.generators.lama --mode generate \
        --checkpoint checkpoints/lama/generator.pth \
        --image_dir data/c2/imgs --mask_dir data/c1/masks_dilated \
        --output_dir data/c2/gen_lama --num_images 10000
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Fast Fourier Convolution (FFC)
# ---------------------------------------------------------------------------

class SpectralTransform(nn.Module):
    """1x1 conv applied to real + imaginary parts of the FFT spectrum."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels * 2, 1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.rfft2(x, norm="ortho")
        ri = torch.cat([fft.real, fft.imag], dim=1)   # (B, 2C, H, W//2+1)
        ri = self.fc(ri)
        real, imag = ri.chunk(2, dim=1)
        out_fft = torch.complex(real, imag)
        return torch.fft.irfft2(out_fft, s=(x.shape[2], x.shape[3]), norm="ortho")


class FFC(nn.Module):
    """Fast Fourier Convolution.

    Splits channels into a local path (standard 3x3 conv) and a global path
    (SpectralTransform). Both paths exchange information via cross-connections.
    """

    def __init__(self, in_channels: int, out_channels: int, ratio_global: float = 0.5):
        super().__init__()
        gin = round(in_channels * ratio_global)
        lin = in_channels - gin
        gout = round(out_channels * ratio_global)
        lout = out_channels - gout

        # local -> local
        self.ll = nn.Conv2d(lin, lout, 3, 1, 1, bias=False) if lin and lout else None
        # global -> local
        self.gl = nn.Conv2d(gin, lout, 1, bias=False) if gin and lout else None
        # local -> global
        self.lg = SpectralTransform(lin, gout) if lin and gout else None
        # global -> global
        self.gg = SpectralTransform(gin, gout) if gin and gout else None

        self.bn_l = nn.BatchNorm2d(lout) if lout else None
        self.bn_g = nn.BatchNorm2d(gout) if gout else None

        self.lin, self.gin = lin, gin
        self.lout, self.gout = lout, gout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xl, xg = x[:, :self.lin], x[:, self.lin:]

        out_l = sum(f(t) for f, t in [(self.ll, xl), (self.gl, xg)] if f is not None)
        out_g = sum(f(t) for f, t in [(self.lg, xl), (self.gg, xg)] if f is not None)

        if self.bn_l is not None and self.lout:
            out_l = F.relu(self.bn_l(out_l), inplace=True)
        if self.bn_g is not None and self.gout:
            out_g = F.relu(self.bn_g(out_g), inplace=True)

        return torch.cat([out_l, out_g], dim=1)


class FFCResBlock(nn.Module):
    """Two stacked FFC layers with a residual connection."""

    def __init__(self, channels: int, ratio_global: float = 0.5):
        super().__init__()
        self.conv1 = FFC(channels, channels, ratio_global)
        self.conv2 = FFC(channels, channels, ratio_global)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class LaMaGenerator(nn.Module):
    """Encoder -> FFC bottleneck -> Decoder inpainting generator."""

    def __init__(self, n_ffc_blocks: int = 9):
        super().__init__()

        def enc_block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )

        def dec_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )

        # Encoder: 512 -> 256 -> 128 -> 64 feature maps
        self.enc1 = enc_block(4, 64, stride=1)   # (B, 64, 512, 512) -- input is RGB+mask
        self.enc2 = enc_block(64, 128)             # (B, 128, 256, 256)
        self.enc3 = enc_block(128, 256)            # (B, 256, 128, 128)
        self.enc4 = enc_block(256, 512)            # (B, 512, 64, 64)

        # Bottleneck: n FFC residual blocks
        self.bottleneck = nn.Sequential(*[FFCResBlock(512) for _ in range(n_ffc_blocks)])

        # Decoder (bilinear upsample + conv)
        self.dec4 = dec_block(512 + 256, 256)
        self.dec3 = dec_block(256 + 128, 128)
        self.dec2 = dec_block(128 + 64, 64)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh()
        )

    def forward(self, x: torch.Tensor, original: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)      # (B, 64, 512, 512)
        e2 = self.enc2(e1)     # (B, 128, 256, 256)
        e3 = self.enc3(e2)     # (B, 256, 128, 128)
        e4 = self.enc4(e3)     # (B, 512, 64, 64)

        b = self.bottleneck(e4)

        d4 = self.dec4(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))

        recon = self.out_conv(d2)
        return recon * mask + original * (1.0 - mask)


# ---------------------------------------------------------------------------
# Discriminator (multi-scale, spectral norm)
# ---------------------------------------------------------------------------

class SpectralNormPatchGAN(nn.Module):
    """Single-scale PatchGAN with spectral normalisation."""

    def __init__(self, in_c: int = 3):
        super().__init__()

        def block(ic, oc, stride=2):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(ic, oc, 4, stride, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.model = nn.Sequential(
            block(in_c, 64), block(64, 128), block(128, 256), block(256, 512),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, 1, 1)),
        )

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """Two-scale discriminator: full resolution + downsampled by 2."""

    def __init__(self):
        super().__init__()
        self.d1 = SpectralNormPatchGAN()
        self.d2 = SpectralNormPatchGAN()
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.d1(x), self.d2(self.pool(x))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InpaintingDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, image_size: int = 512):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.masks = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        assert len(self.images) == len(self.masks), \
            f"Image/mask count mismatch: {len(self.images)} vs {len(self.masks)}"
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("L")

        sz = self.image_size
        image = transforms.functional.resize(image, (sz, sz), InterpolationMode.BILINEAR)
        mask = transforms.functional.resize(mask, (sz, sz), InterpolationMode.NEAREST)

        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        if random.random() < 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        img_t = transforms.functional.normalize(transforms.functional.to_tensor(image),
                                                  [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        mask_t = (transforms.functional.to_tensor(mask) > 0.5).float()
        return img_t, mask_t, self.images[idx], self.masks[idx]


# ---------------------------------------------------------------------------
# Hinge loss helpers
# ---------------------------------------------------------------------------

def hinge_d_loss(real_logits, fake_logits):
    return (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()) * 0.5


def hinge_g_loss(fake_logits):
    return -fake_logits.mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[LaMa] Using device: {device}")

    dataset = InpaintingDataset(args.image_dir, args.mask_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True)

    G = LaMaGenerator(n_ffc_blocks=args.ffc_blocks).to(device)
    D = MultiScaleDiscriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr * 4, betas=(0.0, 0.999))

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    for epoch in range(args.epochs):
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=False)
        for images, masks, *_ in loop:
            images, masks = images.to(device), masks.to(device)
            masked = images * (1.0 - masks)
            gen_in = torch.cat([masked, masks], dim=1)

            # --- Discriminator ---
            opt_D.zero_grad()
            with torch.no_grad():
                fake = G(gen_in, images, masks)
            r1, r2 = D(images)
            f1, f2 = D(fake)
            d_loss = hinge_d_loss(r1, f1) + hinge_d_loss(r2, f2)
            d_loss.backward()
            opt_D.step()

            # --- Generator ---
            opt_G.zero_grad()
            fake = G(gen_in, images, masks)
            f1, f2 = D(fake)
            g_adv = hinge_g_loss(f1) + hinge_g_loss(f2)
            g_rec = F.l1_loss(fake * masks, images * masks)
            g_loss = g_adv + args.lambda_rec * g_rec
            g_loss.backward()
            opt_G.step()

            loop.set_postfix(G=f"{g_loss.item():.4f}", D=f"{d_loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            _save_preview(G, loader, device, epoch + 1, figures_dir)

    torch.save(G.state_dict(), os.path.join(args.output_dir, "generator.pth"))
    torch.save(D.state_dict(), os.path.join(args.output_dir, "discriminator.pth"))
    print(f"[LaMa] Training complete. Weights saved to {args.output_dir}")


def _save_preview(G, loader, device, epoch, output_dir, n=6):
    G.eval()
    with torch.no_grad():
        images, masks, img_names, _ = next(iter(loader))
        images, masks = images.to(device), masks.to(device)
        masked = images * (1.0 - masks)
        generated = G(torch.cat([masked, masks], dim=1), images, masks).cpu()

    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        axes[0, i].set_title(img_names[i], fontsize=7); axes[0, i].axis("off")
        axes[1, i].imshow(masks[i].squeeze().cpu().numpy(), cmap="gray")
        axes[1, i].set_title("Mask"); axes[1, i].axis("off")
        axes[2, i].imshow(generated[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[2, i].set_title("LaMa"); axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch:03d}.png"), dpi=150)
    plt.close(fig)
    G.train()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[LaMa] Generating {args.num_images} images on {device}")

    tf_image = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tf_mask = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t > 0.5).float()),
    ])

    G = LaMaGenerator().to(device)
    G.load_state_dict(torch.load(args.checkpoint, map_location=device))
    G.eval()

    all_images = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    all_masks = [os.path.join(args.mask_dir, f) for f in os.listdir(args.mask_dir)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    os.makedirs(args.output_dir, exist_ok=True)
    mask_out_dir = args.output_dir.rstrip("/") + "_masks"
    os.makedirs(mask_out_dir, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(args.num_images), desc="LaMa generation"):
            img_t = tf_image(Image.open(random.choice(all_images)).convert("RGB")).unsqueeze(0).to(device)
            mask_path = random.choice(all_masks)
            mask_pil = Image.open(mask_path).convert("L")
            mask_t = tf_mask(mask_pil).unsqueeze(0).to(device)

            masked = img_t * (1.0 - mask_t)
            out = G(torch.cat([masked, mask_t], dim=1), img_t, mask_t)

            out_np = (out.squeeze().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
            fname = f"generated_{i:05d}.png"
            Image.fromarray((out_np * 255).astype("uint8")).save(os.path.join(args.output_dir, fname))
            mask_pil.resize((512, 512), Image.NEAREST).save(os.path.join(mask_out_dir, fname))

    print(f"[LaMa] Saved {args.num_images} images to {args.output_dir}")
    print(f"[LaMa] Saved corresponding masks to {mask_out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="LaMa inpainting for microplastic synthesis")
    p.add_argument("--mode", choices=["train", "generate"], required=True)

    p.add_argument("--image_dir", type=str, default="data/c1/imgs")
    p.add_argument("--mask_dir", type=str, default="data/c1/masks_dilated")
    p.add_argument("--device", type=str, default="cuda")

    # train
    p.add_argument("--output_dir", type=str, default="checkpoints/lama")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ffc_blocks", type=int, default=9, help="Number of FFC residual blocks in bottleneck")
    p.add_argument("--lambda_rec", type=float, default=10.0, help="L1 reconstruction loss weight")

    # generate
    p.add_argument("--checkpoint", type=str, default="checkpoints/lama/generator.pth")
    p.add_argument("--num_images", type=int, default=10000)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        generate(args)
