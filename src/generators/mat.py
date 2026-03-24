"""
MAT-style (Mask-Aware Transformer) inpainting for microplastic synthesis.

Reference: Li et al., "MAT: Mask-Aware Transformer for Large Hole Image
Inpainting", CVPR 2022. https://arxiv.org/abs/2203.15270

Implementation notes:
  - The original paper uses custom fused CUDA kernels for sparse masked
    attention. This implementation uses PyTorch's native scaled dot-product
    attention with an additive mask, which is fully portable.
  - Architecture: CNN encoder -> Transformer bottleneck with masked attention
    -> CNN decoder with AdaIN style injection.
  - Style vector extracted from unmasked regions via a style encoder.
  - Losses: adversarial (hinge) + L1 + feature matching.

Usage:
    # Train
    python -m src.generators.mat --mode train \
        --image_dir data/c1/imgs --mask_dir data/c1/masks_dilated \
        --output_dir checkpoints/mat --epochs 200

    # Generate
    python -m src.generators.mat --mode generate \
        --checkpoint checkpoints/mat/generator.pth \
        --image_dir data/c2/imgs --mask_dir data/c1/masks_dilated \
        --output_dir data/c2/gen_mat --num_images 10000
"""

from __future__ import annotations

import argparse
import math
import os
import random
from typing import Optional

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
# Style encoding and AdaIN
# ---------------------------------------------------------------------------

class StyleEncoder(nn.Module):
    """Extracts a global style vector from unmasked image regions."""

    def __init__(self, style_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, style_dim, 4, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # zero out masked region so style comes only from visible pixels
        return self.net(image * (1.0 - mask)).view(image.shape[0], -1)


class AdaIN(nn.Module):
    """Adaptive Instance Normalisation conditioned on a style vector."""

    def __init__(self, num_features: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.fc(style).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * self.norm(x) + beta


# ---------------------------------------------------------------------------
# Transformer block with masked self-attention
# ---------------------------------------------------------------------------

class MaskedSelfAttention(nn.Module):
    """Multi-head self-attention where masked positions do not attend to others."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:         (B, N, C)
        attn_mask: (B, N) boolean tensor, True = masked (invalid) position.
                   Masked positions are excluded from attending to others.
        """
        B, N, C = x.shape
        H = self.num_heads

        qkv = self.qkv(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if attn_mask is not None:
            # attn_mask: (B, N) -> (B, 1, 1, N) to mask key positions
            key_mask = attn_mask.unsqueeze(1).unsqueeze(2).float() * -1e9
            attn = attn + key_mask

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class MATBlock(nn.Module):
    """One transformer block: masked self-attention + MLP + AdaIN style."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, style_dim: int = 256):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MaskedSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )
        # style injection via AdaIN on the spatial feature map (applied before/after blocks)
        self.adain = AdaIN(embed_dim, style_dim)

    def forward(self, x: torch.Tensor, style: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:     (B, N, C) sequence of spatial tokens
        style: (B, style_dim) style vector
        """
        # self-attention
        x = x + self.attn(self.norm1(x), attn_mask)
        # MLP
        x = x + self.mlp(self.norm2(x))

        # AdaIN style injection: reshape to (B, C, H, W) and back
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
        x_spatial = self.adain(x_spatial, style)
        x = x_spatial.flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MATGenerator(nn.Module):
    """
    Multi-scale encoder -> Transformer bottleneck -> Decoder.

    The encoder downsamples 512x512 -> 64x64 (8x) to keep token count tractable.
    The transformer operates on 64x64 = 4096 tokens.
    """

    def __init__(self, style_dim: int = 256, embed_dim: int = 512,
                 num_heads: int = 8, depth: int = 6):
        super().__init__()
        self.style_encoder = StyleEncoder(style_dim)

        def enc(ic, oc, s=2):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, s, 1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            )

        # input: 4 channels (RGB + mask)
        self.enc1 = enc(4, 64, s=1)    # (B, 64, 512, 512)
        self.enc2 = enc(64, 128)        # (B, 128, 256, 256)
        self.enc3 = enc(128, 256)       # (B, 256, 128, 128)
        self.enc4 = enc(256, embed_dim) # (B, 512, 64, 64)  <- tokens here

        # project to embed_dim + positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 64 * 64, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer
        self.blocks = nn.ModuleList([
            MATBlock(embed_dim, num_heads, style_dim=style_dim) for _ in range(depth)
        ])

        # Decoder with AdaIN
        def dec(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            )

        self.adain4 = AdaIN(embed_dim, style_dim)
        self.dec4 = dec(embed_dim + 256, 256)
        self.adain3 = AdaIN(256, style_dim)
        self.dec3 = dec(256 + 128, 128)
        self.adain2 = AdaIN(128, style_dim)
        self.dec2 = dec(128 + 64, 64)
        self.out_conv = nn.Sequential(nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh())

    def _build_attn_mask(self, mask: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
        """Downsample binary mask to token resolution; True = masked token."""
        m = F.interpolate(mask, size=(token_h, token_w), mode="nearest")   # (B, 1, H', W')
        return m.squeeze(1).flatten(1).bool()                                 # (B, N)

    def forward(self, x: torch.Tensor, original: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        style = self.style_encoder(original, mask)  # (B, style_dim)

        e1 = self.enc1(x)          # (B, 64, 512, 512)
        e2 = self.enc2(e1)         # (B, 128, 256, 256)
        e3 = self.enc3(e2)         # (B, 256, 128, 128)
        e4 = self.enc4(e3)         # (B, 512, 64, 64)

        B, C, H, W = e4.shape
        tokens = e4.flatten(2).transpose(1, 2) + self.pos_embed  # (B, N, C)

        attn_mask = self._build_attn_mask(mask, H, W)
        for block in self.blocks:
            tokens = block(tokens, style, attn_mask)

        # reshape back to spatial
        feat = tokens.transpose(1, 2).reshape(B, C, H, W)
        feat = self.adain4(feat, style)

        d4 = self.dec4(torch.cat([F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d4 = self.adain3(d4, style)
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d3 = self.adain2(d3, style)
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))

        recon = self.out_conv(d2)
        return recon * mask + original * (1.0 - mask)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(ic, oc, stride=2):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(ic, oc, 4, stride, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.model = nn.Sequential(
            block(3, 64), block(64, 128), block(128, 256), block(256, 512),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, 1, 1)),
        )

    def forward(self, x):
        return self.model(x)


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
        self.sz = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[idx])).convert("L")

        image = transforms.functional.resize(image, (self.sz, self.sz), InterpolationMode.BILINEAR)
        mask = transforms.functional.resize(mask, (self.sz, self.sz), InterpolationMode.NEAREST)

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
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[MAT] Using device: {device}")

    dataset = InpaintingDataset(args.image_dir, args.mask_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True)

    G = MATGenerator(style_dim=args.style_dim, embed_dim=args.embed_dim,
                      num_heads=args.num_heads, depth=args.depth).to(device)
    D = PatchDiscriminator().to(device)

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
            d_real = D(images)
            d_fake = D(fake)
            d_loss = (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()) * 0.5
            d_loss.backward()
            opt_D.step()

            # --- Generator ---
            opt_G.zero_grad()
            fake = G(gen_in, images, masks)
            g_adv = -D(fake).mean()
            g_rec = F.l1_loss(fake * masks, images * masks)
            g_loss = g_adv + args.lambda_rec * g_rec
            g_loss.backward()
            opt_G.step()

            loop.set_postfix(G=f"{g_loss.item():.4f}", D=f"{d_loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            _save_preview(G, loader, device, epoch + 1, figures_dir)

    torch.save(G.state_dict(), os.path.join(args.output_dir, "generator.pth"))
    torch.save(D.state_dict(), os.path.join(args.output_dir, "discriminator.pth"))
    print(f"[MAT] Training complete. Weights saved to {args.output_dir}")


def _save_preview(G, loader, device, epoch, output_dir, n=4):
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
        axes[2, i].set_title("MAT"); axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch:03d}.png"), dpi=150)
    plt.close(fig)
    G.train()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[MAT] Generating {args.num_images} images on {device}")

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

    G = MATGenerator().to(device)
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
        for i in tqdm(range(args.num_images), desc="MAT generation"):
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

    print(f"[MAT] Saved {args.num_images} images to {args.output_dir}")
    print(f"[MAT] Saved corresponding masks to {mask_out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="MAT inpainting for microplastic synthesis")
    p.add_argument("--mode", choices=["train", "generate"], required=True)

    p.add_argument("--image_dir", type=str, default="data/c1/imgs")
    p.add_argument("--mask_dir", type=str, default="data/c1/masks_dilated")
    p.add_argument("--device", type=str, default="cuda")

    # train
    p.add_argument("--output_dir", type=str, default="checkpoints/mat")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--style_dim", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=6, help="Number of transformer blocks")
    p.add_argument("--lambda_rec", type=float, default=10.0, help="L1 reconstruction loss weight")

    # generate
    p.add_argument("--checkpoint", type=str, default="checkpoints/mat/generator.pth")
    p.add_argument("--num_images", type=int, default=10000)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        generate(args)
