"""
DeepLabV3 segmentation model for microplastic detection.

Binary segmentation: background (0) vs microplastic (1).
Pretrained ResNet-50 backbone, final classifier head replaced for 2-class output.

Usage:
    python scripts/05_train_segmentation.py \
        --data_root data/splits/baseline \
        --output_dir checkpoints/seg_baseline

    python scripts/05_train_segmentation.py \
        --data_root data/splits/gan \
        --output_dir checkpoints/seg_gan
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MicroplasticDataset(Dataset):
    """Pairs RGB images with binary masks (0=background, 1=microplastic)."""

    def __init__(self, root_dir: str, subset: str, img_tf=None, mask_tf=None):
        self.image_dir = Path(root_dir) / subset / "imgs"
        self.mask_dir = Path(root_dir) / subset / "masks"
        self.img_paths = sorted(self.image_dir.glob("*"))
        if not self.img_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")
        self.img_tf = img_tf
        self.mask_tf = mask_tf

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.img_tf:
            img = self.img_tf(img)
        if self.mask_tf:
            mask = self.mask_tf(mask)
        mask = (mask > 0).long().squeeze(0)
        return img, mask


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def denormalise(img: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img.cpu() * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()


def save_inference_preview(model, loader, device, out_path: str, n: int = 3):
    model.eval()
    imgs, gts, preds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)["out"].argmax(1)
            for i in range(x.size(0)):
                imgs.append(denormalise(x[i]))
                gts.append(y[i].cpu().numpy())
                preds.append(pred[i].cpu().numpy())
                if len(imgs) == n:
                    break
            if len(imgs) == n:
                break

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for i in range(n):
        axes[i, 0].imshow(imgs[i]); axes[i, 0].set_title("Image"); axes[i, 0].axis("off")
        axes[i, 1].imshow(gts[i], cmap="gray"); axes[i, 1].set_title("Ground Truth"); axes[i, 1].axis("off")
        axes[i, 2].imshow(preds[i], cmap="gray"); axes[i, 2].set_title("Prediction"); axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimiser, device, epoch):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()
        loss = criterion(model(x)["out"], y)
        loss.backward()
        optimiser.step()
        running += loss.item() * x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running / len(loader.sampler)


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    running = 0.0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x)["out"], y)
            running += loss.item() * x.size(0)
            pbar.set_postfix(val_loss=f"{loss.item():.4f}")
    return running / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    seed_everything()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Segmentation] Using device: {device}")
    print(f"[Segmentation] Data root: {args.data_root}")

    img_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])

    train_ds = MicroplasticDataset(args.data_root, "train", img_tf, mask_tf)
    val_ds = MicroplasticDataset(args.data_root, "val", img_tf, mask_tf)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    train_sampler = RandomSampler(train_ds, replacement=True, num_samples=args.samples_per_epoch)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    model.classifier[-1] = nn.Conv2d(256, 2, 1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, 2, 1)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    viz_dir = Path(args.output_dir) / "viz"
    viz_dir.mkdir(exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device, epoch)
        val_loss = evaluate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

        save_inference_preview(model, val_loader, device,
                               str(viz_dir / f"epoch_{epoch:03d}.jpg"))

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "sched_state": scheduler.state_dict(),
                "best_val": best_val,
            }
            ckpt_path = Path(args.output_dir) / "best_model.pth"
            torch.save(ckpt, ckpt_path)
            print(f"  -> New best saved ({best_val:.4f}): {ckpt_path}")

    print(f"Training finished. Best val loss = {best_val:.4f}")


def build_parser():
    p = argparse.ArgumentParser(description="Train DeepLabV3 segmentation on microplastic data")
    p.add_argument("--data_root", type=str, required=True, help="Split data root (with train/ val/)")
    p.add_argument("--output_dir", type=str, default="checkpoints/seg")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--samples_per_epoch", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
