# %%

#!/usr/bin/env python3


import argparse
import glob
import os
import random
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

# ---------------------------------------------------------------------------- #
# 0. Reproducibility helper
# ---------------------------------------------------------------------------- #

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------- #
# 1. Dataset
# ---------------------------------------------------------------------------- #

class MicroplasticDataset(Dataset):
    """Pairs RGB images with binary masks (0 = bg, >0 = plastic)."""

    def __init__(self, root_dir: str, subset: str, img_tf=None, mask_tf=None):
        self.image_dir = Path(root_dir) / subset / "images"
        self.mask_dir  = Path(root_dir) / subset / "masks"
        self.img_paths = sorted(self.image_dir.glob("*"))
        if not self.img_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")
        self.img_tf = img_tf
        self.mask_tf = mask_tf

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask {mask_path}")

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)           # keep single channel
        if self.img_tf:
            img = self.img_tf(img)
        if self.mask_tf:
            mask = self.mask_tf(mask)
        mask = (mask > 0).long().squeeze(0)    # (H, W)
        return img, mask


# ---------------------------------------------------------------------------- #
# 2. Visual helpers
# ---------------------------------------------------------------------------- #

def denormalise(img: torch.Tensor) -> np.ndarray:
    """Inverse of ImageNet normalisation → numpy [0,1] HxWx3."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img.cpu() * std + mean
    return img.permute(1, 2, 0).clamp(0, 1).numpy()


def visualise_first_samples(loader: DataLoader, out_file: str, num: int = 3):
    imgs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for x, y in loader:
        for i in range(x.size(0)):
            imgs.append(denormalise(x[i]))
            masks.append(y[i].cpu().numpy())
            if len(imgs) == num:
                break
        if len(imgs) == num:
            break

    fig, axes = plt.subplots(num, 2, figsize=(6, 3 * num))
    for i in range(num):
        axes[i, 0].imshow(imgs[i])
        axes[i, 0].set_title("Resized image")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(masks[i], cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)
    print(f"Saved initial visualisation → {out_file}")


def inference_visualisation(model, loader, device, out_path: str, num: int = 3):
    """Run model on *num* val images and save image/gt/pred panel."""
    model.eval()
    imgs, gts, preds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)["out"]
            pred_mask = logits.argmax(1)  # (B,H,W)
            for i in range(x.size(0)):
                imgs.append(denormalise(x[i]))
                gts.append(y[i].cpu().numpy())
                preds.append(pred_mask[i].cpu().numpy())
                if len(imgs) == num:
                    break
            if len(imgs) == num:
                break

    fig, axes = plt.subplots(num, 3, figsize=(9, 3 * num))
    for i in range(num):
        axes[i, 0].imshow(imgs[i]);  axes[i, 0].set_title("Image");  axes[i, 0].axis("off")
        axes[i, 1].imshow(gts[i], cmap="gray"); axes[i, 1].set_title("Ground‑truth"); axes[i, 1].axis("off")
        axes[i, 2].imshow(preds[i], cmap="gray"); axes[i, 2].set_title("Prediction");  axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# 3. Train / eval loops
# ---------------------------------------------------------------------------- #

def train_one_epoch(model, loader, criterion, optim, device, epoch):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optim.zero_grad()
        out = model(x)["out"]
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        running += loss.item() * x.size(0)
        pbar.set_postfix(loss=loss.item())
    return running / len(loader.sampler)


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    running = 0.0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            out = model(x)["out"]
            loss = criterion(out, y)
            running += loss.item() * x.size(0)
            pbar.set_postfix(val_loss=loss.item())
    return running / len(loader.dataset)


# ---------------------------------------------------------------------------- #
# 4. Main
# ---------------------------------------------------------------------------- #

def main(args):
    seed_everything()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------- transforms --------------------------- #
    img_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])

    train_ds = MicroplasticDataset(args.root, "train", img_tf, mask_tf)
    val_ds   = MicroplasticDataset(args.root, "val",   img_tf, mask_tf)
    print(f"Train imgs: {len(train_ds)} | Val imgs: {len(val_ds)}")

    # --- Force 10k samples/epoch via RandomSampler w/ replacement --- #
    train_sampler = RandomSampler(train_ds, replacement=True, num_samples=args.samples_per_epoch)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # initial sanity‑check visual
    if not args.no_vis:
        Path("viz").mkdir(exist_ok=True)
        visualise_first_samples(train_loader, "viz/initial_samples.jpg")

    # ------------------------------ model ---------------------------- #
    model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    model.classifier[-1] = nn.Conv2d(256, 2, 1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, 2, 1)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, 10, gamma=0.5)

    best_val = float("inf")
    Path(args.out_dir).mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device, epoch)
        val_loss   = evaluate(model, val_loader, criterion, device, epoch)
        scheduler.step()
        print(f"Epoch {epoch:03d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

        # End‑of‑epoch inference viz
        viz_file = f"viz/epoch_{epoch:03d}.jpg"
        inference_visualisation(model, val_loader, device, viz_file)
        print(f"  ↳ saved inference preview → {viz_file}")

        # checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "sched_state": scheduler.state_dict(),
                "best_val": best_val,
            }
            path = Path(args.out_dir) / "best_model.pth"
            torch.save(ckpt, path)
            print(f"  ✔ New best model saved ({best_val:.4f}) → {path}")

    print(f"Training finished. Best validation loss = {best_val:.4f}")


# ---------------------------------------------------------------------------- #
# 5. CLI
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    default_root = "/mnt/shared/dils/projects/microplastic/code/base_segmentation/split_data"

    parser = argparse.ArgumentParser(description="Train DeepLabV3 on microplastic dataset")
    parser.add_argument("--root", type=str, default=default_root,
                        help="Dataset root (with train/val)")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Resize H=W (pixels)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total training epochs")
    parser.add_argument("--samples_per_epoch", type=int, default=len(os.listdir("/mnt/shared/dils/projects/microplastic/code/base_segmentation/split_data/train/images")),
                        help="Effective #samples per epoch (duplicates via replacement)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--no_vis", action="store_true", help="Skip initial visualisation")
    args = parser.parse_args()

    main(args)
