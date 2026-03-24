"""
Evaluation metrics and model comparison for segmentation models.

Computes pixel accuracy and IoU for two DeepLabV3 models side-by-side
on a held-out test set and saves comparison visualisations.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dataset (flat directory: imgs/ + masks/ at root)
# ---------------------------------------------------------------------------

class TestDataset(Dataset):
    def __init__(self, root_dir: str, img_tf=None, mask_tf=None):
        self.image_dir = Path(root_dir) / "imgs"
        self.mask_dir = Path(root_dir) / "masks"
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
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.img_tf:
            img = self.img_tf(img)
        if self.mask_tf:
            mask = self.mask_tf(mask)
        mask = (mask > 0).long().squeeze(0)
        return img, mask, img_path.name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def denormalise(img: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img.cpu() * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()


def pixel_accuracy_and_iou(preds: torch.Tensor, targets: torch.Tensor):
    pixel_acc = (preds == targets).float().mean().item()
    intersection = ((preds == 1) & (targets == 1)).sum().item()
    union = ((preds == 1) | (targets == 1)).sum().item()
    iou = intersection / union if union > 0 else 0.0
    return pixel_acc, iou


def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    model.classifier[-1] = nn.Conv2d(256, 2, 1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, 2, 1)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_models(model_a, model_b, loader, device, out_dir: str,
                   label_a: str = "Model A", label_b: str = "Model B",
                   visualize_count: int = 5):
    os.makedirs(out_dir, exist_ok=True)
    stats = {"acc_a": 0.0, "iou_a": 0.0, "acc_b": 0.0, "iou_b": 0.0, "count": 0}

    with torch.no_grad():
        for imgs, masks, names in tqdm(loader, desc="Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)
            out_a = model_a(imgs)["out"].argmax(1)
            out_b = model_b(imgs)["out"].argmax(1)

            for i in range(imgs.size(0)):
                acc_a, iou_a = pixel_accuracy_and_iou(out_a[i], masks[i])
                acc_b, iou_b = pixel_accuracy_and_iou(out_b[i], masks[i])
                stats["acc_a"] += acc_a
                stats["iou_a"] += iou_a
                stats["acc_b"] += acc_b
                stats["iou_b"] += iou_b
                stats["count"] += 1

                if stats["count"] <= visualize_count:
                    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
                    ax[0].imshow(denormalise(imgs[i])); ax[0].set_title("Image"); ax[0].axis("off")
                    ax[1].imshow(masks[i].cpu(), cmap="gray"); ax[1].set_title("Ground Truth"); ax[1].axis("off")
                    ax[2].imshow(out_a[i].cpu(), cmap="gray"); ax[2].set_title(label_a); ax[2].axis("off")
                    ax[3].imshow(out_b[i].cpu(), cmap="gray"); ax[3].set_title(label_b); ax[3].axis("off")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, names[i]), dpi=150)
                    plt.close(fig)

    n = stats["count"]
    print(f"\n{'='*50}")
    print(f"  {label_a}")
    print(f"    Pixel Accuracy: {stats['acc_a']/n:.4f}")
    print(f"    Mean IoU      : {stats['iou_a']/n:.4f}")
    print(f"\n  {label_b}")
    print(f"    Pixel Accuracy: {stats['acc_b']/n:.4f}")
    print(f"    Mean IoU      : {stats['iou_b']/n:.4f}")
    print(f"{'='*50}\n")

    return {
        label_a: {"pixel_acc": stats["acc_a"] / n, "iou": stats["iou_a"] / n},
        label_b: {"pixel_acc": stats["acc_b"] / n, "iou": stats["iou_b"] / n},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Compare two segmentation models on a test set")
    p.add_argument("--model_a", type=str, required=True, help="Checkpoint path for model A (baseline)")
    p.add_argument("--model_b", type=str, required=True, help="Checkpoint path for model B (augmented)")
    p.add_argument("--label_a", type=str, default="Baseline")
    p.add_argument("--label_b", type=str, default="Augmented")
    p.add_argument("--data_root", type=str, default="data/c3", help="Test data root (imgs/ + masks/)")
    p.add_argument("--output_dir", type=str, default="outputs/comparison")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--visualize_count", type=int, default=5,
                   help="Number of side-by-side comparison images to save")
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Metrics] Using device: {device}")

    img_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=Image.NEAREST),
        transforms.PILToTensor(),
    ])

    dataset = TestDataset(args.data_root, img_tf, mask_tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print(f"Loading models...")
    model_a = load_model(args.model_a, device)
    model_b = load_model(args.model_b, device)

    results = compare_models(model_a, model_b, loader, device, args.output_dir,
                              label_a=args.label_a, label_b=args.label_b,
                              visualize_count=args.visualize_count)
    return results


if __name__ == "__main__":
    main()
