# %%

#!/usr/bin/env python3

import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------- #
# Dataset class
# ---------------------------------------------------------------------------- #
class MicroplasticDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, img_tf=None, mask_tf=None):
        self.image_dir = Path(root_dir) / "imgs"
        self.mask_dir  = Path(root_dir) / "masks"
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
        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.img_tf: img = self.img_tf(img)
        if self.mask_tf: mask = self.mask_tf(mask)
        mask = (mask > 0).long().squeeze(0)
        return img, mask, img_path.name


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #

def denormalise(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = img.cpu() * std + mean
    return img.permute(1,2,0).clamp(0,1).numpy()

def compute_metrics(preds, targets):
    correct = (preds == targets).sum().item()
    total = preds.numel()
    pixel_acc = correct / total

    intersection = ((preds == 1) & (targets == 1)).sum().item()
    union = ((preds == 1) | (targets == 1)).sum().item()
    iou = intersection / union if union > 0 else 0.0

    return pixel_acc, iou

def load_model(ckpt_path, device):
    model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    model.classifier[-1] = nn.Conv2d(256, 2, 1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, 2, 1)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------- #
# Main Comparison Logic
# ---------------------------------------------------------------------------- #

def compare_models(modelA, modelB, loader, device, out_dir, visualize_count=5):
    os.makedirs(out_dir, exist_ok=True)
    total_acc_A, total_iou_A = 0.0, 0.0
    total_acc_B, total_iou_B = 0.0, 0.0
    count = 0

    with torch.no_grad():
        for imgs, masks, names in tqdm(loader, desc="Comparing models"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            outA = modelA(imgs)["out"].argmax(1)
            outB = modelB(imgs)["out"].argmax(1)

            for i in range(imgs.size(0)):
                accA, iouA = compute_metrics(outA[i], masks[i])
                accB, iouB = compute_metrics(outB[i], masks[i])
                total_acc_A += accA
                total_iou_A += iouA
                total_acc_B += accB
                total_iou_B += iouB
                count += 1

                if count <= visualize_count:
                    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
                    ax[0].imshow(denormalise(imgs[i]))
                    ax[0].set_title("Image"); ax[0].axis("off")
                    ax[1].imshow(masks[i].cpu(), cmap="gray")
                    ax[1].set_title("Ground Truth"); ax[1].axis("off")
                    ax[2].imshow(outA[i].cpu(), cmap="gray")
                    ax[2].set_title("Model A"); ax[2].axis("off")
                    ax[3].imshow(outB[i].cpu(), cmap="gray")
                    ax[3].set_title("Model B"); ax[3].axis("off")
                    plt.tight_layout()
                    plt.savefig(f"{out_dir}/{names[i]}")
                    plt.close()

    print("\n=== Regular data only model ===")
    print(f"Mean Pixel Accuracy: {total_acc_A/count:.4f}")
    print(f"Mean IoU           : {total_iou_A/count:.4f}")
    print("\n=== Stabel diffusion data model ===")
    print(f"Mean Pixel Accuracy: {(total_acc_B/count):.4f}")
    print(f"Mean IoU           : {(total_iou_B/count):.4f}")


# ---------------------------------------------------------------------------- #
# Entrypoint
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    data_root = "/mnt/shared/dils/projects/microplastic/data/c3"
    modelA_path = "/mnt/shared/dils/projects/microplastic/code/base_segmentation/checkpoints/best_model.pth"
    modelB_path = "/mnt/shared/dils/projects/microplastic/code/stable_diffusion_segmentation/checkpoints/best_model.pth"
    output_dir = "comparison_output"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    img_tf = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.NEAREST),
        transforms.PILToTensor(),
    ])

    dataset = MicroplasticDataset(data_root, img_tf, mask_tf)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    modelA = load_model(modelA_path, device)
    modelB = load_model(modelB_path, device)

    compare_models(modelA, modelB, loader, device, output_dir, visualize_count=5)

# %%
