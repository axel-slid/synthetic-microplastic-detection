from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import SegmentationDataset, read_manifest
from .metrics import compute_binary_metrics, summarize_metric_rows
from .models import build_model, dice_bce_loss
from .utils import append_jsonl, set_seed, write_json


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def validate_epoch(model, loader, device, threshold: float) -> dict[str, float]:
    model.eval()
    rows = []
    total_loss = 0.0
    total = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = dice_bce_loss(logits, masks)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = masks.cpu().numpy()
        for pred, target in zip(probs, targets, strict=False):
            m = compute_binary_metrics(pred[0] >= threshold, target[0] > 0)
            rows.append(m.__dict__)
        total_loss += float(loss.item()) * images.size(0)
        total += images.size(0)
    summary = summarize_metric_rows(rows)
    summary["loss"] = total_loss / max(1, total)
    return summary


def train_semantic(
    manifest_path: str | Path,
    model_spec: dict[str, Any],
    output_dir: str | Path,
    *,
    seed: int,
    image_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    threshold: float,
    device_name: str,
    num_workers: int,
    amp: bool,
) -> Path:
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {"manifest": str(manifest_path), "model": model_spec, "seed": seed})

    frame = read_manifest(manifest_path)
    train_df = frame[frame.split == "train"].reset_index(drop=True)
    val_df = frame[frame.split == "val"].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError(f"Manifest must contain non-empty train and val splits: {manifest_path}")

    device = _device(device_name)
    model = build_model(model_spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    train_loader = DataLoader(
        SegmentationDataset(train_df, image_size=image_size, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        SegmentationDataset(val_df, image_size=image_size, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    best_dice = -1.0
    bad_epochs = 0
    best_path = output_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        seen = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = dice_bce_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * images.size(0)
            seen += images.size(0)

        val = validate_epoch(model, val_loader, device, threshold)
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "seconds": time.time() - started,
            **val,
        }
        append_jsonl(output_dir / "history.jsonl", row)
        if val["dice_mean"] > best_dice:
            best_dice = val["dice_mean"]
            bad_epochs = 0
            torch.save(
                {"model_state": model.state_dict(), "model_spec": model_spec, "seed": seed, "image_size": image_size},
                best_path,
            )
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    return best_path


def load_checkpoint(path: str | Path, device_name: str):
    checkpoint = torch.load(path, map_location="cpu")
    model = build_model(checkpoint["model_spec"])
    device = _device(device_name)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint, device
