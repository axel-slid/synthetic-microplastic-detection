from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    """Small built-in model for smoke tests when optional libraries are unavailable."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.down1 = ConvBlock(3, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = ConvBlock(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        mid = self.mid(self.pool2(d2))
        u2 = self.up2(mid)
        u2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, d1], dim=1))
        return self.head(u1)


class SegFormerBinaryWrapper(nn.Module):
    def __init__(self, model_id: str = "nvidia/segformer-b2-finetuned-ade-512-512") -> None:
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_id,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x).logits
        return torch.nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    library: str | None = None
    architecture: str | None = None
    encoder: str | None = None
    weights: str | None = None
    model_id: str | None = None


def build_model(spec: dict[str, Any]) -> nn.Module:
    library = spec.get("library")
    name = spec["name"]
    if name == "tiny_unet" or library == "builtin":
        return TinyUNet()

    if library == "smp":
        import segmentation_models_pytorch as smp

        cls = getattr(smp, spec["architecture"])
        return cls(
            encoder_name=spec["encoder"],
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

    if library == "monai":
        from monai.networks.nets import UNet

        return UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    if library == "transformers":
        return SegFormerBinaryWrapper(spec.get("model_id") or "nvidia/segformer-b2-finetuned-ade-512-512")

    raise ValueError(f"Unsupported semantic model spec: {spec}")


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = 1 - ((2 * intersection + 1.0) / (denom + 1.0)).mean()
    return bce + dice
