from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class Pair:
    image_path: Path
    mask_path: Path
    name: str


def list_pngs(path: str | Path) -> list[Path]:
    return sorted(p for p in Path(path).glob("*") if p.suffix.lower() in IMAGE_EXTS)


def paired_by_filename(img_dir: str | Path, mask_dir: str | Path) -> tuple[list[Pair], list[str]]:
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    imgs = {p.name: p for p in list_pngs(img_dir)}
    masks = {p.name: p for p in list_pngs(mask_dir)}
    errors: list[str] = []
    for name in sorted(set(imgs) - set(masks)):
        errors.append(f"missing_mask,{name},{imgs[name]}")
    for name in sorted(set(masks) - set(imgs)):
        errors.append(f"missing_image,{name},{masks[name]}")
    pairs = [Pair(imgs[name], masks[name], name) for name in sorted(set(imgs) & set(masks))]
    return pairs, errors


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def pixel_sha256(path: str | Path) -> str:
    with Image.open(path) as im:
        return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()


def average_hash(path: str | Path, size: int = 16) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize((size, size), Image.Resampling.LANCZOS))
    return arr > arr.mean()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def validate_pairs(pairs: Iterable[Pair]) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        img_size = image_size(pair.image_path)
        mask_size = image_size(pair.mask_path)
        with Image.open(pair.mask_path) as mask:
            arr = np.asarray(mask.convert("L"))
            values = np.unique(arr)
        rows.append(
            {
                "name": pair.name,
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path),
                "image_size": f"{img_size[0]}x{img_size[1]}",
                "mask_size": f"{mask_size[0]}x{mask_size[1]}",
                "dimension_match": img_size == mask_size,
                "binary_mask": set(values.tolist()).issubset({0, 255}),
                "foreground_fraction": float((arr > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def write_manifest(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty manifest: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_path", "mask_path", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {path} missing columns: {sorted(missing)}")
    return df


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_l(path: str | Path) -> Image.Image:
    mask = Image.open(path).convert("L")
    return mask.point(lambda v: 255 if v > 0 else 0)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        image = np.asarray(load_rgb(row.image_path).resize((self.image_size, self.image_size))).copy()
        mask = np.asarray(
            load_mask_l(row.mask_path).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        ).copy()

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.ascontiguousarray(np.flip(image, axis=1))
                mask = np.ascontiguousarray(np.flip(mask, axis=1))
            if np.random.rand() < 0.5:
                image = np.ascontiguousarray(np.flip(image, axis=0))
                mask = np.ascontiguousarray(np.flip(mask, axis=0))

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy((mask > 0).astype(np.float32))[None, ...]
        return {"image": image_t, "mask": mask_t, "name": str(row.get("name", Path(row.image_path).name))}


def mask_to_polygons(mask_path: str | Path, min_area: float = 4.0) -> list[np.ndarray]:
    mask = np.asarray(load_mask_l(mask_path))
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        contour = contour.reshape(-1, 2).astype(np.float32)
        if len(contour) >= 3:
            polys.append(contour)
    return polys
