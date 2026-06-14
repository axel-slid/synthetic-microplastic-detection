from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .data import paired_by_filename, write_manifest


def _rows_from_pairs(pairs, split: str, source: str, condition: str) -> list[dict]:
    return [
        {
            "name": pair.name,
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
            "split": split,
            "source": source,
            "condition": condition,
        }
        for pair in pairs
    ]


def baseline_c1_rows(cfg: dict[str, Any], condition: str) -> list[dict]:
    split_root = Path(cfg["paths"]["data_root"]) / "splits" / "baseline"
    if (split_root / "train" / "imgs").exists():
        train, train_errors = paired_by_filename(split_root / "train" / "imgs", split_root / "train" / "masks")
        val, val_errors = paired_by_filename(split_root / "val" / "imgs", split_root / "val" / "masks")
        if train_errors or val_errors:
            raise ValueError(f"Baseline split pairing errors: {train_errors + val_errors}")
        return _rows_from_pairs(train, "train", "c1", condition) + _rows_from_pairs(val, "val", "c1", condition)

    pairs, errors = paired_by_filename(cfg["paths"]["c1_imgs"], cfg["paths"]["c1_masks"])
    if errors:
        raise ValueError(f"C1 pairing errors: {errors}")
    rng = random.Random(13)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    cut = int(round(0.8 * len(shuffled)))
    return _rows_from_pairs(shuffled[:cut], "train", "c1", condition) + _rows_from_pairs(
        shuffled[cut:], "val", "c1", condition
    )


def synthetic_rows(
    img_dir: str | Path,
    mask_dir: str | Path,
    condition: str,
    *,
    train_fraction: float,
    seed: int = 13,
) -> list[dict]:
    if not Path(img_dir).exists() or not Path(mask_dir).exists():
        raise FileNotFoundError(f"Missing synthetic folder for {condition}: {img_dir} or {mask_dir}")
    pairs, errors = paired_by_filename(img_dir, mask_dir)
    if errors:
        raise ValueError(f"Synthetic pairing errors for {condition}: {errors[:20]}")
    if not pairs:
        raise ValueError(f"No synthetic pairs found for {condition}: {img_dir} / {mask_dir}")
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    cut = int(round(train_fraction * len(shuffled)))
    return _rows_from_pairs(shuffled[:cut], "train", "synthetic", condition) + _rows_from_pairs(
        shuffled[cut:], "val", "synthetic", condition
    )


def c3_rows(cfg: dict[str, Any], clean: bool) -> list[dict]:
    pairs, errors = paired_by_filename(cfg["paths"]["c3_imgs"], cfg["paths"]["c3_masks"])
    if errors:
        raise ValueError(f"C3 pairing errors: {errors}")
    excluded = set(cfg.get("c3_clean", {}).get("exclude_primary", [])) if clean else set()
    rows = []
    for pair in pairs:
        if pair.name in excluded:
            continue
        rows.append(
            {
                "name": pair.name,
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path),
                "split": "test",
                "source": "c3_clean" if clean else "c3_all",
                "condition": "test",
            }
        )
    return rows


def build_condition_manifest(cfg: dict[str, Any], condition: dict[str, Any]) -> list[dict]:
    name = condition["name"]
    rows = baseline_c1_rows(cfg, name)
    if condition["kind"] in {"paired_folder", "generated_folder"}:
        rows += synthetic_rows(
            condition["imgs"],
            condition["masks"],
            name,
            train_fraction=float(cfg["training"]["synthetic_train_fraction"]),
        )
    elif condition["kind"] != "c1_only":
        raise ValueError(f"Unknown generation condition kind: {condition}")
    return rows


def write_all_manifests(cfg: dict[str, Any], *, skip_missing_generated: bool = True) -> list[Path]:
    out_dir = Path(cfg["paths"]["manifests"])
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for clean in (True, False):
        path = out_dir / ("c3_clean.csv" if clean else "c3_all.csv")
        write_manifest(c3_rows(cfg, clean=clean), path)
        written.append(path)
    for condition in cfg["generation_conditions"]:
        path = out_dir / f"{condition['name']}.csv"
        try:
            write_manifest(build_condition_manifest(cfg, condition), path)
            written.append(path)
        except (FileNotFoundError, ValueError):
            if condition["kind"] != "generated_folder" or not skip_missing_generated:
                raise
            if path.exists():
                path.unlink()
    return written


def planned_runs(cfg: dict[str, Any], *, require_manifest: bool = False) -> list[dict]:
    runs = []
    for condition in cfg["generation_conditions"]:
        manifest = Path(cfg["paths"]["manifests"]) / f"{condition['name']}.csv"
        if require_manifest and not manifest.exists():
            continue
        for model in cfg["segmentation_models"]:
            for seed in cfg["project"]["seed_values"]:
                run_id = f"{condition['name']}__{model['name']}__seed{seed}"
                runs.append(
                    {
                        "run_id": run_id,
                        "condition": condition["name"],
                        "model": model["name"],
                        "family": model["family"],
                        "seed": seed,
                        "manifest": str(manifest),
                        "output_dir": str(Path(cfg["paths"]["runs"]) / run_id),
                    }
                )
    return runs
