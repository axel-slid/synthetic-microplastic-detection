from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/benchmark.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("manifests", "runs", "reports"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
