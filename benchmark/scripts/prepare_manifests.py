#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microplastic_benchmark.config import ensure_dirs, load_config
from microplastic_benchmark.manifests import write_all_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--require-generated", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    written = write_all_manifests(cfg, skip_missing_generated=not args.require_generated)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
