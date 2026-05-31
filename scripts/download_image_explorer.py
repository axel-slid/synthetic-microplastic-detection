#!/usr/bin/env python3
"""Download metadata and optional images from Microplastic Image Explorer."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


APP_URL = "https://www.openanalysis.org/microplastic_image_explorer/app.json"
IMAGE_BASE_URL = "https://d2jrxerjcsjhs7.cloudfront.net/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_bytes(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def download_to_file(url: str, path: Path, retries: int = 3, timeout: int = 120) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(path.name + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            digest = hashlib.sha256()
            total = 0
            with urllib.request.urlopen(request, timeout=timeout) as response, part_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    total += len(chunk)
                    handle.write(chunk)
            os.replace(part_path, path)
            return total, digest.hexdigest()
        except (TimeoutError, urllib.error.URLError, OSError) as error:
            last_error = error
            if part_path.exists():
                part_path.unlink()
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def decode_app_bundle(app_files: list[dict[str, Any]], metadata_dir: Path) -> list[dict[str, Any]]:
    bundle_dir = metadata_dir / "app_bundle"
    manifest: list[dict[str, Any]] = []

    for entry in app_files:
        out_path = bundle_dir / entry["name"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if entry.get("type") == "binary":
            out_path.write_bytes(base64.b64decode(entry["content"]))
        else:
            out_path.write_text(entry["content"], encoding="utf-8")
        manifest.append(
            {
                "source_name": entry["name"],
                "path": str(out_path),
                "type": entry.get("type", ""),
                "size_bytes": out_path.stat().st_size,
                "sha256": sha256_file(out_path),
            }
        )

    primary_metadata = bundle_dir / "image_metadata.csv"
    if not primary_metadata.exists():
        raise FileNotFoundError("app bundle did not contain image_metadata.csv")

    copied_metadata = metadata_dir / "image_metadata.csv"
    copied_metadata.write_bytes(primary_metadata.read_bytes())
    manifest.append(
        {
            "source_name": "image_metadata.csv",
            "path": str(copied_metadata),
            "type": "text",
            "size_bytes": copied_metadata.stat().st_size,
            "sha256": sha256_file(copied_metadata),
        }
    )
    return manifest


def load_metadata_rows(metadata_csv: Path) -> list[dict[str, str]]:
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{metadata_csv} has no rows")
    if "file_names" not in rows[0]:
        raise ValueError(f"{metadata_csv} does not contain a file_names column")
    return rows


def image_url(file_name: str) -> str:
    return IMAGE_BASE_URL + urllib.parse.quote(file_name, safe="/")


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    def counts(column: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for row in rows:
            value = (row.get(column) or "").strip() or "(blank)"
            values[value] = values.get(value, 0) + 1
        return dict(sorted(values.items(), key=lambda item: (-item[1], item[0])))

    return {
        "record_count": len(rows),
        "unique_file_names": len({row["file_names"] for row in rows if row.get("file_names")}),
        "columns": list(rows[0].keys()) if rows else [],
        "counts_by_citation": counts("citation"),
        "counts_by_morphology": counts("morphology"),
        "counts_by_polymer": counts("polymer"),
        "counts_by_color": counts("color"),
        "nonblank_size_count": sum(bool((row.get("size") or "").strip()) for row in rows),
    }


def download_image(row: dict[str, str], images_dir: Path, skip_existing: bool, retries: int) -> dict[str, Any]:
    file_name = row["file_names"]
    url = image_url(file_name)
    out_path = images_dir / file_name

    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return {
            "file_name": file_name,
            "url": url,
            "path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "sha256": sha256_file(out_path),
            "status": "existing",
        }

    size_bytes, checksum = download_to_file(url, out_path, retries=retries)
    return {
        "file_name": file_name,
        "url": url,
        "path": str(out_path),
        "size_bytes": size_bytes,
        "sha256": checksum,
        "status": "downloaded",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/microplastic_image_explorer"))
    parser.add_argument("--download-images", action="store_true", help="Download all image files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit image downloads for smoke tests.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    metadata_dir = output_dir / "metadata"
    manifests_dir = output_dir / "manifests"
    images_dir = output_dir / "images"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {APP_URL}")
    app_payload = fetch_bytes(APP_URL, retries=args.retries)
    app_json_path = metadata_dir / "app.json"
    app_json_path.write_bytes(app_payload)
    app_files = json.loads(app_payload)

    metadata_manifest = decode_app_bundle(app_files, metadata_dir)
    metadata_manifest.append(
        {
            "source_name": "app.json",
            "path": str(app_json_path),
            "type": "json",
            "size_bytes": app_json_path.stat().st_size,
            "sha256": sha256_file(app_json_path),
        }
    )
    write_csv(
        manifests_dir / "metadata_manifest.csv",
        metadata_manifest,
        ["source_name", "path", "type", "size_bytes", "sha256"],
    )

    rows = load_metadata_rows(metadata_dir / "image_metadata.csv")
    summary = summarize(rows)
    summary.update({"source_app_url": APP_URL, "image_base_url": IMAGE_BASE_URL})

    if args.download_images:
        selected_rows = rows[: args.limit] if args.limit is not None else rows
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {len(selected_rows)} images")
        image_manifest: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_image, row, images_dir, not args.no_skip_existing, args.retries): row
                for row in selected_rows
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    image_manifest.append(future.result())
                except Exception as error:
                    failures.append({"file_name": row["file_names"], "url": image_url(row["file_names"]), "error": repr(error)})
                if index % 500 == 0 or index == len(selected_rows):
                    print(f"Processed {index}/{len(selected_rows)} images; failures={len(failures)}")

        image_manifest.sort(key=lambda item: str(item["file_name"]))
        write_csv(
            manifests_dir / "image_manifest.csv",
            image_manifest,
            ["file_name", "url", "path", "size_bytes", "sha256", "status"],
        )
        if failures:
            write_csv(manifests_dir / "failed_downloads.csv", failures, ["file_name", "url", "error"])
        image_bytes = sum(int(row["size_bytes"]) for row in image_manifest)
        summary.update(
            {
                "downloaded_image_count": len(image_manifest),
                "failed_image_count": len(failures),
                "image_bytes": image_bytes,
                "image_gb_decimal": image_bytes / 1_000_000_000,
                "image_gib_binary": image_bytes / (1024**3),
            }
        )
    else:
        summary.update({"downloaded_image_count": 0, "failed_image_count": 0})

    (manifests_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
