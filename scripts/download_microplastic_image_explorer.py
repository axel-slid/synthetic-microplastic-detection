#!/usr/bin/env python3
"""Download the OpenAnalysis Microplastic Image Explorer dataset.

The Shinylive app publishes its metadata in app.json. The image metadata file
lists image file names stored behind a CloudFront base URL.
"""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APP_URL = "https://www.openanalysis.org/microplastic_image_explorer/app.json"
IMAGE_BASE_URL = "https://d2jrxerjcsjhs7.cloudfront.net/"


@dataclass(frozen=True)
class ImageRecord:
    file_name: str
    url: str


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
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def download_to_file(url: str, path: Path, retries: int = 3, timeout: int = 120) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            digest = hashlib.sha256()
            total = 0
            with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    digest.update(chunk)
                    handle.write(chunk)
            os.replace(tmp, path)
            return total, digest.hexdigest()
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if tmp.exists():
                tmp.unlink()
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def write_app_bundle(app_files: list[dict[str, Any]], metadata_dir: Path) -> list[dict[str, str | int]]:
    manifest: list[dict[str, str | int]] = []
    bundle_dir = metadata_dir / "app_bundle"
    for entry in app_files:
        rel_path = Path(entry["name"])
        out_path = bundle_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if entry.get("type") == "binary":
            payload = base64.b64decode(entry["content"])
            out_path.write_bytes(payload)
        else:
            payload = entry["content"].encode("utf-8")
            out_path.write_bytes(payload)

        manifest.append(
            {
                "path": str(out_path),
                "source_name": entry["name"],
                "type": entry.get("type", ""),
                "size_bytes": out_path.stat().st_size,
                "sha256": sha256_file(out_path),
            }
        )

    image_metadata = bundle_dir / "image_metadata.csv"
    if image_metadata.exists():
        target = metadata_dir / "image_metadata.csv"
        target.write_bytes(image_metadata.read_bytes())
        manifest.append(
            {
                "path": str(target),
                "source_name": "image_metadata.csv",
                "type": "text",
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    return manifest


def load_image_records(metadata_csv: Path, limit: int | None = None) -> list[ImageRecord]:
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if "file_names" not in rows[0]:
        raise ValueError(f"{metadata_csv} does not include a file_names column")

    records = [
        ImageRecord(
            file_name=row["file_names"],
            url=IMAGE_BASE_URL + urllib.parse.quote(row["file_names"], safe="/"),
        )
        for row in rows
        if row.get("file_names")
    ]
    if limit is not None:
        return records[:limit]
    return records


def download_one(record: ImageRecord, images_dir: Path, skip_existing: bool, retries: int) -> dict[str, str | int]:
    out_path = images_dir / record.file_name
    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return {
            "file_name": record.file_name,
            "url": record.url,
            "path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "sha256": sha256_file(out_path),
            "status": "existing",
        }

    size_bytes, checksum = download_to_file(record.url, out_path, retries=retries)
    return {
        "file_name": record.file_name,
        "url": record.url,
        "path": str(out_path),
        "size_bytes": size_bytes,
        "sha256": checksum,
        "status": "downloaded",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_image_metadata(metadata_csv: Path) -> dict[str, Any]:
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

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
        "counts_by_size": counts("size"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/microplastic_image_explorer"))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N images for testing.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Redownload files that already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    metadata_dir = output_dir / "metadata"
    manifests_dir = output_dir / "manifests"
    images_dir = output_dir / "images"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching app bundle: {APP_URL}")
    app_payload = fetch_bytes(APP_URL)
    app_json_path = metadata_dir / "app.json"
    app_json_path.write_bytes(app_payload)
    app_files = json.loads(app_payload)

    metadata_manifest = write_app_bundle(app_files, metadata_dir)
    metadata_manifest.append(
        {
            "path": str(app_json_path),
            "source_name": "app.json",
            "type": "json",
            "size_bytes": app_json_path.stat().st_size,
            "sha256": sha256_file(app_json_path),
        }
    )
    write_csv(
        manifests_dir / "metadata_manifest.csv",
        metadata_manifest,
        ["path", "source_name", "type", "size_bytes", "sha256"],
    )

    metadata_csv = metadata_dir / "image_metadata.csv"
    records = load_image_records(metadata_csv, limit=args.limit)
    print(f"Downloading {len(records)} images to {images_dir}")

    image_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    skip_existing = not args.no_skip_existing
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, record, images_dir, skip_existing, args.retries): record
            for record in records
        }
        for index, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            try:
                image_rows.append(future.result())
            except Exception as error:
                failures.append({"file_name": record.file_name, "url": record.url, "error": repr(error)})
            if index % 500 == 0 or index == len(records):
                print(f"Processed {index}/{len(records)} images; failures={len(failures)}")

    image_rows.sort(key=lambda row: str(row["file_name"]))
    write_csv(
        manifests_dir / "image_manifest.csv",
        image_rows,
        ["file_name", "url", "path", "size_bytes", "sha256", "status"],
    )
    if failures:
        write_csv(manifests_dir / "failed_downloads.csv", failures, ["file_name", "url", "error"])

    total_bytes = sum(int(row["size_bytes"]) for row in image_rows)
    summary = summarize_image_metadata(metadata_csv)
    summary.update(
        {
            "source_app_url": APP_URL,
            "image_base_url": IMAGE_BASE_URL,
            "downloaded_image_count": len(image_rows),
            "failed_image_count": len(failures),
            "image_bytes": total_bytes,
            "image_gb_decimal": total_bytes / 1_000_000_000,
            "image_gib_binary": total_bytes / (1024**3),
            "output_dir": str(output_dir),
            "generated_at_unix": int(time.time()),
        }
    )
    (manifests_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2)[:4000])
    if failures:
        print(f"Download finished with {len(failures)} failures. See {manifests_dir / 'failed_downloads.csv'}")
        return 1
    print("Download finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
