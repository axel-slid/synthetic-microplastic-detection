#!/usr/bin/env python3
"""Filter Microplastic Image Explorer metadata and optionally download matches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


IMAGE_BASE_URL = "https://d2jrxerjcsjhs7.cloudfront.net/"


def image_url(file_name: str) -> str:
    return IMAGE_BASE_URL + urllib.parse.quote(file_name, safe="/")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def contains(value: str, query: str | None) -> bool:
    if query is None:
        return True
    return query.casefold() in (value or "").casefold()


def exact_or_contains(row: dict[str, str], column: str, query: str | None, exact: bool) -> bool:
    if query is None:
        return True
    value = (row.get(column) or "").strip()
    if exact:
        return value.casefold() == query.casefold()
    return contains(value, query)


def filter_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    file_regex = re.compile(args.file_regex) if args.file_regex else None
    size_regex = re.compile(args.size_regex, flags=re.IGNORECASE) if args.size_regex else None

    for row in rows:
        if not exact_or_contains(row, "citation", args.citation, args.exact):
            continue
        if not exact_or_contains(row, "researcher", args.researcher, args.exact):
            continue
        if not exact_or_contains(row, "morphology", args.morphology, args.exact):
            continue
        if not exact_or_contains(row, "polymer", args.polymer, args.exact):
            continue
        if not exact_or_contains(row, "color", args.color, args.exact):
            continue
        if args.has_size and not (row.get("size") or "").strip():
            continue
        if args.no_blank_morphology and not (row.get("morphology") or "").strip():
            continue
        if size_regex and not size_regex.search(row.get("size") or ""):
            continue
        if file_regex and not file_regex.search(row.get("file_names") or ""):
            continue
        output.append(row)

    if args.limit is not None:
        output = output[: args.limit]
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                    total += len(chunk)
                    digest.update(chunk)
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


def download_match(row: dict[str, str], images_dir: Path, retries: int, skip_existing: bool) -> dict[str, Any]:
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
    parser.add_argument("--metadata", type=Path, required=True, help="Path to image_metadata.csv")
    parser.add_argument("--output", type=Path, required=True, help="Filtered metadata CSV")
    parser.add_argument("--write-urls", type=Path, help="Optional text file with one image URL per match")
    parser.add_argument("--citation")
    parser.add_argument("--researcher")
    parser.add_argument("--morphology")
    parser.add_argument("--polymer")
    parser.add_argument("--color")
    parser.add_argument("--has-size", action="store_true")
    parser.add_argument("--no-blank-morphology", action="store_true")
    parser.add_argument("--size-regex", help="Regex applied to the raw size field, e.g. '>500|1\\s*MM'")
    parser.add_argument("--file-regex", help="Regex applied to file_names")
    parser.add_argument("--exact", action="store_true", help="Use exact case-insensitive string matching")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--images-dir", type=Path, default=Path("outputs/images"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/download_manifest.csv"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.metadata)
    matches = filter_rows(rows, args)

    if not rows:
        raise ValueError(f"{args.metadata} has no rows")
    write_rows(args.output, matches, list(rows[0].keys()))

    if args.write_urls:
        args.write_urls.parent.mkdir(parents=True, exist_ok=True)
        args.write_urls.write_text(
            "\n".join(image_url(row["file_names"]) for row in matches) + ("\n" if matches else ""),
            encoding="utf-8",
        )

    print(f"Matched {len(matches)} of {len(rows)} metadata rows")

    if args.download_images:
        image_manifest: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_match, row, args.images_dir, args.retries, not args.no_skip_existing): row
                for row in matches
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    image_manifest.append(future.result())
                except Exception as error:
                    failures.append({"file_name": row["file_names"], "url": image_url(row["file_names"]), "error": repr(error)})
                if index % 250 == 0 or index == len(matches):
                    print(f"Processed {index}/{len(matches)} downloads; failures={len(failures)}")

        image_manifest.sort(key=lambda item: str(item["file_name"]))
        write_rows(args.manifest, image_manifest, ["file_name", "url", "path", "size_bytes", "sha256", "status"])
        if failures:
            write_rows(args.manifest.with_name("failed_downloads.csv"), failures, ["file_name", "url", "error"])
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
