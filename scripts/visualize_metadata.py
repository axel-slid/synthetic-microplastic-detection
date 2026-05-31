#!/usr/bin/env python3
"""Visualize particle-size metadata and polymers for Microplastic Image Explorer."""

from __future__ import annotations

import argparse
import csv
import re
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def size_labeled(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if (row.get("size") or "").strip()]


def value_counts(rows: list[dict[str, str]], column: str) -> Counter[str]:
    return Counter((row.get(column) or "").strip() or "(blank)" for row in rows)


def wrapped(label: str, width: int = 30) -> str:
    return "\n".join(textwrap.wrap(label, width=width)) if len(label) > width else label


def safe_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "blank"


def save_size_availability(rows: list[dict[str, str]], output: Path) -> None:
    labeled = len(size_labeled(rows))
    blank = len(rows) - labeled
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Has particle size", "No particle size"], [labeled, blank], color=["#2f6f73", "#b7b7b7"])
    ax.set_title("Particle-size Metadata Availability", loc="left", fontsize=15, weight="bold")
    ax.set_ylabel("Image records")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for bar in bars:
        value = int(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_size_labeled_polymer_counts(rows: list[dict[str, str]], output: Path) -> None:
    counts = value_counts(size_labeled(rows), "polymer")
    items = counts.most_common()
    labels = [wrapped(label) for label, _ in reversed(items)]
    values = [count for _, count in reversed(items)]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.65 * len(items) + 1.4)))
    ax.barh(labels, values, color="#2f6f73")
    ax.set_title("Polymers Among Records With Particle-size Metadata", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Image records with nonblank size")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for index, value in enumerate(values):
        ax.text(value, index, f" {value:,}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def select_examples(rows: list[dict[str, str]], images_dir: Path, max_examples: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_sizes: set[str] = set()
    for row in rows:
        file_name = row.get("file_names") or ""
        size = (row.get("size") or "").strip()
        if not file_name or not size or not (images_dir / file_name).exists():
            continue
        if size in seen_sizes:
            continue
        selected.append(row)
        seen_sizes.add(size)
        if len(selected) >= max_examples:
            return selected

    for row in rows:
        file_name = row.get("file_names") or ""
        if file_name and (images_dir / file_name).exists() and row not in selected:
            selected.append(row)
        if len(selected) >= max_examples:
            break
    return selected


def save_montage(rows: list[dict[str, str]], images_dir: Path, output: Path, title: str, max_examples: int) -> bool:
    selected = select_examples(rows, images_dir, max_examples)
    if not selected:
        return False

    thumb_w, thumb_h = 220, 160
    label_h = 72
    title_h = 46
    columns = min(4, max(1, len(selected)))
    row_count = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, title_h + row_count * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(12)
    title_font = load_font(16, bold=True)
    draw.text((10, 12), title, fill="#1f1f1f", font=title_font)

    for index, row in enumerate(selected):
        x = (index % columns) * thumb_w
        y = title_h + (index // columns) * (thumb_h + label_h)
        image_path = images_dir / row["file_names"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            sheet.paste(image, (x + (thumb_w - image.width) // 2, y + (thumb_h - image.height) // 2))

        label_parts = [
            (row.get("morphology") or "morphology blank").strip() or "morphology blank",
            (row.get("color") or "color blank").strip() or "color blank",
            (row.get("size") or "size blank").strip() or "size blank",
            row["file_names"],
        ]
        label = f"{label_parts[0]} | {label_parts[1]}\n{label_parts[2]}\n{label_parts[3]}"
        draw.text((x + 8, y + thumb_h + 6), label, fill="#222222", font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)
    return True


def save_polymer_montages(rows: list[dict[str, str]], images_dir: Path, output_dir: Path, max_examples: int) -> list[tuple[str, Path, int]]:
    labeled = size_labeled(rows)
    polymers = sorted(
        {row.get("polymer", "").strip() for row in labeled if row.get("polymer", "").strip()},
        key=str.casefold,
    )
    outputs: list[tuple[str, Path, int]] = []
    for polymer in polymers:
        polymer_rows = [row for row in labeled if (row.get("polymer") or "").strip() == polymer]
        path = output_dir / f"polymer_montage_{safe_name(polymer)}.jpg"
        if save_montage(
            polymer_rows,
            images_dir,
            path,
            title=f"{polymer} ({len(polymer_rows):,} size-labeled records)",
            max_examples=max_examples,
        ):
            outputs.append((polymer, path, len(polymer_rows)))
    return outputs


def write_summary(rows: list[dict[str, str]], montages: list[tuple[str, Path, int]], output: Path) -> None:
    labeled = size_labeled(rows)
    lines = [
        "# Focused Metadata Summary",
        "",
        f"- total image records: {len(rows):,}",
        f"- records with particle-size metadata: {len(labeled):,}",
        f"- records without particle-size metadata: {len(rows) - len(labeled):,}",
        "",
        "## Polymers Among Size-labeled Records",
        "",
    ]
    for polymer, count in value_counts(labeled, "polymer").most_common():
        lines.append(f"- {polymer}: {count:,}")
    lines.extend(["", "## Polymer Montages", ""])
    for polymer, path, count in montages:
        lines.append(f"- {polymer}: {count:,} records, `{path.name}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/microplastic_image_explorer/metadata/image_metadata.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/microplastic_image_explorer/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--max-examples", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_size_availability(rows, args.output_dir / "size_availability.png")
    save_size_labeled_polymer_counts(rows, args.output_dir / "size_labeled_polymer_counts.png")
    montages = save_polymer_montages(rows, args.images_dir, args.output_dir, max_examples=args.max_examples)
    write_summary(rows, montages, args.output_dir / "metadata_summary.md")
    print(f"Wrote focused metadata visualizations to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
