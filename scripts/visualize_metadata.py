#!/usr/bin/env python3
"""Create plots and example-image sheets for Microplastic Image Explorer metadata."""

from __future__ import annotations

import argparse
import csv
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value_counts(rows: list[dict[str, str]], column: str) -> Counter[str]:
    return Counter((row.get(column) or "").strip() or "(blank)" for row in rows)


def wrapped_label(label: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(label, width=width)) if len(label) > width else label


def save_barh(counts: Counter[str], title: str, output: Path, top_n: int = 12) -> None:
    items = counts.most_common(top_n)
    labels = [wrapped_label(label) for label, _ in reversed(items)]
    values = [count for _, count in reversed(items)]

    height = max(4.8, 0.48 * len(items) + 1.2)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(labels, values, color="#2f6f73")
    ax.set_title(title, loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("Image records")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for index, value in enumerate(values):
        ax.text(value, index, f" {value:,}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_size_availability(rows: list[dict[str, str]], output: Path) -> None:
    nonblank = sum(bool((row.get("size") or "").strip()) for row in rows)
    blank = len(rows) - nonblank
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["Has size metadata", "Blank size"], [nonblank, blank], color=["#2f6f73", "#b7b7b7"])
    ax.set_title("Particle-size Metadata Availability", loc="left", fontsize=15, weight="bold")
    ax.set_ylabel("Image records")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate([nonblank, blank]):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def pick_examples(rows: list[dict[str, str]], images_dir: Path, max_examples: int, require_size: bool) -> list[dict[str, str]]:
    candidates = [row for row in rows if (row.get("size") or "").strip()] if require_size else rows
    preferred: list[dict[str, str]] = []
    seen_morphologies: set[str] = set()
    for row in candidates:
        file_name = row.get("file_names") or ""
        morphology = (row.get("morphology") or "").strip()
        if not file_name or not morphology or morphology in seen_morphologies:
            continue
        if (images_dir / file_name).exists():
            preferred.append(row)
            seen_morphologies.add(morphology)
        if len(preferred) >= max_examples:
            return preferred

    for row in candidates:
        file_name = row.get("file_names") or ""
        if file_name and (images_dir / file_name).exists() and row not in preferred:
            preferred.append(row)
        if len(preferred) >= max_examples:
            break
    return preferred


def make_contact_sheet(
    rows: list[dict[str, str]],
    images_dir: Path,
    output: Path,
    max_examples: int = 12,
    require_size: bool = False,
) -> None:
    selected = pick_examples(rows, images_dir, max_examples, require_size=require_size)
    if not selected:
        print(f"No example images found in {images_dir}; skipping contact sheet.")
        return

    thumb_w, thumb_h = 220, 160
    label_h = 64
    columns = 4
    rows_count = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows_count * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    for index, row in enumerate(selected):
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + label_h)
        image_path = images_dir / row["file_names"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            ox = x + (thumb_w - image.width) // 2
            oy = y + (thumb_h - image.height) // 2
            sheet.paste(image, (ox, oy))

        morphology = (row.get("morphology") or "unknown").strip()
        color = (row.get("color") or "unknown").strip()
        size = (row.get("size") or "size blank").strip()
        label = f"{morphology} | {color}\n{size}\n{row['file_names']}"
        draw.text((x + 8, y + thumb_h + 6), label, fill="#222222", font=font_bold if index == 0 else font)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def write_summary(rows: list[dict[str, str]], output: Path) -> None:
    lines = [
        "# Metadata Summary",
        "",
        f"- image records: {len(rows):,}",
        f"- unique file names: {len({row.get('file_names', '') for row in rows if row.get('file_names')}):,}",
        f"- records with nonblank `size`: {sum(bool((row.get('size') or '').strip()) for row in rows):,}",
        "",
        "## Top Citations",
        "",
    ]
    for label, count in value_counts(rows, "citation").most_common(8):
        lines.append(f"- {label}: {count:,}")
    lines.extend(["", "## Top Morphologies", ""])
    for label, count in value_counts(rows, "morphology").most_common(10):
        lines.append(f"- {label}: {count:,}")
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

    save_barh(value_counts(rows, "citation"), "Image Records by Source Citation", args.output_dir / "citation_counts.png", top_n=8)
    save_barh(value_counts(rows, "morphology"), "Image Records by Morphology", args.output_dir / "morphology_counts.png", top_n=10)
    save_barh(value_counts(rows, "polymer"), "Image Records by Polymer", args.output_dir / "polymer_counts.png", top_n=10)
    save_barh(value_counts(rows, "color"), "Image Records by Color", args.output_dir / "color_counts.png", top_n=12)
    save_barh(value_counts([row for row in rows if (row.get("size") or "").strip()], "size"), "Top Nonblank Particle-size Values", args.output_dir / "size_counts.png", top_n=15)
    save_size_availability(rows, args.output_dir / "size_availability.png")
    make_contact_sheet(rows, args.images_dir, args.output_dir / "example_images.jpg", max_examples=args.max_examples)
    make_contact_sheet(
        rows,
        args.images_dir,
        args.output_dir / "size_labeled_examples.jpg",
        max_examples=args.max_examples,
        require_size=True,
    )
    write_summary(rows, args.output_dir / "metadata_summary.md")
    print(f"Wrote metadata visualizations to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
