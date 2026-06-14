#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    "c2_sdxl_inpaint",
    "c2_flux_inpaint",
    "c2_sd2_inpaint",
    "c2_sd2_fiber_inpaint",
    "c2_sdxl_texture_inpaint",
]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def mask_panel(mask: Image.Image) -> Image.Image:
    return mask.convert("L").point(lambda x: 255 if x > 0 else 0).convert("RGB")


def overlay_panel(image: Image.Image, mask: Image.Image, alpha: int) -> Image.Image:
    base = image.convert("RGBA")
    binary = mask.convert("L").point(lambda x: alpha if x > 0 else 0)
    red = Image.new("RGBA", base.size, (255, 0, 0, 0))
    red.putalpha(binary)
    return Image.alpha_composite(base, red).convert("RGB")


def draw_label(draw: ImageDraw.ImageDraw, x: int, text: str, panel_width: int, label_height: int) -> None:
    bbox = draw.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        (x + (panel_width - text_width) // 2, (label_height - text_height) // 2 - 1),
        text,
        fill=(20, 20, 20),
    )


def make_plot(task: tuple[str, str, str, str, str, str, bool, int]) -> tuple[str, bool, str]:
    method, generated_name, original_path_text, inpainted_path_text, mask_path_text, out_path_text, overwrite, alpha = task
    out_path = Path(out_path_text)
    if out_path.exists() and not overwrite:
        return method, False, generated_name

    original_path = resolve_path(original_path_text)
    inpainted_path = resolve_path(inpainted_path_text)
    mask_path = resolve_path(mask_path_text)
    with Image.open(original_path) as original_raw, Image.open(inpainted_path) as inpainted_raw, Image.open(mask_path) as mask_raw:
        original = original_raw.convert("RGB")
        inpainted = inpainted_raw.convert("RGB")
        mask = mask_raw.convert("L")
        if inpainted.size != original.size:
            inpainted = inpainted.resize(original.size, Image.Resampling.BICUBIC)
        if mask.size != original.size:
            mask = mask.resize(original.size, Image.Resampling.NEAREST)

        panels = [original, mask_panel(mask), inpainted, overlay_panel(inpainted, mask, alpha)]
        label_height = 34
        width, height = original.size
        canvas = Image.new("RGB", (width * 4, height + label_height), "white")
        draw = ImageDraw.Draw(canvas)
        labels = ["Original", "Mask", "Inpainted", "Overlay"]
        for idx, panel in enumerate(panels):
            x = idx * width
            canvas.paste(panel, (x, label_height))
            draw_label(draw, x, labels[idx], width, label_height)
            if idx:
                draw.line((x, 0, x, height + label_height), fill=(220, 220, 220), width=2)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, compress_level=3)
    return method, True, generated_name


def tasks_for_method(method: str, output_root: Path, overwrite: bool, alpha: int) -> list[tuple[str, str, str, str, str, str, bool, int]]:
    log_path = ROOT / "data" / "c2" / method / "generation_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    tasks: list[tuple[str, str, str, str, str, str, bool, int]] = []
    with log_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            generated_name = row["generated_name"]
            original_path = row["background_path"]
            inpainted_path = str(Path("data") / "c2" / method / generated_name)
            mask_path = row["output_mask_path"]
            out_name = f"{Path(generated_name).stem}_qc.png"
            out_path = output_root / method / out_name
            tasks.append((method, generated_name, original_path, inpainted_path, mask_path, str(out_path), overwrite, alpha))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--output-root", default="results/plots/generation_qc")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overlay-alpha", type=int, default=110)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    output_root = resolve_path(args.output_root)
    all_tasks = []
    for method in args.methods:
        method_tasks = tasks_for_method(method, output_root, args.overwrite, args.overlay_alpha)
        if args.limit is not None:
            method_tasks = method_tasks[: args.limit]
        print(f"{method}: queued {len(method_tasks)} plots -> {output_root / method}", flush=True)
        all_tasks.extend(method_tasks)

    completed = 0
    written = 0
    total = len(all_tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(make_plot, task) for task in all_tasks]
        for future in as_completed(futures):
            _method, did_write, _generated_name = future.result()
            completed += 1
            if did_write:
                written += 1
            if completed % args.progress_every == 0 or completed == total:
                print(f"completed={completed}/{total} written={written}", flush=True)

    print(f"done total={total} written={written} output_root={output_root}", flush=True)


if __name__ == "__main__":
    main()
