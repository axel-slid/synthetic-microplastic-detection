#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


DEFAULT_PROMPT = (
    "Create a realistic microscope image of an environmental water sample containing "
    "new microplastic particles. Include natural debris, fibers, sediment, bubbles, "
    "and varied particle morphologies. Use sharp optical microscopy style, not an illustration."
)

IMAGE_PRICING = {
    "standard": {
        "0.5k": {"tokens": 747, "usd_per_image": 0.045},
        "1k": {"tokens": 1120, "usd_per_image": 0.067},
        "2k": {"tokens": 1680, "usd_per_image": 0.101},
        "4k": {"tokens": 2520, "usd_per_image": 0.151},
    },
    "batch": {
        "0.5k": {"tokens": 747, "usd_per_image": 0.022},
        "1k": {"tokens": 1120, "usd_per_image": 0.034},
        "2k": {"tokens": 1680, "usd_per_image": 0.050},
        "4k": {"tokens": 2520, "usd_per_image": 0.076},
    },
}


def estimated_cost(count: int, image_size: str, tier: str) -> dict[str, float | int | str]:
    if tier not in IMAGE_PRICING:
        raise ValueError(f"Unknown pricing tier: {tier}")
    if image_size not in IMAGE_PRICING[tier]:
        raise ValueError(f"Unknown image size: {image_size}")
    row = IMAGE_PRICING[tier][image_size]
    return {
        "count": count,
        "image_size": image_size,
        "pricing_tier": tier,
        "output_tokens_per_image": row["tokens"],
        "usd_per_image": row["usd_per_image"],
        "estimated_usd": round(count * row["usd_per_image"], 4),
    }


def print_cost_estimate(cost: dict[str, float | int | str]) -> None:
    print(
        "estimated_cost "
        f"count={cost['count']} "
        f"image_size={cost['image_size']} "
        f"tier={cost['pricing_tier']} "
        f"output_tokens_per_image={cost['output_tokens_per_image']} "
        f"usd_per_image=${cost['usd_per_image']} "
        f"estimated_usd=${cost['estimated_usd']}"
    )


def load_api_key(path: str | None) -> str | None:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return None
    if path:
        key = Path(path).read_text(encoding="utf-8").strip()
        if key:
            os.environ["GEMINI_API_KEY"] = key
            return key
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate unlabeled microplastic microscopy images with Gemini image generation, also known as Nano Banana."
    )
    parser.add_argument("--out", default="data/c2/gemini_new_microplastic_text2image")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model", default="gemini-3.1-flash-image")
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-size", choices=sorted(IMAGE_PRICING["standard"]), default="0.5k")
    parser.add_argument("--pricing-tier", choices=sorted(IMAGE_PRICING), default="standard")
    parser.add_argument("--estimate-cost-only", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cost = estimated_cost(args.count, args.image_size, args.pricing_tier)
    print_cost_estimate(cost)
    if args.estimate_cost_only:
        return
    if args.max_cost_usd is not None and float(cost["estimated_usd"]) > args.max_cost_usd:
        raise SystemExit(
            f"Refusing to run: estimated ${cost['estimated_usd']} exceeds --max-cost-usd ${args.max_cost_usd}."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit("Install google-genai or run `pip install google-genai` before using this script.") from exc

    load_api_key(args.api_key_file)
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY, or pass --api-key-file, before using this script.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "generation_log.csv"
    log_exists = log_path.exists() and not args.overwrite
    client = genai.Client()

    with log_path.open("a" if log_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "generated_name",
                "seed",
                "model_id",
                "prompt",
                "generation_mode",
                "image_size",
                "pricing_tier",
                "estimated_usd_per_image",
            ],
        )
        if not log_exists:
            writer.writeheader()
        for i in range(args.count):
            generated_name = f"gemini_{i:05d}.png"
            out_path = out_dir / generated_name
            if out_path.exists() and not args.overwrite:
                continue
            prompt = f"{args.prompt}\nSeed hint: {args.seed + i}."
            response = client.models.generate_content(model=args.model, contents=[prompt])
            saved = False
            for part in response.parts:
                if getattr(part, "inline_data", None) is not None:
                    part.as_image().save(out_path)
                    saved = True
                    break
            if not saved:
                print(f"No image returned for {generated_name}", file=sys.stderr)
                continue
            writer.writerow(
                {
                    "generated_name": generated_name,
                    "seed": args.seed + i,
                    "model_id": args.model,
                    "prompt": args.prompt,
                    "generation_mode": "nano_banana_text2image_unlabeled",
                    "image_size": args.image_size,
                    "pricing_tier": args.pricing_tier,
                    "estimated_usd_per_image": cost["usd_per_image"],
                }
            )
            f.flush()
    print(out_dir)


if __name__ == "__main__":
    main()
