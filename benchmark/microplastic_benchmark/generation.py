from __future__ import annotations

import csv
import inspect
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageChops, ImageEnhance, ImageFilter

from .data import list_pngs, load_mask_l, load_rgb, paired_by_filename


def random_affine_mask(mask: Image.Image, size: int, rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
    mask = mask.resize((size, size), Image.Resampling.NEAREST)
    angle = rng.uniform(-20, 20)
    translate = (rng.uniform(-0.12, 0.12) * size, rng.uniform(-0.12, 0.12) * size)
    scale = rng.uniform(0.75, 1.25)
    transformed = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    canvas = Image.new("L", (size, size), 0)
    scaled = transformed.resize((max(1, int(size * scale)), max(1, int(size * scale))), Image.Resampling.NEAREST)
    x = int((size - scaled.width) / 2 + translate[0])
    y = int((size - scaled.height) / 2 + translate[1])
    canvas.paste(scaled, (x, y))
    return canvas.point(lambda v: 255 if v > 0 else 0), {
        "angle_degrees": angle,
        "translation_pixels": [translate[0], translate[1]],
        "scale": scale,
    }


def _mask_bbox(mask: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(mask.convert("L")) > 0
    ys, xs = np.where(arr)
    if len(xs) == 0:
        raise ValueError("Source mask is empty.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width, height = image_size
    return max(0, x0 - padding), max(0, y0 - padding), min(width, x1 + padding), min(height, y1 + padding)


def _target_fraction(method_cfg: dict[str, Any], rng: random.Random) -> float:
    low, high = method_cfg.get("insertion_area_fraction", [0.003, 0.035])
    low = max(float(low), 1e-6)
    high = max(float(high), low)
    # A log-uniform draw gives many realistic thin/small particles and a few larger fibers.
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def random_affine_object(
    image: Image.Image,
    mask: Image.Image,
    size: int,
    rng: random.Random,
    method_cfg: dict[str, Any],
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Return full-canvas RGB object pixels and the corresponding binary label mask."""

    image = image.convert("RGB")
    mask = mask.convert("L").point(lambda v: 255 if v > 0 else 0)
    bbox = _expand_bbox(_mask_bbox(mask), image.size, int(method_cfg.get("source_crop_padding", 16)))
    obj = image.crop(bbox)
    obj_mask = mask.crop(bbox)

    mask_pixels = max(1, int((np.asarray(obj_mask) > 0).sum()))
    target_pixels = _target_fraction(method_cfg, rng) * size * size
    scale = math.sqrt(target_pixels / mask_pixels)
    max_patch_fraction = float(method_cfg.get("max_patch_fraction", 0.85))
    max_dim = max(1, int(size * max_patch_fraction))
    scaled_w = max(1, int(round(obj.width * scale)))
    scaled_h = max(1, int(round(obj.height * scale)))
    if max(scaled_w, scaled_h) > max_dim:
        shrink = max_dim / max(scaled_w, scaled_h)
        scaled_w = max(1, int(round(scaled_w * shrink)))
        scaled_h = max(1, int(round(scaled_h * shrink)))
        scale *= shrink

    obj = obj.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
    obj_mask = obj_mask.resize((scaled_w, scaled_h), Image.Resampling.NEAREST)

    angle = rng.uniform(
        float(method_cfg.get("rotation_degrees_min", -35)),
        float(method_cfg.get("rotation_degrees_max", 35)),
    )
    obj = obj.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0))
    obj_mask = obj_mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0)
    obj_mask = obj_mask.point(lambda v: 255 if v > 0 else 0)

    if obj.width > size or obj.height > size:
        shrink = min(size / obj.width, size / obj.height) * 0.95
        obj = obj.resize((max(1, int(obj.width * shrink)), max(1, int(obj.height * shrink))), Image.Resampling.BICUBIC)
        obj_mask = obj_mask.resize(obj.size, Image.Resampling.NEAREST).point(lambda v: 255 if v > 0 else 0)
        scale *= shrink

    color_jitter = method_cfg.get("color_jitter", {})
    brightness = rng.uniform(float(color_jitter.get("brightness_min", 0.85)), float(color_jitter.get("brightness_max", 1.15)))
    contrast = rng.uniform(float(color_jitter.get("contrast_min", 0.9)), float(color_jitter.get("contrast_max", 1.15)))
    obj = ImageEnhance.Brightness(obj).enhance(brightness)
    obj = ImageEnhance.Contrast(obj).enhance(contrast)

    x = rng.randint(0, max(0, size - obj.width))
    y = rng.randint(0, max(0, size - obj.height))
    object_canvas = Image.new("RGB", (size, size), (0, 0, 0))
    mask_canvas = Image.new("L", (size, size), 0)
    object_canvas.paste(obj, (x, y))
    mask_canvas.paste(obj_mask, (x, y))
    mask_canvas = mask_canvas.point(lambda v: 255 if v > 0 else 0)
    actual_fraction = float((np.asarray(mask_canvas) > 0).mean())
    return object_canvas, mask_canvas, {
        "angle_degrees": angle,
        "scale": scale,
        "paste_xy": [x, y],
        "paste_size": [obj.width, obj.height],
        "source_bbox": list(bbox),
        "target_foreground_fraction": actual_fraction,
        "brightness": brightness,
        "contrast": contrast,
    }


def composite_object(background: Image.Image, object_canvas: Image.Image, mask: Image.Image, method_cfg: dict[str, Any]) -> Image.Image:
    opacity = float(method_cfg.get("object_opacity", 0.95))
    feather_radius = float(method_cfg.get("alpha_feather_radius", 0.6))
    alpha = mask.convert("L")
    if feather_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    if opacity < 1.0:
        alpha = alpha.point(lambda v: int(v * opacity))
    out = background.convert("RGB").copy()
    out.paste(object_canvas.convert("RGB"), (0, 0), alpha)
    return out


def boundary_blend_mask(mask: Image.Image, width: int = 5) -> Image.Image:
    width = max(1, int(width))
    kernel = width * 2 + 1
    binary = mask.convert("L").point(lambda v: 255 if v > 0 else 0)
    dilated = binary.filter(ImageFilter.MaxFilter(kernel))
    eroded = binary.filter(ImageFilter.MinFilter(kernel))
    return ImageChops.subtract(dilated, eroded).point(lambda v: 255 if v > 0 else 0)


def dilate_mask(mask: Image.Image, pixels: int) -> Image.Image:
    pixels = max(0, int(pixels))
    binary = mask.convert("L").point(lambda v: 255 if v > 0 else 0)
    if pixels <= 0:
        return binary
    return binary.filter(ImageFilter.MaxFilter(pixels * 2 + 1)).point(lambda v: 255 if v > 0 else 0)


def masked_change_metrics(before: Image.Image, after: Image.Image, mask: Image.Image) -> dict[str, float]:
    before_arr = np.asarray(before.convert("RGB")).astype(np.float32)
    after_arr = np.asarray(after.convert("RGB").resize(before.size, Image.Resampling.BICUBIC)).astype(np.float32)
    mask_arr = np.asarray(mask.convert("L").resize(before.size, Image.Resampling.NEAREST)) > 0
    if mask_arr.sum() == 0:
        return {
            "masked_mad": 0.0,
            "outside_mad": 0.0,
            "changed_px_frac": 0.0,
            "mask_frac": 0.0,
        }
    diff = np.abs(after_arr - before_arr).mean(axis=2)
    outside = ~mask_arr
    return {
        "masked_mad": float(diff[mask_arr].mean()),
        "outside_mad": float(diff[outside].mean()) if outside.any() else 0.0,
        "changed_px_frac": float((diff[mask_arr] > 12.0).mean()),
        "mask_frac": float(mask_arr.mean()),
    }


def passes_generation_qc(metrics: dict[str, float], method_cfg: dict[str, Any]) -> bool:
    return (
        metrics["mask_frac"] >= float(method_cfg.get("min_mask_area_fraction", 0.002))
        and metrics["masked_mad"] >= float(method_cfg.get("min_masked_mad", 12.0))
        and metrics["changed_px_frac"] >= float(method_cfg.get("min_changed_px_fraction", 0.35))
    )


def passes_visible_object_qc(metrics: dict[str, float], method_cfg: dict[str, Any]) -> bool:
    return (
        metrics["mask_frac"] >= float(method_cfg.get("min_mask_area_fraction", 0.002))
        and metrics["masked_mad"] >= float(method_cfg.get("min_background_masked_mad", method_cfg.get("min_masked_mad", 12.0)))
        and metrics["changed_px_frac"]
        >= float(method_cfg.get("min_background_changed_px_fraction", method_cfg.get("min_changed_px_fraction", 0.35)))
    )


def load_inpaint_pipeline(method_cfg: dict[str, Any], device: str):
    model_id = method_cfg["model_id"]
    fallback = method_cfg.get("fallback_model_id")
    try:
        pipe = _load_inpaint_pipeline_by_model_id(model_id)
    except Exception:
        if not fallback:
            raise
        pipe = _load_inpaint_pipeline_by_model_id(fallback)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def _load_inpaint_pipeline_by_model_id(model_id: str):
    # Compatibility for diffusers releases that still import this transformers constant.
    import transformers.utils as transformers_utils

    if not hasattr(transformers_utils, "FLAX_WEIGHTS_NAME"):
        transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

    if "FLUX" in model_id or "flux" in model_id:
        from diffusers import FluxFillPipeline

        return FluxFillPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    if "xl" in model_id.lower() or "sdxl" in model_id.lower():
        from diffusers import StableDiffusionXLInpaintPipeline

        return StableDiffusionXLInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    from diffusers import StableDiffusionInpaintPipeline

    return StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16)


def load_text2image_pipeline(method_cfg: dict[str, Any], device: str):
    model_id = method_cfg["model_id"]
    fallback = method_cfg.get("fallback_model_id")
    try:
        pipe = _load_text2image_pipeline_by_model_id(model_id)
    except Exception:
        if not fallback:
            raise
        pipe = _load_text2image_pipeline_by_model_id(fallback)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def _load_text2image_pipeline_by_model_id(model_id: str):
    # Compatibility for diffusers releases that still import this transformers constant.
    import transformers.utils as transformers_utils

    if not hasattr(transformers_utils, "FLAX_WEIGHTS_NAME"):
        transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

    if "xl" in model_id.lower() or "sdxl" in model_id.lower():
        from diffusers import StableDiffusionXLPipeline

        return StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    from diffusers import StableDiffusionPipeline

    return StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)


def generate_text2image_set(
    *,
    method_cfg: dict[str, Any],
    output_imgs: str | Path,
    count: int,
    image_size: int,
    seed: int,
    device: str,
    overwrite: bool = False,
) -> None:
    """Generate fully novel, unlabeled microplastic microscopy images from text.

    These images are not paired with segmentation masks. They should be used for
    qualitative inspection, pretraining, or later manual/SAM-assisted annotation.
    """

    output_imgs = Path(output_imgs)
    output_imgs.mkdir(parents=True, exist_ok=True)
    pipe = load_text2image_pipeline(method_cfg, device)
    prompt = method_cfg["prompt"]
    negative_prompt = method_cfg.get("negative_prompt")
    log_path = output_imgs / "generation_log.csv"
    log_exists = log_path.exists() and not overwrite
    with log_path.open("a" if log_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "generated_name",
                "seed",
                "model_id",
                "prompt",
                "negative_prompt",
                "generation_mode",
                "image_size",
                "guidance_scale",
                "steps",
            ],
        )
        if not log_exists:
            writer.writeheader()
        for i in range(count):
            generated_name = f"text2image_{i:05d}.png"
            out_img_path = output_imgs / generated_name
            if out_img_path.exists() and not overwrite:
                continue
            generator = torch.Generator(device=device).manual_seed(seed + i)
            call_args = {
                "prompt": prompt,
                "height": image_size,
                "width": image_size,
                "guidance_scale": float(method_cfg.get("guidance_scale", 7.0)),
                "num_inference_steps": int(method_cfg.get("steps", 30)),
                "generator": generator,
            }
            signature = inspect.signature(pipe.__call__)
            if "negative_prompt" in signature.parameters:
                call_args["negative_prompt"] = negative_prompt
            result = pipe(**call_args)
            result.images[0].convert("RGB").save(out_img_path)
            writer.writerow(
                {
                    "generated_name": generated_name,
                    "seed": seed + i,
                    "model_id": method_cfg["model_id"],
                    "prompt": prompt,
                    "negative_prompt": negative_prompt or "",
                    "generation_mode": "stable_diffusion_text2image_unlabeled",
                    "image_size": image_size,
                    "guidance_scale": float(method_cfg.get("guidance_scale", 7.0)),
                    "steps": int(method_cfg.get("steps", 30)),
                }
            )
            f.flush()


def generate_inpaint_set(
    *,
    method_cfg: dict[str, Any],
    backgrounds_dir: str | Path,
    source_images_dir: str | Path | None = None,
    source_masks_dir: str | Path,
    output_imgs: str | Path,
    output_masks: str | Path,
    count: int,
    image_size: int,
    seed: int,
    device: str,
    exclude_backgrounds: set[str],
    overwrite: bool = False,
) -> None:
    output_imgs = Path(output_imgs)
    output_masks = Path(output_masks)
    output_imgs.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    backgrounds = [p for p in list_pngs(backgrounds_dir) if p.name not in exclude_backgrounds]
    masks = list_pngs(source_masks_dir)
    if not backgrounds or not masks:
        raise ValueError("Need at least one C2 background and one source mask for generation.")
    source_pairs = None
    seed_object = bool(method_cfg.get("seed_object", source_images_dir is not None))
    if seed_object:
        if source_images_dir is None:
            raise ValueError("source_images_dir is required when seed_object is enabled.")
        source_pairs, errors = paired_by_filename(source_images_dir, source_masks_dir)
        if errors:
            raise ValueError(f"Source image/mask pairing errors: {errors[:5]}")
        if not source_pairs:
            raise ValueError("Need at least one paired source image/mask for object-seeded generation.")

    diffusion_blend = bool(method_cfg.get("diffusion_blend", False))
    pipe = load_inpaint_pipeline(method_cfg, device) if diffusion_blend else None
    log_path = output_imgs / "generation_log.csv"
    log_exists = log_path.exists() and not overwrite
    with log_path.open("a" if log_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "generated_name",
                "background_path",
                "source_image_path",
                "source_mask_path",
                "output_mask_path",
                "seed",
                "attempt",
                "model_id",
                "prompt",
                "generation_mode",
                "qc_masked_mad_vs_background",
                "qc_changed_px_frac_vs_background",
                "qc_masked_mad_vs_seeded",
                "qc_changed_px_frac_vs_seeded",
                "qc_mask_frac",
                "transform_json",
            ],
        )
        if not log_exists:
            writer.writeheader()
        i = 0
        attempt = 0
        max_attempts = int(math.ceil(count * float(method_cfg.get("max_generation_attempt_multiplier", 3.0))))
        while i < count and attempt < max_attempts:
            attempt += 1
            bg_path = rng.choice(backgrounds)
            generated_name = f"generated_{i:05d}.png"
            bg = load_rgb(bg_path).resize((image_size, image_size), Image.Resampling.LANCZOS)
            out_img_path = output_imgs / generated_name
            out_mask_path = output_masks / generated_name
            if out_img_path.exists() and out_mask_path.exists() and not overwrite:
                continue

            source_image_path = ""
            if seed_object and source_pairs is not None:
                pair = rng.choice(source_pairs)
                source_image_path = str(pair.image_path)
                mask_path = pair.mask_path
                object_canvas, gen_mask, transform = random_affine_object(
                    load_rgb(pair.image_path),
                    load_mask_l(pair.mask_path),
                    image_size,
                    rng,
                    method_cfg,
                )
                seeded_image = composite_object(bg, object_canvas, gen_mask, method_cfg)
                generation_mode = "object_seeded_composite"
            else:
                mask_path = rng.choice(masks)
                src_mask = load_mask_l(mask_path)
                gen_mask, transform = random_affine_mask(src_mask, image_size, rng)
                seeded_image = bg
                generation_mode = "diffusion_full_mask"

            if diffusion_blend:
                blend_mode = method_cfg.get("diffusion_mask_mode", "full")
                blend_mask = (
                    boundary_blend_mask(gen_mask, int(method_cfg.get("blend_boundary_width", 5)))
                    if blend_mode == "boundary"
                    else gen_mask
                )
                blend_mask = dilate_mask(blend_mask, int(method_cfg.get("inpaint_mask_dilation_px", 0)))
                generator = torch.Generator(device=device).manual_seed(seed + attempt)
                call_args = {
                    "prompt": method_cfg["prompt"],
                    "image": seeded_image,
                    "mask_image": blend_mask,
                    "guidance_scale": float(method_cfg.get("guidance_scale", 7.0)),
                    "num_inference_steps": int(method_cfg.get("steps", 30)),
                    "generator": generator,
                }
                if pipe is None:
                    raise RuntimeError("diffusion_blend enabled but pipeline was not loaded.")
                signature = inspect.signature(pipe.__call__)
                if "negative_prompt" in signature.parameters:
                    call_args["negative_prompt"] = method_cfg.get("negative_prompt")
                if "strength" in signature.parameters:
                    call_args["strength"] = float(method_cfg.get("strength", 0.98))
                result = pipe(**call_args)
                final_image = result.images[0].convert("RGB")
                if final_image.size != seeded_image.size:
                    final_image = final_image.resize(seeded_image.size, Image.Resampling.BICUBIC)
                seeded_object_mix = float(method_cfg.get("seeded_object_mix", 0.0))
                if seeded_object_mix > 0:
                    mix_mask = np.asarray(gen_mask.convert("L"), dtype=np.float32) / 255.0
                    mix_mask = np.clip(mix_mask[..., None] * min(seeded_object_mix, 1.0), 0.0, 1.0)
                    final_arr = np.asarray(final_image.convert("RGB"), dtype=np.float32)
                    seeded_arr = np.asarray(seeded_image.convert("RGB"), dtype=np.float32)
                    final_image = Image.fromarray(np.clip(final_arr * (1.0 - mix_mask) + seeded_arr * mix_mask, 0, 255).astype(np.uint8))
                generation_mode += f"+diffusion_{blend_mode}_blend"
            else:
                final_image = seeded_image

            bg_metrics = masked_change_metrics(bg, final_image, gen_mask)
            seeded_metrics = masked_change_metrics(seeded_image, final_image, gen_mask)
            if bool(method_cfg.get("reject_low_change", True)):
                if diffusion_blend:
                    if not (passes_visible_object_qc(bg_metrics, method_cfg) and passes_generation_qc(seeded_metrics, method_cfg)):
                        continue
                elif not passes_generation_qc(bg_metrics, method_cfg):
                    continue

            final_image.save(out_img_path)
            gen_mask.save(out_mask_path)
            writer.writerow(
                {
                    "generated_name": generated_name,
                    "background_path": str(bg_path),
                    "source_image_path": source_image_path,
                    "source_mask_path": str(mask_path),
                    "output_mask_path": str(out_mask_path),
                    "seed": seed + attempt,
                    "attempt": attempt,
                    "model_id": method_cfg["model_id"],
                    "prompt": method_cfg["prompt"],
                    "generation_mode": generation_mode,
                    "qc_masked_mad_vs_background": bg_metrics["masked_mad"],
                    "qc_changed_px_frac_vs_background": bg_metrics["changed_px_frac"],
                    "qc_masked_mad_vs_seeded": seeded_metrics["masked_mad"],
                    "qc_changed_px_frac_vs_seeded": seeded_metrics["changed_px_frac"],
                    "qc_mask_frac": bg_metrics["mask_frac"],
                    "transform_json": json.dumps(transform, sort_keys=True),
                }
            )
            f.flush()
            i += 1
        if i < count:
            raise RuntimeError(f"Generated only {i}/{count} images after {attempt} attempts; QC may be too strict.")
