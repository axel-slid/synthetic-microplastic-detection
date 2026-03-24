"""
Stable Diffusion inpainting fine-tuned on microplastic data.

Fine-tunes the UNet of stabilityai/stable-diffusion-2-inpainting on c1
images, then uses the trained UNet for batch synthetic generation on c2.

Usage:
    # Train
    python -m src.generators.stable_diffusion --mode train \
        --image_dir data/c1/imgs --mask_dir data/c1/masks_dilated \
        --output_dir checkpoints/sd --epochs 100

    # Generate
    python -m src.generators.stable_diffusion --mode generate \
        --checkpoint checkpoints/sd/unet_final \
        --image_dir data/c2/imgs --mask_dir data/c1/masks_dilated \
        --output_dir data/c2/gen_sd --num_images 10000
"""

from __future__ import annotations

import argparse
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

# Diffusers / HuggingFace
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

logger = get_logger(__name__, log_level="INFO")

BASE_MODEL = "stabilityai/stable-diffusion-2-inpainting"
PROMPT = (
    "microscopic plastic filaments, synthetic fibers, transparent polymer threads, "
    "small colorful microplastic fragments embedded in organic matter, detailed, close-up"
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MicroplasticInpaintingDataset(Dataset):
    def __init__(self, image_folder: str, mask_folder: str, image_size: int = 512):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.filenames = sorted(
            f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.image_size = image_size

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        try:
            image = Image.open(os.path.join(self.image_folder, name)).convert("RGB")
            mask = Image.open(os.path.join(self.mask_folder, name)).convert("L")
        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self))

        sz = self.image_size
        image = TF.resize(image, (sz, sz), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (sz, sz), interpolation=TF.InterpolationMode.NEAREST)

        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image); mask = TF.hflip(mask)
        if torch.rand(1).item() > 0.5:
            image = TF.vflip(image); mask = TF.vflip(mask)

        angle = torch.empty(1).uniform_(-15, 15).item()
        scale = torch.empty(1).uniform_(0.9, 1.1).item()
        image = TF.affine(image, angle, [0, 0], scale, 0, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.affine(mask, angle, [0, 0], scale, 0, interpolation=TF.InterpolationMode.NEAREST)

        img_t = TF.normalize(TF.to_tensor(image), [0.5], [0.5])
        mask_t = torch.as_tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)
        return {"pixel_values": img_t, "mask": mask_t}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    project_conf = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_conf,
    )
    set_seed(42)

    # Load SD components
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                   weight_decay=1e-2, eps=1e-8)

    dataset = MicroplasticInpaintingDataset(args.image_dir, args.mask_dir, args.resolution)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True)

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    max_steps = args.epochs * steps_per_epoch

    lr_scheduler = get_scheduler("constant", optimizer=optimizer,
                                  num_warmup_steps=0, num_training_steps=max_steps * args.grad_accum)

    unet, optimizer, loader, lr_scheduler = accelerator.prepare(unet, optimizer, loader, lr_scheduler)

    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    text_inputs = tokenizer(PROMPT, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        enc_hidden = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

    global_step = 0
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process, desc="SD Training")

    for epoch in range(args.epochs):
        unet.train()
        for batch in loader:
            with accelerator.accumulate(unet):
                pix = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                msk = batch["mask"].to(accelerator.device, dtype=weight_dtype)

                latents = vae.encode(pix).latent_dist.sample() * vae.config.scaling_factor
                masked_latents = vae.encode(pix * (1 - msk)).latent_dist.sample() * vae.config.scaling_factor
                latent_mask = F.interpolate(msk, scale_factor=1 / 8, mode="nearest")

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                          (latents.shape[0],), device=latents.device).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                unet_in = torch.cat([noisy, masked_latents, latent_mask], dim=1)

                pred = unet(unet_in, timesteps, encoder_hidden_states=enc_hidden).sample
                target = noise if noise_scheduler.config.prediction_type == "epsilon" \
                    else noise_scheduler.get_velocity(latents, noise, timesteps)
                loss = F.mse_loss(pred.float(), target.float())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    progress.update(1)
                    progress.set_postfix(loss=f"{loss.item():.4f}", epoch=epoch)

        if accelerator.is_main_process:
            ep_dir = os.path.join(args.output_dir, f"epoch-{epoch}")
            accelerator.save_state(ep_dir)
            accelerator.unwrap_model(unet).save_pretrained(os.path.join(ep_dir, "unet"))

    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "unet_final")
        accelerator.unwrap_model(unet).save_pretrained(final_path)
        print(f"[SD] Training complete. Final UNet saved to {final_path}")

    accelerator.end_training()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _postprocess(tensor):
    img = (tensor / 2 + 0.5).clamp(0, 1)
    img = img.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
    return Image.fromarray((img * 255).astype("uint8"))


@torch.no_grad()
def _inpaint_one(img_path, mask_path, unet, vae, scheduler, device, dtype, image_size, steps):
    pil_img = Image.open(img_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    pil_mask = Image.open(mask_path).convert("L")

    # random affine on mask to vary placement
    mask_t = TF.to_tensor(
        TF.affine(pil_mask.resize((image_size, image_size), Image.NEAREST),
                  angle=random.uniform(-20, 20),
                  translate=[random.uniform(-0.15, 0.15) * image_size,
                              random.uniform(-0.15, 0.15) * image_size],
                  scale=1.0, shear=0,
                  interpolation=TF.InterpolationMode.NEAREST, fill=0)
    )
    mask_t = (mask_t > 0.5).float().unsqueeze(0).to(device, dtype)

    img_t = TF.normalize(TF.to_tensor(pil_img), [0.5] * 3, [0.5] * 3).unsqueeze(0).to(device, dtype)
    masked_latent = vae.encode(img_t * (1 - mask_t)).latent_dist.sample() * vae.config.scaling_factor
    latent_mask = F.interpolate(mask_t, scale_factor=1 / 8, mode="nearest")

    scheduler.set_timesteps(steps, device=device)
    lat = torch.randn((1, vae.config.latent_channels, image_size // 8, image_size // 8), device=device, dtype=dtype)
    null_emb = torch.zeros(1, 77, unet.config.cross_attention_dim, device=device, dtype=dtype)

    for t in scheduler.timesteps:
        noise = unet(torch.cat([lat, masked_latent, latent_mask], dim=1), t,
                     encoder_hidden_states=null_emb).sample
        lat = scheduler.step(noise, t, lat).prev_sample

    pred = _postprocess(vae.decode(lat / vae.config.scaling_factor).sample)
    mask_bin = TF.to_pil_image(mask_t.squeeze().cpu()).point(lambda x: 255 if x > 128 else 0, "1")
    composite = Image.composite(pred, pil_img, mask_bin)

    return composite, TF.to_pil_image(mask_t.squeeze().cpu())


def generate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7 \
        else torch.float32
    print(f"[SD] Generating {args.num_images} images on {device} ({dtype})")

    unet = UNet2DConditionModel.from_pretrained(args.checkpoint, torch_dtype=dtype).to(device).eval()
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=dtype).to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    all_images = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    all_masks = [os.path.join(args.mask_dir, f) for f in os.listdir(args.mask_dir)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    os.makedirs(args.output_dir, exist_ok=True)
    mask_out_dir = args.output_dir.rstrip("/") + "_masks"
    os.makedirs(mask_out_dir, exist_ok=True)

    for i in tqdm(range(args.num_images), desc="SD generation"):
        img_path = random.choice(all_images)
        mask_path = random.choice(all_masks)
        try:
            composite, mask_pil = _inpaint_one(img_path, mask_path, unet, vae, scheduler,
                                                device, dtype, 512, args.inference_steps)
            fname = f"generated_{i:05d}.png"
            composite.save(os.path.join(args.output_dir, fname))
            mask_pil.save(os.path.join(mask_out_dir, fname))
        except Exception as e:
            print(f"[SD] Skipping sample {i}: {e}")

    print(f"[SD] Saved {args.num_images} images to {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Stable Diffusion inpainting for microplastic synthesis")
    p.add_argument("--mode", choices=["train", "generate"], required=True)

    p.add_argument("--image_dir", type=str, default="data/c1/imgs")
    p.add_argument("--mask_dir", type=str, default="data/c1/masks_dilated")
    p.add_argument("--device", type=str, default="cuda")

    # train
    p.add_argument("--output_dir", type=str, default="checkpoints/sd")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--workers", type=int, default=2)

    # generate
    p.add_argument("--checkpoint", type=str, default="checkpoints/sd/unet_final")
    p.add_argument("--num_images", type=int, default=10000)
    p.add_argument("--inference_steps", type=int, default=50)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        generate(args)
