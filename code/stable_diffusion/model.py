# %% --- this one is for the inpainting model with text conditioning and dilation


import os
import math
import shutil
import warnings
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionInpaintPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm.auto import tqdm

check_min_version("0.28.0")
logger = get_logger(__name__, log_level="INFO")

# -----------------------------------------------------------------------------#
#                               DATASET                                        #
# -----------------------------------------------------------------------------#
class MicroplasticInpaintingDataset(Dataset):
    """
    Returns a dict with keys:
      • "pixel_values":   3×H×W tensor in [-1, 1]
      • "mask":           1×H×W float32 tensor  (0 = background, 1 = foreground)
    Image and mask always undergo *identical* random transforms.
    """
    def __init__(self, image_folder, mask_folder, image_size=(512, 512)):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_filenames = sorted(
            [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        self.image_size = image_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # ---------- load ------------------------------------------------------ #
        img_name   = self.image_filenames[idx]
        image_path = os.path.join(self.image_folder, img_name)
        mask_path  = os.path.join(self.mask_folder,  img_name)

        try:
            image = Image.open(image_path).convert("RGB")
            mask  = Image.open(mask_path).convert("L")
        except FileNotFoundError:
            logger.error(f"Missing file: {image_path} or {mask_path}")
            return self.__getitem__((idx + 1) % len(self))

        # ---------- resize (common, deterministic) --------------------------- #
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        # ---------- stochastic geometry (sample *once*) ---------------------- #
        # 1) flips
        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        if torch.rand(1).item() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # 2) affine (same params for both)
        angle  = torch.empty(1).uniform_(-15, 15).item()        # degrees
        scale  = torch.empty(1).uniform_(0.9, 1.1).item()
        shear  = 0.0
        translate = [0, 0]   # could randomise if wanted

        image = TF.affine(image, angle, translate, scale, shear, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.affine(mask,  angle, translate, scale, shear, interpolation=TF.InterpolationMode.NEAREST)

        # ---------- to tensor / normalise ------------------------------------ #
        image_tensor = TF.to_tensor(image)                    # 0‑1
        image_tensor = TF.normalize(image_tensor, [0.5], [0.5])  # → [-1, 1]
        mask_tensor  = torch.as_tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)

        return {"pixel_values": image_tensor, "mask": mask_tensor}


# -----------------------------------------------------------------------------#
#                              CONFIG                                          #
# -----------------------------------------------------------------------------#
class TrainingConfig:
    def __init__(self):
        self.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-inpainting"
        self.revision = None

        self.train_image_data_dir = "/mnt/shared/dils/projects/microplastic/data/c1/imgs"
        self.train_mask_data_dir = "/mnt/shared/dils/projects/microplastic/data/c1/masks_dilated"

        self.output_dir = "model_text_dilated"
        self.logging_dir = "logs"
        self.report_to = "tensorboard"

        self.resolution = 512
        self.train_batch_size = 1
        self.dataloader_num_workers = 2

        self.num_train_epochs = 100
        self.max_train_steps = None
        self.gradient_accumulation_steps = 4
        self.gradient_checkpointing = False

        self.learning_rate = 1e-5
        self.lr_scheduler = "constant"
        self.lr_warmup_steps = 0
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_weight_decay = 1e-2
        self.adam_epsilon = 1e-08
        self.max_grad_norm = 1.0

        self.mixed_precision = "fp16"  # "fp16", "bf16" or "no"
        self.enable_xformers_memory_efficient_attention = True
        self.seed = 42

        self.checkpointing_steps = 500
        self.save_on_epoch_end = True


# -----------------------------------------------------------------------------#
#                              HELPERS                                         #
# -----------------------------------------------------------------------------#
def save_augmented_preview(img_t, mask_t, save_dir, idx, show: bool = False):
    """Save side‑by‑side RGB and mask (img_t is in [-1,1])."""
    img_np = ((img_t.detach().cpu() + 1) * 0.5).permute(1, 2, 0).numpy()
    mask_np = mask_t.detach().cpu().squeeze().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(img_np)
    axs[0].set_title("Augmented Image")
    axs[0].axis("off")
    axs[1].imshow(mask_np, cmap="gray")
    axs[1].set_title("Augmented Mask")
    axs[1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"augmented_{idx}.png"))
    if show:              # Only show for the first two
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(fig)


def run_single_inference(
    pipe,
    init_image: Image.Image,
    init_mask: Image.Image,
    prompt: str,
    num_steps: int,
    guidance: float,
    dtype,
):
    device_type = "cuda" if pipe.device.type == "cuda" else "cpu"
    autocast_ctx = torch.autocast(device_type, dtype=dtype) if device_type == "cuda" else nullcontext()
    with autocast_ctx:
        out = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=init_mask,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
        )
    return out.images[0]


def print_settings_summary(cfg: TrainingConfig):
    """Log a neat settings table once before training."""
    dash = "-" * 60
    lines = [
        dash,
        "TRAINING CONFIGURATION SUMMARY",
        dash,
        f"Output directory           : {cfg.output_dir}",
        f"Figures directory          : {os.path.join(cfg.output_dir, 'figures')}",
        f"Pre‑trained model          : {cfg.pretrained_model_name_or_path}",
        "",
        f"Image resolution           : {cfg.resolution}×{cfg.resolution}",
        f"Batch size / GPU           : {cfg.train_batch_size}",
        f"Grad accumulation steps    : {cfg.gradient_accumulation_steps}",
        "",
        f"Epochs                     : {cfg.num_train_epochs}",
        f"Mixed precision            : {cfg.mixed_precision}",
        f"Learning rate              : {cfg.learning_rate}",
        f"Scheduler                  : {cfg.lr_scheduler}",
        "",
        f"Checkpoint every N steps   : {cfg.checkpointing_steps}",
        f"Save model each epoch      : {cfg.save_on_epoch_end}",
        dash,
    ]
    for l in lines:
        logger.info(l)


# -----------------------------------------------------------------------------#
#                              MAIN                                            #
# -----------------------------------------------------------------------------#
def main(args: TrainingConfig):
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ---------- accelerator & logging -------------
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    project_conf = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_conf,
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    # Print once – clean, easy‑to‑spot config dump
    if accelerator.is_main_process:
        print_settings_summary(args)

    # --------------------  LOAD SD COMPONENTS  -------------------- #
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(f"xFormers not enabled: {e}")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --------------------------- DATA ----------------------------- #
    train_dataset = MicroplasticInpaintingDataset(
        image_folder=args.train_image_data_dir,
        mask_folder=args.train_mask_data_dir,
        image_size=(args.resolution, args.resolution),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # -------- training length ---------- #
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # -------- accelerate prep ---------- #
    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )

    weight_dtype = (
        torch.float16
        if accelerator.mixed_precision == "fp16"
        else torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Prompt (fixed)
    prompt = (
        "microscopic plastic filaments, synthetic fibers, transparent polymer threads, "
        "small colorful microplastic fragments embedded in organic matter, detailed, close‑up"
    )
    text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        enc_hidden = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

    # --------------------- TRAIN LOOP ----------------------------- #
    global_step = 0
    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    augmented_saved = 0  # counter for preview images

    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(unet):
                pix = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                msk = batch["mask"].to(accelerator.device, dtype=weight_dtype)

                # ---- Save & show first two augmented samples ---- #
                if augmented_saved < 2 and accelerator.is_main_process:
                    for b in range(pix.size(0)):
                        if augmented_saved >= 2:
                            break
                        save_augmented_preview(
                            pix[b].float(), msk[b].float(), figures_dir, augmented_saved, show=True
                        )
                        augmented_saved += 1

                # --------------- Latent prep ------------------- #
                latents = vae.encode(pix).latent_dist.sample() * vae.config.scaling_factor
                masked_latents = vae.encode(pix * (1 - msk)).latent_dist.sample() * vae.config.scaling_factor
                latent_mask = F.interpolate(msk, scale_factor=1 / 8, mode="nearest")

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                unet_in = torch.cat([noisy_latents, masked_latents, latent_mask], dim=1)

                model_pred = unet(unet_in, timesteps, encoder_hidden_states=enc_hidden).sample
                target = (
                    noise if noise_scheduler.config.prediction_type == "epsilon"
                    else noise_scheduler.get_velocity(latents, noise, timesteps)
                )
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    accelerator.log({"train_loss": loss.detach().item()}, step=global_step)
                    progress.update(1); progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

                    # periodic checkpoint
                    if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                        ckpt = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(ckpt)
                        accelerator.unwrap_model(unet).save_pretrained(os.path.join(ckpt, "unet"))
                        logger.info(f"Saved checkpoint to {ckpt}")

                if global_step >= args.max_train_steps:
                    break

        # ---------------- END EPOCH ------------------ #
        if accelerator.is_main_process:
            if args.save_on_epoch_end:
                ep_dir = os.path.join(args.output_dir, f"epoch-{epoch}")
                accelerator.save_state(ep_dir)
                accelerator.unwrap_model(unet).save_pretrained(os.path.join(ep_dir, "unet"))
                logger.info(f"Saved epoch model to {ep_dir}")

            # quick inference snapshot
            inf_dir = os.path.join(args.output_dir, "inference_outputs")
            os.makedirs(inf_dir, exist_ok=True)

            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                scheduler=noise_scheduler,
                safety_checker=None,
                torch_dtype=weight_dtype,
            )
            pipe.unet = accelerator.unwrap_model(unet)
            pipe = pipe.to(accelerator.device)
            if args.enable_xformers_memory_efficient_attention:
                try: pipe.enable_xformers_memory_efficient_attention()
                except Exception: pass

            sample_name = train_dataset.image_filenames[0]
            base_img = Image.open(os.path.join(args.train_image_data_dir, sample_name)).convert("RGB").resize(
                (args.resolution, args.resolution)
            )
            base_msk = Image.open(os.path.join(args.train_mask_data_dir, sample_name)).convert("L").resize(
                (args.resolution, args.resolution)
            )

            result = run_single_inference(
                pipe, base_img, base_msk, prompt, num_steps=50, guidance=7.5, dtype=weight_dtype
            )
            result.save(os.path.join(inf_dir, f"epoch_{epoch}.png"))
            del pipe; torch.cuda.empty_cache()

    # -------------------- FINAL SAVE --------------------------- #
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "unet_final")
        accelerator.unwrap_model(unet).save_pretrained(final_path)
        sched_cfg = os.path.join(args.pretrained_model_name_or_path, "scheduler", "scheduler_config.json")
        if os.path.exists(sched_cfg):
            shutil.copy(sched_cfg, os.path.join(final_path, "scheduler_config.json"))
        logger.info(f"Saved final model to {final_path}")

    accelerator.end_training()


# -----------------------------------------------------------------------------#
#                              ENTRY                                           #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    cfg = TrainingConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    main(cfg)

# %%
