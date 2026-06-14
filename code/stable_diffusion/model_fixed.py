# %%

#!/usr/bin/env python
# coding: utf-8
"""
Micro‑plastic in‑painting with Stable‑Diffusion U‑Net fine‑tuning.
-----------------------------------------------------------------
• Random 80/20 train‑validation split (seeded).
• Per‑epoch validation loss logging + CSV dump.
"""
import os, math, shutil, warnings
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import functional as TF
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (AutoencoderKL, DDPMScheduler,
                       UNet2DConditionModel, StableDiffusionInpaintPipeline)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm.auto import tqdm

check_min_version("0.28.0")
logger = get_logger(__name__, log_level="INFO")

# --------------------------------------------------------------------------- #
#                              DATASET                                        #
# --------------------------------------------------------------------------- #
class MicroplasticInpaintingDataset(Dataset):
    """Returns: {"pixel_values": 3×H×W ∈[-1,1], "mask": 1×H×W float32}"""
    def __init__(self, image_folder, mask_folder, image_size=(512, 512)):
        self.image_folder, self.mask_folder = image_folder, mask_folder
        self.image_filenames = sorted(
            [f for f in os.listdir(image_folder)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        self.image_size = image_size

    def __len__(self): return len(self.image_filenames)

    def __getitem__(self, idx):
        name   = self.image_filenames[idx]
        img_p  = os.path.join(self.image_folder, name)
        msk_p  = os.path.join(self.mask_folder,  name)

        try:
            img = Image.open(img_p).convert("RGB")
            msk = Image.open(msk_p).convert("L")
        except FileNotFoundError:
            logger.error(f"Missing {img_p} or {msk_p}")
            return self.__getitem__((idx + 1) % len(self))

        # --- resize (deterministic) -----------------------------------------
        img = TF.resize(img, self.image_size,
                        interpolation=TF.InterpolationMode.BILINEAR)
        msk = TF.resize(msk, self.image_size,
                        interpolation=TF.InterpolationMode.NEAREST)

        # --- random flips ----------------------------------------------------
        if torch.rand(1).item() > .5:
            img, msk = TF.hflip(img), TF.hflip(msk)
        if torch.rand(1).item() > .5:
            img, msk = TF.vflip(img), TF.vflip(msk)

        # --- random affine (same params) ------------------------------------
        ang  = torch.empty(1).uniform_(-15, 15).item()
        scl  = torch.empty(1).uniform_(0.9, 1.1).item()
        img = TF.affine(img, ang, [0, 0], scl, 0.,
                        interpolation=TF.InterpolationMode.BILINEAR)
        msk = TF.affine(msk, ang, [0, 0], scl, 0.,
                        interpolation=TF.InterpolationMode.NEAREST)

        # --- tensors --------------------------------------------------------
        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, [0.5], [0.5])        # → [-1,1]
        msk_t = torch.as_tensor(np.array(msk),
                                dtype=torch.float32).unsqueeze(0)
        return {"pixel_values": img_t, "mask": msk_t}

# --------------------------------------------------------------------------- #
#                             CONFIG                                          #
# --------------------------------------------------------------------------- #
class TrainingConfig:
    def __init__(self):
        # data ----------------------------------------------------------------
        self.image_dir = "/mnt/shared/dils/projects/microplastic/data/c1/imgs"
        self.mask_dir  = "/mnt/shared/dils/projects/microplastic/data/c1/masks_dilated"
        self.train_split = 0.80          # 80 % training, 20 % validation

        # model / output ------------------------------------------------------
        self.pretrained_model = "stabilityai/stable-diffusion-2-inpainting"
        self.revision = None
        self.output_dir  = "model_text_dilated"
        self.figures_dir = "figures"
        self.logging_dir = "logs"
        self.report_to   = "tensorboard"

        # training hyper‑params ----------------------------------------------
        self.resolution  = 512
        self.batch_size  = 1
        self.workers     = 2
        self.num_epochs  = 100
        self.grad_accum  = 4
        self.lr          = 1e-5
        self.lr_scheduler = "constant"
        self.mixed_precision = "fp16"    # "fp16" | "bf16" | "no"

        # xFormers / misc -----------------------------------------------------
        self.enable_xformers = True
        self.seed  = 42
        self.ckpt_steps = 500
        self.save_on_epoch_end = True

# --------------------------------------------------------------------------- #
#                             HELPERS                                         #
# --------------------------------------------------------------------------- #
def save_aug_preview(img_t, msk_t, save_dir, idx, show=False):
    img_np = ((img_t.cpu() + 1) * 0.5).permute(1,2,0).numpy()
    msk_np = msk_t.cpu().squeeze().numpy()
    fig, ax = plt.subplots(1,2, figsize=(6,3))
    ax[0].imshow(img_np); ax[0].axis("off"); ax[0].set_title("Augmented img")
    ax[1].imshow(msk_np, cmap="gray"); ax[1].axis("off"); ax[1].set_title("Augmented mask")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"augmented_{idx}.png"))
    if show: plt.show(block=False); plt.pause(.001)
    plt.close(fig)

def run_inference(pipe, img, msk, prompt, steps, guidance, dtype):
    device = "cuda" if pipe.device.type == "cuda" else "cpu"
    ctx = torch.autocast(device, dtype=dtype) if device=="cuda" else nullcontext()
    with ctx:
        res = pipe(prompt=prompt, image=img, mask_image=msk,
                   num_inference_steps=steps, guidance_scale=guidance)
    return res.images[0]

# --------------------------------------------------------------------------- #
#                              MAIN                                           #
# --------------------------------------------------------------------------- #
def main(cfg: TrainingConfig):
    # directories ------------------------------------------------------------
    os.makedirs(cfg.output_dir,  exist_ok=True)
    fig_dir = os.path.join(cfg.output_dir, cfg.figures_dir)
    os.makedirs(fig_dir, exist_ok=True)

    # accelerator ------------------------------------------------------------
    acc = Accelerator(gradient_accumulation_steps=cfg.grad_accum,
                      mixed_precision=cfg.mixed_precision,
                      log_with=cfg.report_to,
                      project_config=ProjectConfiguration(
                          project_dir=cfg.output_dir,
                          logging_dir=os.path.join(cfg.output_dir,
                                                   cfg.logging_dir)))
    set_seed(cfg.seed)

    # data -------------------------------------------------------------------
    full_ds = MicroplasticInpaintingDataset(cfg.image_dir, cfg.mask_dir,
                                            image_size=(cfg.resolution,cfg.resolution))
    train_len = int(cfg.train_split * len(full_ds))
    val_len   = len(full_ds) - train_len
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(full_ds, [train_len,val_len], generator=gen)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.workers, pin_memory=True)

    # models -----------------------------------------------------------------
    sched = DDPMScheduler.from_pretrained(cfg.pretrained_model, subfolder="scheduler")
    vae   = AutoencoderKL.from_pretrained(cfg.pretrained_model, subfolder="vae",
                                          revision=cfg.revision)
    unet  = UNet2DConditionModel.from_pretrained(cfg.pretrained_model, subfolder="unet",
                                                 revision=cfg.revision)
    tok   = CLIPTokenizer.from_pretrained(cfg.pretrained_model, subfolder="tokenizer")
    txtenc= CLIPTextModel.from_pretrained(cfg.pretrained_model, subfolder="text_encoder")

    vae.requires_grad_(False); txtenc.requires_grad_(False); unet.train()
    if cfg.enable_xformers:
        try: unet.enable_xformers_memory_efficient_attention()
        except Exception as e: logger.warning(f"xFormers not enabled: {e}")

    opt = torch.optim.AdamW(unet.parameters(), lr=cfg.lr,
                            betas=(0.9,0.999), weight_decay=1e-2, eps=1e-8)

    # training length / scheduler -------------------------------------------
    max_steps = cfg.num_epochs * math.ceil(len(train_dl)/cfg.grad_accum)
    lr_sched  = get_scheduler(cfg.lr_scheduler, optimizer=opt,
                              num_warmup_steps=0,
                              num_training_steps=max_steps*cfg.grad_accum)

    # accelerator prepare ----------------------------------------------------
    unet, opt, train_dl, val_dl, lr_sched = acc.prepare(
        unet, opt, train_dl, val_dl, lr_sched)

    # dtype / devices --------------------------------------------------------
    w_dtype = (torch.float16 if acc.mixed_precision=="fp16"
               else torch.bfloat16 if acc.mixed_precision=="bf16"
               else torch.float32)
    vae.to(acc.device, dtype=w_dtype)
    txtenc.to(acc.device, dtype=w_dtype)

    # prompt embedding -------------------------------------------------------
    prompt = ("microscopic plastic filaments, synthetic fibers, transparent polymer threads, "
              "small colorful microplastic fragments embedded in organic matter, detailed, close‑up")
    with torch.no_grad():
        enc_hidden = txtenc(
            tok(prompt, padding="max_length",
                max_length=tok.model_max_length, return_tensors="pt"
               ).input_ids.to(acc.device))[0]

    # -------------------- TRAIN -------------------------------------------
    global_step = 0
    val_history = []
    aug_saved   = 0
    pbar = tqdm(range(max_steps), disable=not acc.is_local_main_process)

    for epoch in range(cfg.num_epochs):
        unet.train()
        for batch in train_dl:
            with acc.accumulate(unet):
                pix = batch["pixel_values"].to(acc.device, dtype=w_dtype)
                msk = batch["mask"].to(acc.device, dtype=w_dtype)

                if aug_saved < 2 and acc.is_main_process:
                    for b in range(pix.size(0)):
                        if aug_saved >= 2: break
                        save_aug_preview(pix[b].float(), msk[b].float(),
                                         fig_dir, aug_saved, show=True)
                        aug_saved += 1

                # ----- latent prep ------------------------------------------
                lat  = vae.encode(pix).latent_dist.sample() * vae.config.scaling_factor
                lat_m= vae.encode(pix*(1-msk)).latent_dist.sample() * vae.config.scaling_factor
                msk_lat = F.interpolate(msk, scale_factor=1/8, mode="nearest")

                noise = torch.randn_like(lat)
                t = torch.randint(0, sched.config.num_train_timesteps,
                                  (lat.size(0),), device=lat.device).long()
                noisy_lat = sched.add_noise(lat, noise, t)
                unet_in = torch.cat([noisy_lat, lat_m, msk_lat], dim=1)

                pred   = unet(unet_in, t, encoder_hidden_states=enc_hidden).sample
                target = (noise if sched.config.prediction_type=="epsilon"
                          else sched.get_velocity(lat, noise, t))
                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(unet.parameters(), 1.0)
                opt.step(); lr_sched.step(); opt.zero_grad()

                if acc.sync_gradients:
                    global_step += 1
                    acc.log({"train_loss": loss.item()}, step=global_step)
                    pbar.update(1); pbar.set_postfix(loss=f"{loss.item():.4f}")

                    if global_step % cfg.ckpt_steps == 0 and acc.is_main_process:
                        ck_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        acc.save_state(ck_dir)
                        acc.unwrap_model(unet).save_pretrained(os.path.join(ck_dir, "unet"))
                        logger.info(f"Saved checkpoint {ck_dir}")

                if global_step >= max_steps:
                    break

        # ------------------ VALIDATION ------------------------------------
        unet.eval()
        v_total, v_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                pix = batch["pixel_values"].to(acc.device, dtype=w_dtype)
                msk = batch["mask"].to(acc.device, dtype=w_dtype)

                lat  = vae.encode(pix).latent_dist.sample()*vae.config.scaling_factor
                lat_m= vae.encode(pix*(1-msk)).latent_dist.sample()*vae.config.scaling_factor
                msk_lat = F.interpolate(msk, scale_factor=1/8, mode="nearest")

                noise = torch.randn_like(lat)
                t = torch.randint(0, sched.config.num_train_timesteps,
                                  (lat.size(0),), device=lat.device).long()
                noisy_lat = sched.add_noise(lat, noise, t)
                unet_in = torch.cat([noisy_lat, lat_m, msk_lat], dim=1)
                pred = unet(unet_in, t, encoder_hidden_states=enc_hidden).sample
                tgt  = (noise if sched.config.prediction_type=="epsilon"
                        else sched.get_velocity(lat, noise, t))
                v_loss = F.mse_loss(pred.float(), tgt.float(), reduction="mean")
                v_total += v_loss.item(); v_batches += 1
        val_loss = v_total / max(1, v_batches)
        val_history.append(val_loss)

        if acc.is_main_process:
            logger.info(f"[Epoch {epoch:03d}]  val_loss = {val_loss:.6f}")
        acc.log({"val_loss": val_loss}, step=global_step)

        # ----------------- optional inference snapshot ---------------------
        if acc.is_main_process:
            inf_dir = os.path.join(cfg.output_dir, "inference_outputs")
            os.makedirs(inf_dir, exist_ok=True)
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                cfg.pretrained_model, vae=vae, text_encoder=txtenc,
                tokenizer=tok, scheduler=sched, safety_checker=None,
                torch_dtype=w_dtype)
            pipe.unet = acc.unwrap_model(unet)
            pipe = pipe.to(acc.device)
            if cfg.enable_xformers:
                try: pipe.enable_xformers_memory_efficient_attention()
                except Exception: pass

            samp_name = full_ds.image_filenames[0]
            base_img = Image.open(os.path.join(cfg.image_dir, samp_name)
                                 ).convert("RGB").resize((cfg.resolution,cfg.resolution))
            base_msk = Image.open(os.path.join(cfg.mask_dir,  samp_name)
                                 ).convert("L"  ).resize((cfg.resolution,cfg.resolution))
            res = run_inference(pipe, base_img, base_msk, prompt,
                                steps=50, guidance=7.5, dtype=w_dtype)
            res.save(os.path.join(inf_dir, f"epoch_{epoch}.png"))
            del pipe; torch.cuda.empty_cache()

        # epoch‑end checkpoint ---------------------------------------------
        if cfg.save_on_epoch_end and acc.is_main_process:
            ep_dir = os.path.join(cfg.output_dir, f"epoch-{epoch}")
            acc.save_state(ep_dir)
            acc.unwrap_model(unet).save_pretrained(os.path.join(ep_dir, "unet"))

    # --------------------- SUMMARY & CSV -----------------------------------
    if acc.is_main_process:
        dash = "-" * 34
        logger.info("\n" + dash + "\nVALIDATION‑LOSS HISTORY\n" + dash)
        logger.info("Epoch |  Val‑Loss")
        logger.info(dash)
        for ep, vl in enumerate(val_history):
            logger.info(f"{ep:5d} | {vl:.6f}")
        logger.info(dash)

        csv_p = os.path.join(cfg.output_dir, "val_history.csv")
        with open(csv_p, "w") as f:
            f.write("epoch,val_loss\n")
            for ep, vl in enumerate(val_history):
                f.write(f"{ep},{vl:.6f}\n")
        logger.info(f"Validation‑loss history written to {csv_p}")

        # final save --------------------------------------------------------
        final_dir = os.path.join(cfg.output_dir, "unet_final")
        acc.unwrap_model(unet).save_pretrained(final_dir)
        sched_cfg = os.path.join(cfg.pretrained_model, "scheduler",
                                 "scheduler_config.json")
        if os.path.exists(sched_cfg):
            shutil.copy(sched_cfg, os.path.join(final_dir, "scheduler_config.json"))
        logger.info(f"Saved final model to {final_dir}")

    acc.end_training()

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = TrainingConfig()
    main(cfg)
