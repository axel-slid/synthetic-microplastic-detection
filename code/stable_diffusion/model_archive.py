# %% old version of the model which doesnt have the text conditioning and the dilated masks path

import os
import argparse 
import math
import shutil

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

check_min_version("0.28.0")

logger = get_logger(__name__, log_level="INFO")

class MicroplasticInpaintingDataset(Dataset):
    def __init__(self, image_folder, mask_folder, image_size=(512, 512), center_crop=False, random_flip=False):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.image_size = image_size

        self.image_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size) if center_crop else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST), # Use NEAREST for masks
            transforms.CenterCrop(image_size) if center_crop else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]

        mask_name = img_name

        image_path = os.path.join(self.image_folder, img_name)
        mask_path = os.path.join(self.mask_folder, mask_name)

        try:
            original_image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except FileNotFoundError:
            logger.error(f"File not found: {image_path} or {mask_path}")
            return self.__getitem__((idx + 1) % len(self))

        original_image_tensor = self.image_transforms(original_image)
        mask_tensor = self.mask_transforms(mask)

        return {
            "pixel_values": original_image_tensor,
            "mask": mask_tensor
        }


class TrainingConfig:
    def __init__(self):
      
        self.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-inpainting"
        self.revision = None
        self.train_image_data_dir = "/mnt/shared/dils/projects/microplastic/data/c1/imgs"
        self.train_mask_data_dir = "/mnt/shared/dils/projects/microplastic/data/c1/masks"
        self.output_dir = "microplastic-inpainting-model-script"
        self.logging_dir = "logs"
        self.report_to = "tensorboard"

   
        self.resolution = 512
        self.center_crop = False
        self.random_flip = True
        self.train_batch_size = 1
        self.dataloader_num_workers = 2


        self.num_train_epochs = 50
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


        self.mixed_precision = "fp16"
        self.enable_xformers_memory_efficient_attention = True
        self.seed = 42

        self.checkpointing_steps = 500
        self.save_on_epoch_end = True


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)

    if unet.config.in_channels != 9:
        logger.warning(
            f"Loaded U-Net has {unet.config.in_channels} input channels, but inpainting typically uses 9. "
            f"Ensure your model {args.pretrained_model_name_or_path} is an inpainting model."
        )

    vae.requires_grad_(False)
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}. Continuing without it.")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = MicroplasticInpaintingDataset(
        image_folder=args.train_image_data_dir,
        mask_folder=args.train_mask_data_dir,
        image_size=(args.resolution, args.resolution),
        center_crop=args.center_crop,
        random_flip=args.random_flip,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                mask = batch["mask"].to(dtype=weight_dtype)

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                masked_image_for_vae = pixel_values * (1 - mask)
                masked_image_latents = vae.encode(masked_image_for_vae).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae.config.scaling_factor

                latent_mask = F.interpolate(
                    mask, scale_factor=1/8, mode="nearest"
                )

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                unet_input = torch.cat([noisy_latents, masked_image_latents, latent_mask], dim=1)

                # Prepare null embeddings for encoder_hidden_states
                # SD 2.x models expect cross_attention_dim of 1024 and often a sequence length of 77
                cross_attention_dim = unet.config.cross_attention_dim
                # Default sequence length for CLIP-based text encoders used with Stable Diffusion
                # If you had a tokenizer, you might use tokenizer.model_max_length
                text_encoder_seq_len = 77

                null_embeddings = torch.zeros(
                    bsz, text_encoder_seq_len, cross_attention_dim,
                    dtype=unet_input.dtype, # Use the same dtype as other inputs to U-Net
                    device=unet_input.device
                )

                model_pred = unet(unet_input, timesteps, encoder_hidden_states=null_embeddings).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        unwrapped_unet.save_pretrained(os.path.join(save_path, "unet"))
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.save_on_epoch_end:
                save_path = os.path.join(args.output_dir, f"epoch-{epoch}")
                accelerator.save_state(save_path)
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet.save_pretrained(os.path.join(save_path, "unet"))
                logger.info(f"Saved state to {save_path} at end of epoch {epoch}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(os.path.join(args.output_dir, "unet_final"))

        scheduler_config_path = os.path.join(args.pretrained_model_name_or_path, "scheduler", "scheduler_config.json")
        if os.path.exists(scheduler_config_path):
            output_scheduler_path = os.path.join(args.output_dir, "unet_final", "scheduler_config.json")
            os.makedirs(os.path.dirname(output_scheduler_path), exist_ok=True)
            shutil.copy(scheduler_config_path, output_scheduler_path)

    accelerator.end_training()

config = TrainingConfig()

if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir, exist_ok=True)

main(config)





# %%

import os
import argparse
import math
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionInpaintPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from transformers import CLIPTextModel, CLIPTokenizer

from PIL import Image
from tqdm.auto import tqdm

check_min_version("0.28.0")

logger = get_logger(__name__, log_level="INFO")


class MicroplasticInpaintingDataset(Dataset):
    def __init__(self, image_folder, mask_folder, image_size=(512, 512)):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.image_size = image_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_folder, img_name)
        mask_path = os.path.join(self.mask_folder, img_name)

        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except FileNotFoundError:
            logger.error(f"Missing file: {image_path} or {mask_path}")
            return self.__getitem__((idx + 1) % len(self))

        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        seed = torch.seed()
        torch.manual_seed(seed)
        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if torch.rand(1) > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        angle = torch.randint(-15, 15, (1,)).item()
        scale = torch.FloatTensor(1).uniform_(0.9, 1.1).item()
        translate = [0, 0]
        image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=0)
        mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=0)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, [0.5], [0.5])
        mask_tensor = TF.to_tensor(mask)

        return {
            "pixel_values": image_tensor,
            "mask": mask_tensor
        }


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

        self.mixed_precision = "fp16"
        self.enable_xformers_memory_efficient_attention = True
        self.seed = 42

        self.checkpointing_steps = 500
        self.save_on_epoch_end = True


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)

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

    train_dataset = MicroplasticInpaintingDataset(
        image_folder=args.train_image_data_dir,
        mask_folder=args.train_mask_data_dir,
        image_size=(args.resolution, args.resolution)
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else (
        torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    prompt = (
        "microscopic plastic filaments, synthetic fibers, transparent polymer threads, "
        "small colorful microplastic fragments embedded in organic matter, detailed, close-up"
    )
    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    with torch.no_grad():
        encoder_hidden_states = text_encoder(text_inputs.input_ids.to(accelerator.device))[0]

    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                mask = batch["mask"].to(accelerator.device, dtype=weight_dtype)

                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                masked_latents = vae.encode(pixel_values * (1 - mask)).latent_dist.sample() * vae.config.scaling_factor
                latent_mask = F.interpolate(mask, scale_factor=1 / 8, mode="nearest")

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                unet_input = torch.cat([noisy_latents, masked_latents, latent_mask], dim=1)

                model_pred = unet(unet_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    accelerator.log({"train_loss": loss.detach().item()}, step=global_step)
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item(), step=global_step)

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet"))
                            logger.info(f"Saved checkpoint to {save_path}")

                if global_step >= args.max_train_steps:
                    break

        if args.save_on_epoch_end and accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"epoch-{epoch}")
            accelerator.save_state(save_path)
            accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet"))
            logger.info(f"Saved epoch model to {save_path}")

    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "unet_final")
        accelerator.unwrap_model(unet).save_pretrained(final_path)
        scheduler_config = os.path.join(args.pretrained_model_name_or_path, "scheduler", "scheduler_config.json")
        if os.path.exists(scheduler_config):
            shutil.copy(scheduler_config, os.path.join(final_path, "scheduler_config.json"))
        logger.info(f"Saved final model to {final_path}")

    accelerator.end_training()


config = TrainingConfig()
os.makedirs(config.output_dir, exist_ok=True)
main(config)

# %%