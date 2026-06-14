# %%

import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import load_image 
from tqdm.auto import tqdm

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR_C2 = "/mnt/shared/dils/projects/microplastic/data/c2/imgs"
MASK_DIR_C1 = "/mnt/shared/dils/projects/microplastic/data/c1/masks"


TRAINED_UNET_PATH = "/mnt/shared/dils/projects/microplastic/code/stable_diffusion/model/unet_final" 

BASE_MODEL_PATH = "stabilityai/stable-diffusion-2-inpainting" 

IMAGE_SIZE = 512
NUM_INFERENCE_STEPS = 50 # number of denoising steps


WEIGHT_DTYPE = torch.float32
if DEVICE.type == "cuda":
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
         WEIGHT_DTYPE = torch.float16
    else:
        print("Warning: CUDA device does not support float16 well or not using mixed precision. Using float32.")


# --- Helper Functions ---
def load_random_file_path(directory, extensions=('.png', '.jpg', '.jpeg')): # returns random file path
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                all_files.append(os.path.join(root, file))
    
    if not all_files:
        raise FileNotFoundError(f"No files with extensions {extensions} found in {directory}")
    return random.choice(all_files)

def preprocess_image(image_path, size):
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None
        
  
    plot_image = raw_image.resize((size, size), Image.BILINEAR)

    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), 
    ])
    image_tensor = transform(raw_image).unsqueeze(0) 
    return plot_image, image_tensor

def preprocess_mask(mask_path, size, apply_random_transforms=True):
    try:
        raw_mask = Image.open(mask_path).convert("L") 
    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_path}")
        return None, None

    transform_ops = []
    if apply_random_transforms:
        transform_ops.append(transforms.RandomAffine(
            degrees=(-20, 20),
            translate=(0.15, 0.15), 
            shear=(-15, 15, -15, 15), 
            interpolation=transforms.InterpolationMode.NEAREST,
            fill=0
        ))
    
    transform_ops.append(transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST))
    
    pil_transform = transforms.Compose(transform_ops)
    transformed_pil_mask = pil_transform(raw_mask)

    tensor_transform = transforms.ToTensor()
    mask_tensor = tensor_transform(transformed_pil_mask)
    mask_tensor = (mask_tensor > 0.5).float()

    return transformed_pil_mask, mask_tensor.unsqueeze(0)

def postprocess_image_tensor(image_tensor):
  
    image = (image_tensor / 2 + 0.5).clamp(0, 1)
   
    image = image.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
   
    image = (image * 255).astype("uint8")
    return Image.fromarray(image)

def plot_results(original_image, transformed_mask, predicted_image, inpainted_composite_image):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image (C2)")
    axs[0].axis("off")

    axs[1].imshow(transformed_mask, cmap='gray')
    axs[1].set_title("Transformed Mask (from C1)")
    axs[1].axis("off")

    axs[2].imshow(predicted_image)
    axs[2].set_title("Model Prediction (Raw Output)")
    axs[2].axis("off")
    
    axs[3].imshow(inpainted_composite_image)
    axs[3].set_title("Final Inpainted Image")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()

# --- Main Inference Logic ---
def run_inference():
    print(f"Using device: {DEVICE}")
    print(f"Using weight dtype: {WEIGHT_DTYPE}")

   
    try:
        unet = UNet2DConditionModel.from_pretrained(TRAINED_UNET_PATH, torch_dtype=WEIGHT_DTYPE)
    except Exception as e:
        print(f"Error loading UNet from {TRAINED_UNET_PATH}: {e}")
        print("Please ensure TRAINED_UNET_PATH is correctly set to your trained model directory.")
        return

    vae = AutoencoderKL.from_pretrained(BASE_MODEL_PATH, subfolder="vae", torch_dtype=WEIGHT_DTYPE)
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_PATH, subfolder="scheduler")

    unet.to(DEVICE)
    vae.to(DEVICE)
    unet.eval()
    vae.eval() 


   
    try:
        image_path = load_random_file_path(IMAGE_DIR_C2)
        mask_path = load_random_file_path(MASK_DIR_C1)
        print(f"Selected image: {image_path}")
        print(f"Selected mask: {mask_path}")
    except FileNotFoundError as e:
        print(e)
        return

  
    original_pil_image, image_tensor = preprocess_image(image_path, IMAGE_SIZE)
    transformed_pil_mask, mask_tensor = preprocess_mask(mask_path, IMAGE_SIZE, apply_random_transforms=True)

    if image_tensor is None or mask_tensor is None:
        print("Failed to load or preprocess image/mask.")
        return

    image_tensor = image_tensor.to(DEVICE, dtype=WEIGHT_DTYPE)
    mask_tensor = mask_tensor.to(DEVICE, dtype=WEIGHT_DTYPE)


    masked_image_tensor = image_tensor * (1 - mask_tensor) 

    with torch.no_grad(): 
        masked_image_latents = vae.encode(masked_image_tensor).latent_dist.sample()
        masked_image_latents = masked_image_latents * vae.config.scaling_factor

    latent_mask = F.interpolate(
        mask_tensor, 
        scale_factor=1/8, 
        mode="nearest"
    )


    scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
    

    latents_shape = (
        image_tensor.shape[0], 
        vae.config.latent_channels,
        IMAGE_SIZE // 8, 
        IMAGE_SIZE // 8
    )
    current_latents = torch.randn(latents_shape, device=DEVICE, dtype=WEIGHT_DTYPE)
    

    bsz = current_latents.shape[0]
    cross_attention_dim = unet.config.cross_attention_dim 

    text_encoder_seq_len = 77 
    null_embeddings = torch.zeros(
        bsz, text_encoder_seq_len, cross_attention_dim,
        dtype=WEIGHT_DTYPE, 
        device=DEVICE
    )

    print("Starting denoising loop...")
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad(): 

            unet_input = torch.cat([current_latents, masked_image_latents, latent_mask], dim=1)
            
            noise_pred = unet(unet_input, t, encoder_hidden_states=null_embeddings).sample
            current_latents = scheduler.step(noise_pred, t, current_latents).prev_sample


    with torch.no_grad():
        current_latents = 1 / vae.config.scaling_factor * current_latents
        predicted_image_tensor = vae.decode(current_latents).sample
    

    predicted_pil_image = postprocess_image_tensor(predicted_image_tensor)

    if original_pil_image.size != (IMAGE_SIZE, IMAGE_SIZE):
        original_pil_image_resized = original_pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    else:
        original_pil_image_resized = original_pil_image

    mask_for_composite = transformed_pil_mask.convert("1").convert("RGB")

    inpainted_composite_image = Image.composite(predicted_pil_image, original_pil_image_resized, transformed_pil_mask.convert("L").point(lambda x: 255 if x > 128 else 0, '1'))

    print("Plotting results...")
    plot_results(original_pil_image_resized, transformed_pil_mask, predicted_pil_image, inpainted_composite_image)

# %%
for i in range(5):
    print(f"Running inference iteration {i+1}...")
    run_inference()
    print(f"Finished iteration {i+1}\n")
# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a 3‑row comparison plot (Original, Guiding Mask, Generated Image)
for micro‑plastic in‑painting.  Saves figure to ./figures/inference_grid.png
and shows it on‐screen.
"""

import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from tqdm.auto import tqdm

# ─────────────────── Configuration ────────────────────
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR_C2      = "/mnt/shared/dils/projects/microplastic/data/c2/imgs"
MASK_DIR_C1       = "/mnt/shared/dils/projects/microplastic/data/c1/masks"
TRAINED_UNET_PATH = "/mnt/shared/dils/projects/microplastic/code/stable_diffusion/model_old/unet_final"
BASE_MODEL_PATH   = "stabilityai/stable-diffusion-2-inpainting"
IMAGE_SIZE        = 512
NUM_INFER_STEPS   = 50
NUM_ROWS          = 3                    # how many rows you want
FIGURE_PATH       = Path("figures/inference_grid.png")

# silence most library warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ────────────────── Helper utilities ───────────────────
def random_file(root, exts=(".png", ".jpg", ".jpeg")):
    pool = [os.path.join(dp, f) for dp, _, fs in os.walk(root)
            for f in fs if f.lower().endswith(exts)]
    if not pool:
        raise FileNotFoundError(f"No images with {exts} in {root}")
    return random.choice(pool)

def pil_to_torch(pil, normalize=True):
    tf = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                            interpolation=transforms.InterpolationMode.BILINEAR),
          transforms.ToTensor()]
    if normalize:         # map to [-1,1] for SD‑style VAE
        tf.append(transforms.Normalize([0.5]*3, [0.5]*3))
    return transforms.Compose(tf)(pil)

def mask_to_torch(mask_pil):
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()])
    m = tf(mask_pil)
    return (m > 0.5).float()             # binarise

def postprocess(tensor):
    img = (tensor / 2 + 0.5).clamp(0, 1)               # [-1,1] → [0,1]
    img = img.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
    return Image.fromarray((img * 255).astype("uint8"))

# ─────────────── In‑painting one sample ────────────────
@torch.no_grad()
def inpaint_once(image_path, mask_path, unet, vae, scheduler, dtype):
    # load PILs
    pil_img  = Image.open(image_path).convert("RGB")
    pil_mask = Image.open(mask_path).convert("L")  # mono

    # torch tensors
    img_t   = pil_to_torch(pil_img).unsqueeze(0).to(DEVICE, dtype)
    mask_t  = mask_to_torch(pil_mask).unsqueeze(0).to(DEVICE, dtype)

    # knock the masked region out of the original image
    masked_img_t = img_t * (1 - mask_t)

    # encode masked image to latent space
    masked_latent = vae.encode(masked_img_t).latent_dist.sample()
    masked_latent = masked_latent * vae.config.scaling_factor

    # down‑sample the mask to latent resolution (8×)
    latent_mask = F.interpolate(mask_t, scale_factor=1/8, mode="nearest")

    # DDPM sampling
    scheduler.set_timesteps(NUM_INFER_STEPS, device=DEVICE)
    lat = torch.randn((1, vae.config.latent_channels,
                       IMAGE_SIZE//8, IMAGE_SIZE//8),
                      device=DEVICE, dtype=dtype)

    # null text‑conditioning
    null_embeddings = torch.zeros(
        1, 77, unet.config.cross_attention_dim, device=DEVICE, dtype=dtype)

    for t in tqdm(scheduler.timesteps, leave=False):
        unet_in = torch.cat([lat, masked_latent, latent_mask], dim=1)
        noise   = unet(unet_in, t, encoder_hidden_states=null_embeddings).sample
        lat     = scheduler.step(noise, t, lat).prev_sample

    # decode to RGB
    lat = lat / vae.config.scaling_factor
    pred_img_t = vae.decode(lat).sample
    pred_pil   = postprocess(pred_img_t)

    # blend prediction back onto original image with the mask
    pil_mask_bin = pil_mask.convert("L").point(lambda x: 255 if x > 128 else 0, "1")
    composite = Image.composite(pred_pil, pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)),
                                pil_mask_bin)

    return (pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)),
            pil_mask.resize((IMAGE_SIZE, IMAGE_SIZE)),
            composite)

# ──────────────────── Main script ──────────────────────
def main():
    fig_dir = FIGURE_PATH.parent
    fig_dir.mkdir(exist_ok=True, parents=True)

    # dtype (fp16 where it actually helps)
    dtype = (torch.float16 if DEVICE.type == "cuda"
             and torch.cuda.get_device_capability()[0] >= 7 else torch.float32)

    # load once
    unet      = UNet2DConditionModel.from_pretrained(TRAINED_UNET_PATH,
                                                     torch_dtype=dtype).to(DEVICE)
    vae       = AutoencoderKL.from_pretrained(BASE_MODEL_PATH, subfolder="vae",
                                              torch_dtype=dtype).to(DEVICE)
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_PATH, subfolder="scheduler")

    unet.eval(); vae.eval()

    originals, masks, generated = [], [], []
    for _ in range(NUM_ROWS):
        img_p = random_file(IMAGE_DIR_C2)
        msk_p = random_file(MASK_DIR_C1)
        o, m, g = inpaint_once(img_p, msk_p, unet, vae, scheduler, dtype)
        originals.append(o); masks.append(m); generated.append(g)

    # ─────────────── Plot grid ────────────────
    fig, axs = plt.subplots(NUM_ROWS, 3, figsize=(9, 3*NUM_ROWS))
    col_titles = ["Original Image", "Guiding Mask", "Generated Image"]
    for c, title in enumerate(col_titles):
        axs[0, c].set_title(title, fontsize=14, pad=10)

    for r in range(NUM_ROWS):
        axs[r, 0].imshow(originals[r]);  axs[r, 0].axis("off")
        axs[r, 1].imshow(masks[r], cmap="gray"); axs[r, 1].axis("off")
        axs[r, 2].imshow(generated[r]);  axs[r, 2].axis("off")

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to {FIGURE_PATH.resolve()}")

if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Produce a comparison grid for micro‑plastic in‑painting:
         Original | Guiding Mask | Generated Image
Rows = NUM_ROWS random samples.
"""

import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from tqdm.auto import tqdm

# ────────────────────────── Configuration ──────────────────────────
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR_C2      = "/mnt/shared/dils/projects/microplastic/data/c2/imgs"
MASK_DIR_C1       = "/mnt/shared/dils/projects/microplastic/data/c1/masks"
TRAINED_UNET_PATH = "/mnt/shared/dils/projects/microplastic/code/stable_diffusion/model_old/unet_final"
BASE_MODEL_PATH   = "stabilityai/stable-diffusion-2-inpainting"

IMAGE_SIZE      = 512          # square side length fed to VAE/UNet
NUM_INFER_STEPS = 50           # DDPM denoising steps
NUM_ROWS        = 3            # rows in the final grid
FIGURE_PATH     = Path("figures/inference_grid.png")

# Hide most library warnings for a clean log
warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────── Helper utilities ────────────────────────
def random_file(root, exts=(".png", ".jpg", ".jpeg")):
    pool = [os.path.join(dp, f) for dp, _, fs in os.walk(root)
            for f in fs if f.lower().endswith(exts)]
    if not pool:
        raise FileNotFoundError(f"No images with {exts} in {root}")
    return random.choice(pool)

def pil_to_torch(pil, normalize=True):
    tf = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ]
    if normalize:
        tf.append(transforms.Normalize([0.5] * 3, [0.5] * 3))  # → [-1,1]
    return transforms.Compose(tf)(pil)

def mask_to_torch(pil_mask):
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    m = tf(pil_mask)
    return (m > 0.5).float()   # binarise

def postprocess(tensor):
    img = (tensor / 2 + 0.5).clamp(0, 1)                        # → [0,1]
    img = img.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
    return Image.fromarray((img * 255).astype("uint8"))

# ────────────────── In‑paint a single (image, mask) ─────────────────
@torch.no_grad()
def inpaint_once(image_path, mask_path, unet, vae, scheduler, dtype):
    # 1. Load & resize PILs so EVERYTHING is IMAGE_SIZE×IMAGE_SIZE
    pil_img  = Image.open(image_path).convert("RGB")
    pil_mask = Image.open(mask_path).convert("L")  # mono / 8‑bit
    pil_img  = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    pil_mask = pil_mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

    # 2. Torch tensors
    img_t  = pil_to_torch(pil_img).unsqueeze(0).to(DEVICE, dtype)
    mask_t = mask_to_torch(pil_mask).unsqueeze(0).to(DEVICE, dtype)

    # knock out masked region
    masked_img_t = img_t * (1 - mask_t)

    # 3. Encode to latent space
    masked_latent = vae.encode(masked_img_t).latent_dist.sample()
    masked_latent = masked_latent * vae.config.scaling_factor

    # 4. Down‑sample binary mask to latent resolution (×8)
    latent_mask = F.interpolate(mask_t, scale_factor=1 / 8, mode="nearest")

    # 5. DDPM sampling loop
    scheduler.set_timesteps(NUM_INFER_STEPS, device=DEVICE)
    lat = torch.randn(
        (1, vae.config.latent_channels, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
        device=DEVICE,
        dtype=dtype,
    )

    null_emb = torch.zeros(
        1, 77, unet.config.cross_attention_dim, device=DEVICE, dtype=dtype
    )

    for t in tqdm(scheduler.timesteps, leave=False):
        noise = unet(
            torch.cat([lat, masked_latent, latent_mask], dim=1),
            t,
            encoder_hidden_states=null_emb,
        ).sample
        lat = scheduler.step(noise, t, lat).prev_sample

    # 6. Decode → RGB
    lat = lat / vae.config.scaling_factor
    pred_pil = postprocess(vae.decode(lat).sample)

    # 7. Composite prediction back onto original background
    mask_bin = pil_mask.point(lambda x: 255 if x > 128 else 0, "1")  # strict binary
    composite = Image.composite(pred_pil, pil_img, mask_bin)

    return pil_img, pil_mask, composite

# ────────────────────────────── Main ────────────────────────────────
def main():
    # Create ./figures
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # dtype: fp16 when the GPU can handle it
    dtype = (torch.float16 if DEVICE.type == "cuda" and
             torch.cuda.get_device_capability()[0] >= 7 else torch.float32)

    # Load models once
    print("Loading models …")
    unet = UNet2DConditionModel.from_pretrained(TRAINED_UNET_PATH,
                                                torch_dtype=dtype).to(DEVICE)
    vae  = AutoencoderKL.from_pretrained(BASE_MODEL_PATH,
                                         subfolder="vae",
                                         torch_dtype=dtype).to(DEVICE)
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_PATH,
                                              subfolder="scheduler")
    unet.eval(); vae.eval()

    originals, masks, generated = [], [], []
    for _ in range(NUM_ROWS):
        img_p = random_file(IMAGE_DIR_C2)
        msk_p = random_file(MASK_DIR_C1)
        o, m, g = inpaint_once(img_p, msk_p, unet, vae, scheduler, dtype)
        originals.append(o); masks.append(m); generated.append(g)

    # ─────────────── Create matplotlib grid ────────────────
    fig, axs = plt.subplots(NUM_ROWS, 3, figsize=(9, 3 * NUM_ROWS))
    col_titles = ["Original Image", "Guiding Mask", "Generated Image"]
    for c, t in enumerate(col_titles):
        axs[0, c].set_title(t, fontsize=14, pad=10)

    for r in range(NUM_ROWS):
        axs[r, 0].imshow(originals[r]);  axs[r, 0].axis("off")
        axs[r, 1].imshow(masks[r], cmap="gray"); axs[r, 1].axis("off")
        axs[r, 2].imshow(generated[r]);  axs[r, 2].axis("off")

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"\n✓ Figure saved to: {FIGURE_PATH.resolve()}\n")

if __name__ == "__main__":
    main()

# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a comparison grid (Original • Mask • Generated) for micro‑plastic
in‑painting, with optional hash‑based image selection.

If HASH_LIST[i] is  None or missing, the i‑th row uses a random image.
"""

import os
import random
import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from tqdm.auto import tqdm

# ─────────────────────── User‑editable section ──────────────────────
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR_C2      = "/mnt/shared/dils/projects/microplastic/data/c2/imgs"
MASK_DIR_C1       = "/mnt/shared/dils/projects/microplastic/data/c1/masks"

# model is now in model_old/
TRAINED_UNET_PATH = (
    "/mnt/shared/dils/projects/microplastic/code/stable_diffusion/model_text_dilated/unet_final"
)
BASE_MODEL_PATH   = "stabilityai/stable-diffusion-2-inpainting"

IMAGE_SIZE      = 512
NUM_INFER_STEPS = 50
NUM_ROWS        = 3                          # number of rows in the grid

FIGURE_PATH     = Path("figures/inference_grid.png")

# Optional: put hashes here to pin rows.  Use None for “random”.
HASH_LIST: List[Optional[str]] = [None, None, None]          # e.g. ["4f1b9d02", None, "c8ad7a91"]
# ────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────── Helper utilities ─────────────────────────
def random_file(root, exts=(".png", ".jpg", ".jpeg")):
    pool = [os.path.join(dp, f) for dp, _, fs in os.walk(root)
            for f in fs if f.lower().endswith(exts)]
    if not pool:
        raise FileNotFoundError(f"No images with {exts} in {root}")
    return random.choice(pool)

def find_image_by_hash(root, h, exts=(".png", ".jpg", ".jpeg")) -> Optional[str]:
    """Return the first file whose stem starts with the given hash h."""
    for dp, _, fs in os.walk(root):
        for f in fs:
            stem, ext = os.path.splitext(f)
            if stem.startswith(h) and f.lower().endswith(exts):
                return os.path.join(dp, f)
    return None

def extract_hash(path: str) -> str:
    """Filename without extension (useful as the 'assigned hash')."""
    return Path(path).stem

def pil_to_torch(pil, normalize=True):
    tf = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
    if normalize:
        tf.append(transforms.Normalize([0.5] * 3, [0.5] * 3))
    return transforms.Compose(tf)(pil)

def mask_to_torch(pil_mask):
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    m = tf(pil_mask)
    return (m > 0.5).float()

def postprocess(tensor):
    img = (tensor / 2 + 0.5).clamp(0, 1)
    img = img.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
    return Image.fromarray((img * 255).astype("uint8"))

# ────────────────── In‑paint a single (image, mask) ─────────────────
@torch.no_grad()
def inpaint_once(image_path, mask_path, unet, vae, scheduler, dtype):
    pil_img  = Image.open(image_path).convert("RGB")
    pil_mask = Image.open(mask_path).convert("L")

    pil_img  = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    pil_mask = pil_mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

    img_t  = pil_to_torch(pil_img).unsqueeze(0).to(DEVICE, dtype)
    mask_t = mask_to_torch(pil_mask).unsqueeze(0).to(DEVICE, dtype)

    masked_img_t = img_t * (1 - mask_t)

    masked_latent = vae.encode(masked_img_t).latent_dist.sample()
    masked_latent = masked_latent * vae.config.scaling_factor

    latent_mask = F.interpolate(mask_t, scale_factor=1 / 8, mode="nearest")

    scheduler.set_timesteps(NUM_INFER_STEPS, device=DEVICE)
    lat = torch.randn(
        (1, vae.config.latent_channels, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
        device=DEVICE,
        dtype=dtype,
    )

    null_emb = torch.zeros(
        1, 77, unet.config.cross_attention_dim, device=DEVICE, dtype=dtype
    )

    for t in tqdm(scheduler.timesteps, leave=False):
        noise = unet(
            torch.cat([lat, masked_latent, latent_mask], dim=1),
            t,
            encoder_hidden_states=null_emb,
        ).sample
        lat = scheduler.step(noise, t, lat).prev_sample

    lat = lat / vae.config.scaling_factor
    pred_pil = postprocess(vae.decode(lat).sample)

    mask_bin = pil_mask.point(lambda x: 255 if x > 128 else 0, "1")
    composite = Image.composite(pred_pil, pil_img, mask_bin)

    return pil_img, pil_mask, composite

# ────────────────────────────── Main ────────────────────────────────
def main():
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    dtype = (
        torch.float16
        if DEVICE.type == "cuda" and torch.cuda.get_device_capability()[0] >= 7
        else torch.float32
    )

    print("Loading models …")
    unet = UNet2DConditionModel.from_pretrained(
        TRAINED_UNET_PATH, torch_dtype=dtype
    ).to(DEVICE)
    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL_PATH, subfolder="vae", torch_dtype=dtype
    ).to(DEVICE)
    scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_PATH, subfolder="scheduler")
    unet.eval()
    vae.eval()

    originals, masks, generated, hashes = [], [], [], []

    for row_idx in range(NUM_ROWS):
        # Decide which image to use for this row
        if row_idx < len(HASH_LIST) and HASH_LIST[row_idx]:
            img_path = find_image_by_hash(IMAGE_DIR_C2, HASH_LIST[row_idx])
            if img_path is None:
                print(
                    f"[row {row_idx}] ⚠️  Hash “{HASH_LIST[row_idx]}” not found "
                    "— falling back to random."
                )
                img_path = random_file(IMAGE_DIR_C2)
        else:
            img_path = random_file(IMAGE_DIR_C2)

        mask_path = random_file(MASK_DIR_C1)

        o, m, g = inpaint_once(img_path, mask_path, unet, vae, scheduler, dtype)
        originals.append(o)
        masks.append(m)
        generated.append(g)
        hashes.append(extract_hash(img_path))

    # ─────────────── Plot grid ────────────────
    fig, axs = plt.subplots(NUM_ROWS, 3, figsize=(9, 3 * NUM_ROWS))
    col_titles = ["Original Image", "Guiding Mask", "Generated Image"]
    for c, t in enumerate(col_titles):
        axs[0, c].set_title(t, fontsize=14, pad=10)

    for r in range(NUM_ROWS):
        axs[r, 0].imshow(originals[r])
        axs[r, 0].axis("off")
        axs[r, 1].imshow(masks[r], cmap="gray")
        axs[r, 1].axis("off")
        axs[r, 2].imshow(generated[r])
        axs[r, 2].axis("off")

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.show()

    # Print hashes used
    print("\nHashes for images in this figure:")
    for idx, h in enumerate(hashes):
        print(f"  row {idx}: {h}")
    print(f"\n✓ Figure saved to: {FIGURE_PATH.resolve()}\n")


if __name__ == "__main__":
    for i in range(5):
        main()

# %%
