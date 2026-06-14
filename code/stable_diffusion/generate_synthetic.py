# %%

import os
import random
import matplotlib.pyplot as plt # Keep for potential debugging, though not used in batch
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
# from diffusers.utils import load_image # Not strictly needed if PIL is used consistently
from tqdm.auto import tqdm
import json # For logging transform details

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR_C2 = "/mnt/shared/dils/projects/microplastic/data/c2/imgs"
MASK_DIR_C1 = "/mnt/shared/dils/projects/microplastic/data/c1/masks_dilated"

# IMPORTANT: Set this path to your trained UNet model directory
TRAINED_UNET_PATH = "/mnt/shared/dils/projects/microplastic/code/stable_diffusion/model/unet_final" 

BASE_MODEL_PATH = "stabilityai/stable-diffusion-2-inpainting" # Used for VAE and scheduler

IMAGE_SIZE = 512
NUM_INFERENCE_STEPS = 50 # Number of denoising steps
NUM_IMAGES_TO_GENERATE = 10000 # Number of images to generate

# Determine weight dtype
WEIGHT_DTYPE = torch.float32
if DEVICE.type == "cuda":
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
         WEIGHT_DTYPE = torch.float16
    else:
        print("Warning: CUDA device does not support float16 well. Using float32.")


# --- Helper Functions ---
def load_all_file_paths(directory, extensions=('.png', '.jpg', '.jpeg'), toplevel_only=False):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    all_files = []
    if toplevel_only:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path) and item.lower().endswith(extensions):
                all_files.append(item_path)
    else:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(extensions):
                    all_files.append(os.path.join(root, file))
    
    if not all_files:
        raise FileNotFoundError(f"No files with extensions {extensions} found in {directory}")
    return all_files

def preprocess_image(image_path, size):
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None
        
    # This resized PIL image will be used for compositing later
    resized_pil_image = raw_image.resize((size, size), Image.BILINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(), # Converts the resized_pil_image
        transforms.Normalize([0.5], [0.5]), # Normalize to [-1, 1]
    ])
    # Apply ToTensor and Normalize on the already resized PIL image
    image_tensor = transform(resized_pil_image).unsqueeze(0) # Add batch dim
    return resized_pil_image, image_tensor

def preprocess_mask(mask_path, size, apply_random_transforms=True):
    try:
        raw_mask = Image.open(mask_path).convert("L") # Ensure grayscale
    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_path}")
        return None, None, {}

    transform_details = {"applied": apply_random_transforms}
    
    pil_transform_ops = []
    if apply_random_transforms:
        # Define ranges for random affine transformations
        degrees_range = (-20, 20)
        translate_fraction_range = (0.15, 0.15) 
        shear_degrees_range = (-15, 15, -15, 15) # x-shear min/max, y-shear min/max

        # Sample actual parameters for affine transformation
        angle = random.uniform(*degrees_range)
        
        max_dx = translate_fraction_range[0] * size
        max_dy = translate_fraction_range[1] * size
        # Ensure translations are integers for transforms.functional.affine if it expects that
        # For PIL transforms, float is fine. For functional.affine, list of floats is fine.
        translations = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
        
        shear_x_degrees = random.uniform(shear_degrees_range[0], shear_degrees_range[1])
        shear_y_degrees = random.uniform(shear_degrees_range[2], shear_degrees_range[3])
        shear_params = [shear_x_degrees, shear_y_degrees]

        pil_transform_ops.append(transforms.Lambda(lambda img: transforms.functional.affine(
            img, 
            angle=angle, 
            translate=list(translations),
            scale=1.0, # No random scaling in the original example
            shear=list(shear_params), 
            interpolation=transforms.InterpolationMode.NEAREST,
            fill=0  # Fill with black
        )))
        transform_details.update({
            "type": "random_affine",
            "angle_degrees": angle,
            "translation_pixels": translations,
            "shear_degrees": shear_params, # [x_shear, y_shear]
            "interpolation": "NEAREST",
            "fill_value": 0
        })
    
    pil_transform_ops.append(transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST))
    
    pil_transform_pipeline = transforms.Compose(pil_transform_ops)
    transformed_pil_mask = pil_transform_pipeline(raw_mask)

    # Convert to tensor and ensure binary
    tensor_transform = transforms.ToTensor() # Converts L mode PIL to [0, 1] tensor
    mask_tensor = tensor_transform(transformed_pil_mask)
    mask_tensor = (mask_tensor > 0.5).float() # Binarize: values > 0.5 become 1.0, else 0.0

    return transformed_pil_mask, mask_tensor.unsqueeze(0), transform_details

def postprocess_image_tensor(image_tensor):
    image = (image_tensor / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
    image = (image * 255).astype("uint8")
    return Image.fromarray(image)

# --- Core Inpainting Function ---
def generate_single_inpainted_image(
    original_image_path, 
    mask_path, 
    vae, 
    unet, 
    scheduler, 
    device, 
    weight_dtype, 
    image_size, 
    num_inference_steps,
    apply_random_mask_transforms=True
):
    # 1. Preprocess image and mask
    # original_pil_image_for_compositing is the PIL Image, resized, RGB
    original_pil_image_for_compositing, image_tensor = preprocess_image(original_image_path, image_size)
    
    # transformed_pil_mask_for_logging is the PIL Image of the mask after transforms, L mode
    # transform_details is a dict
    transformed_pil_mask_for_logging, mask_tensor, transform_details = preprocess_mask(
        mask_path, image_size, apply_random_transforms=apply_random_mask_transforms
    )

    if image_tensor is None or mask_tensor is None:
        print(f"Skipping due to preprocessing failure for: {original_image_path} or {mask_path}")
        return None, None, {} 

    image_tensor = image_tensor.to(device, dtype=weight_dtype)
    mask_tensor = mask_tensor.to(device, dtype=weight_dtype)

    # 2. Prepare inputs for U-Net
    masked_image_tensor = image_tensor * (1 - mask_tensor)
    with torch.no_grad():
        masked_image_latents = vae.encode(masked_image_tensor).latent_dist.sample()
        masked_image_latents = masked_image_latents * vae.config.scaling_factor
    
    latent_mask = F.interpolate(mask_tensor, scale_factor=1/8, mode="nearest")

    # 3. Denoising loop
    scheduler.set_timesteps(num_inference_steps, device=device)
    latents_shape = (
        image_tensor.shape[0], 
        vae.config.latent_channels,
        image_size // 8, 
        image_size // 8
    )
    current_latents = torch.randn(latents_shape, device=device, dtype=weight_dtype)
    
    bsz = current_latents.shape[0]
    cross_attention_dim = unet.config.cross_attention_dim 
    text_encoder_seq_len = 77 
    null_embeddings = torch.zeros(
        bsz, text_encoder_seq_len, cross_attention_dim,
        dtype=weight_dtype, device=device
    )

    # Consider adding tqdm(scheduler.timesteps, leave=False) if inner loop progress is needed
    for t in scheduler.timesteps:
        with torch.no_grad():
            unet_input = torch.cat([current_latents, masked_image_latents, latent_mask], dim=1)
            noise_pred = unet(unet_input, t, encoder_hidden_states=null_embeddings).sample
            current_latents = scheduler.step(noise_pred, t, current_latents).prev_sample

    # 4. Decode latents to image
    with torch.no_grad():
        current_latents = 1 / vae.config.scaling_factor * current_latents
        predicted_image_tensor = vae.decode(current_latents).sample
    
    predicted_pil_image = postprocess_image_tensor(predicted_image_tensor)

    # 5. Create a composite inpainted image
    # The mask for Image.composite needs to be 'L' or '1' and binarized (0 or 255)
    # transformed_pil_mask_for_logging is already 'L' mode from preprocess_mask.
    # We need to ensure it's binarized appropriately for compositing.
    composite_mask_pil = transformed_pil_mask_for_logging.point(lambda x: 255 if x > 128 else 0, '1')

    inpainted_composite_image = Image.composite(
        predicted_pil_image, 
        original_pil_image_for_compositing, # This is the resized original PIL image
        composite_mask_pil # This is the binarized transformed PIL mask
    )
    
    return inpainted_composite_image, transformed_pil_mask_for_logging, transform_details


# --- Main Batch Generation Logic ---
def batch_generate_inpainted_images():
    print(f"Starting batch generation...")
    print(f"Using device: {DEVICE}")
    print(f"Using weight dtype: {WEIGHT_DTYPE}")

    # 1. Setup output directories
    base_output_dir = IMAGE_DIR_C2 # Output directories will be inside IMAGE_DIR_C2
    output_dir_c2_gen = os.path.join(base_output_dir, "c2_gen")
    output_dir_c2_gen_mask = os.path.join(base_output_dir, "c2_gen_mask") # Parallel directory for masks

    os.makedirs(output_dir_c2_gen, exist_ok=True)
    os.makedirs(output_dir_c2_gen_mask, exist_ok=True) # Create mask directory

    log_file_path = os.path.join(output_dir_c2_gen, "generation_log.txt") # Log remains in c2_gen
    print(f"Generated images will be saved to: {output_dir_c2_gen}")
    print(f"Corresponding transformed masks will be saved to: {output_dir_c2_gen_mask}")

    # 2. Load models (once)
    print("Loading models...")
    try:
        unet = UNet2DConditionModel.from_pretrained(TRAINED_UNET_PATH, torch_dtype=WEIGHT_DTYPE)
        vae = AutoencoderKL.from_pretrained(BASE_MODEL_PATH, subfolder="vae", torch_dtype=WEIGHT_DTYPE)
        scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_PATH, subfolder="scheduler")
        
        unet.to(DEVICE).eval()
        vae.to(DEVICE).eval()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"FATAL: Error loading models: {e}")
        print("Please ensure TRAINED_UNET_PATH and BASE_MODEL_PATH are correct.")
        return

    # 3. Load all image and mask paths (once)
    print("Loading file paths...")
    try:
        # Load images from top-level of IMAGE_DIR_C2 to avoid picking from c2_gen
        all_image_paths = load_all_file_paths(IMAGE_DIR_C2, toplevel_only=True)
        # Load masks, assuming they might be in subdirs of MASK_DIR_C1 if not flat
        all_mask_paths = load_all_file_paths(MASK_DIR_C1, toplevel_only=False) 
        
        if not all_image_paths:
            print(f"FATAL: No images found in {IMAGE_DIR_C2}. Please check the path and ensure it contains images directly (not in subfolders).")
            return
        if not all_mask_paths:
            print(f"FATAL: No masks found in {MASK_DIR_C1}. Please check the path.")
            return
        print(f"Found {len(all_image_paths)} images in {IMAGE_DIR_C2}")
        print(f"Found {len(all_mask_paths)} masks in {MASK_DIR_C1}")
    except FileNotFoundError as e:
        print(f"FATAL: {e}")
        return
    except Exception as e:
        print(f"FATAL: Error loading file paths: {e}")
        return

    # 4. Generation loop
    print(f"Starting generation of {NUM_IMAGES_TO_GENERATE} images...")
    with open(log_file_path, "w") as log_file:
        log_file.write("GeneratedFileName,OriginalImagePath,MaskPath,MaskTransformDetails\n") # CSV Header

        for i in tqdm(range(NUM_IMAGES_TO_GENERATE), desc="Generating Images"):
            original_image_path = random.choice(all_image_paths)
            mask_path = random.choice(all_mask_paths)
            
            # Filename will be the same for the generated image and its corresponding mask
            generated_filename_base = f"generated_{i:05d}"
            generated_image_filename = f"{generated_filename_base}.png"
            output_image_save_path = os.path.join(output_dir_c2_gen, generated_image_filename)
            output_mask_save_path = os.path.join(output_dir_c2_gen_mask, generated_image_filename) # Mask saved with same name

            try:
                inpainted_image, transformed_mask_pil, transform_details = generate_single_inpainted_image(
                    original_image_path,
                    mask_path,
                    vae,
                    unet,
                    scheduler,
                    DEVICE,
                    WEIGHT_DTYPE,
                    IMAGE_SIZE,
                    NUM_INFERENCE_STEPS,
                    apply_random_mask_transforms=True 
                )

                if inpainted_image and transformed_mask_pil:
                    inpainted_image.save(output_image_save_path)
                    print(f"Generated image saved to: {output_image_save_path}")
                    transformed_mask_pil.save(output_mask_save_path) # Save the transformed mask

                    # Sanitize paths for CSV if they contain commas
                    log_original_path = original_image_path.replace(",", ";")
                    log_mask_path = mask_path.replace(",", ";")
                    log_entry = f"{generated_image_filename},{log_original_path},{log_mask_path},{json.dumps(transform_details)}\n"
                    log_file.write(log_entry)
                else:
                    print(f"Warning: Generation failed or mask not returned for image index {i} (orig: {original_image_path}, mask: {mask_path}). Skipping.")
            
            except Exception as e:
                print(f"Error during generation for image index {i} (orig: {original_image_path}, mask: {mask_path}): {e}")
                # Optionally, log this specific error to the log file or a separate error log
                log_file.write(f"{generated_image_filename},{original_image_path},{mask_path},ERROR: {str(e).replace(',', ';').replace(chr(10),' ').replace(chr(13),' ')}\n")


    print(f"\nBatch generation complete.")
    print(f"Generated {NUM_IMAGES_TO_GENERATE} images (or attempted to) in {output_dir_c2_gen}")
    print(f"Transformed masks saved in {output_dir_c2_gen_mask}")
    print(f"Log saved to {log_file_path}")


# --- Entry Point ---
if __name__ == "__main__":
    # Basic path checks before starting
    paths_ok = True
    if not os.path.exists(IMAGE_DIR_C2):
        print(f"ERROR: Image directory C2 not found: {IMAGE_DIR_C2}")
        paths_ok = False
    if not os.path.exists(MASK_DIR_C1):
        print(f"ERROR: Mask directory C1 not found: {MASK_DIR_C1}")
        paths_ok = False
    if not os.path.exists(TRAINED_UNET_PATH): # Check if the directory exists
        print(f"ERROR: Trained UNET path not found: {TRAINED_UNET_PATH}. Please set it correctly.")
        paths_ok = False
    elif not os.path.exists(os.path.join(TRAINED_UNET_PATH, "config.json")): # Check for a key file
         print(f"ERROR: Trained UNET path {TRAINED_UNET_PATH} does not seem to contain a model (missing config.json).")
         paths_ok = False


    if paths_ok:
        batch_generate_inpainted_images()
    else:
        print("Please correct the paths and try again.")

# %%