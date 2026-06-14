# %%

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm  # For progress bars

# Set random seed for reproducibility
random.seed(42)

# Input directories
real_imgs_dir = Path("/mnt/shared/dils/projects/microplastic/data/c1/imgs")
real_masks_dir = Path("/mnt/shared/dils/projects/microplastic/data/c1/masks")
gen_imgs_dir = Path("/mnt/shared/dils/projects/microplastic/data/c2/gen_imgs_1")
gen_masks_dir = Path("/mnt/shared/dils/projects/microplastic/data/c2/gen_masks_1")

# Output base directory
base_output = Path("/mnt/shared/dils/projects/microplastic/code/stable_diffusion_segmentation/split_data")
train_img_out = base_output / "train/imgs"
train_mask_out = base_output / "train/masks"
val_img_out = base_output / "val/imgs"
val_mask_out = base_output / "val/masks"

# Create output directories
for path in [train_img_out, train_mask_out, val_img_out, val_mask_out]:
    path.mkdir(parents=True, exist_ok=True)

# Function to collect image-mask pairs
def collect_pairs(img_dir, mask_dir):
    pairs = []
    for img_file in sorted(img_dir.iterdir()):
        if img_file.is_file():
            mask_file = mask_dir / img_file.name
            if mask_file.exists():
                pairs.append((img_file, mask_file))
            else:
                print(f"Warning: No matching mask for {img_file.name}")
    return pairs

print("Collecting real image-mask pairs...")
real_pairs = collect_pairs(real_imgs_dir, real_masks_dir)

print("Collecting generated image-mask pairs...")
gen_pairs = collect_pairs(gen_imgs_dir, gen_masks_dir)

# Combine and shuffle
all_pairs = real_pairs + gen_pairs
random.shuffle(all_pairs)

# 80/20 split
split_idx = int(0.8 * len(all_pairs))
train_pairs = all_pairs[:split_idx]
val_pairs = all_pairs[split_idx:]

# Copy image-mask pairs with tqdm
def copy_pairs(pairs, img_out, mask_out, desc=""):
    for img_path, mask_path in tqdm(pairs, desc=desc):
        shutil.copy(img_path, img_out / img_path.name)
        shutil.copy(mask_path, mask_out / mask_path.name)

print(f"Copying {len(train_pairs)} training pairs...")
copy_pairs(train_pairs, train_img_out, train_mask_out, desc="Copying training data")

print(f"Copying {len(val_pairs)} validation pairs...")
copy_pairs(val_pairs, val_img_out, val_mask_out, desc="Copying validation data")

print("\n✅ Done.")
print(f"Total samples: {len(all_pairs)}")
print(f"Training samples: {len(train_pairs)}")
print(f"Validation samples: {len(val_pairs)}")
print(f"Split data saved at: {base_output}")

# %%
