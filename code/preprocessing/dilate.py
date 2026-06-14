# %%

import os
import cv2
import numpy as np

def dilate_masks(src_folder, dst_folder):
    """
    Dilates all binary mask images in the source folder by 2 pixels and saves them to the destination folder.

    Parameters:
    src_folder (str): Path to the folder containing the input mask images.
    dst_folder (str): Path to the folder where the dilated mask images will be saved.
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Define a 3x3 kernel for dilation
    kernel = np.ones((3, 3), np.uint8)

    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)

        # Skip if it's not a file
        if not os.path.isfile(src_path):
            continue

        # Read the image in grayscale mode
        mask = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Skipping {filename}: not a valid image.")
            continue

        # Apply dilation with 2 iterations (approx. 2-pixel thickness)
        dilated = cv2.dilate(mask, kernel, iterations=4)

        dst_path = os.path.join(dst_folder, filename)
        cv2.imwrite(dst_path, dilated)

# Example usage: update these paths as needed
source_folder = "/mnt/shared/dils/projects/microplastic/data/c1/masks"
destination_folder = "/mnt/shared/dils/projects/microplastic/data/c1/masks_dilated"
dilate_masks(source_folder, destination_folder)
# %%

