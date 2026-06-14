# %%

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import matplotlib.pyplot as plt
import os

def run_inference_and_plot(
    generator_ckpt_path: str,
    image_path: str,
    mask_path: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    resize_to: tuple = (512, 512),
):
    """
    Loads a trained generator, applies it to an image+mask, and shows a 3-panel matplotlib plot:
        - Original image
        - Mask
        - Inpainted output

    Args:
        generator_ckpt_path (str): Path to trained generator weights (.pth)
        image_path (str): Path to input image (RGB)
        mask_path (str): Path to binary mask image (L)
        device (torch.device): CUDA or CPU
        resize_to (tuple): Resize (H, W) for input
    """
    # --- Generator model definition (same as in training) ---
    class Generator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            def conv(in_c, out_c):
                return torch.nn.Sequential(
                    torch.nn.Conv2d(in_c, out_c, 4, 2, 1),
                    torch.nn.ReLU(inplace=True),
                )
            def deconv(in_c, out_c):
                return torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                    torch.nn.ReLU(inplace=True),
                )
            self.encoder = torch.nn.Sequential(
                conv(4, 64), conv(64, 128), conv(128, 256), conv(256, 512)
            )
            self.decoder = torch.nn.Sequential(
                deconv(512, 256), deconv(256, 128), deconv(128, 64),
                torch.nn.ConvTranspose2d(64, 3, 4, 2, 1),
                torch.nn.Tanh()
            )

        def forward(self, x, original, mask):
            latent = self.encoder(x)
            recon = self.decoder(latent)
            return recon * mask + original * (1.0 - mask)

    # --- Preprocessing transforms ---
    tf_image = transforms.Compose([
        transforms.Resize(resize_to, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tf_mask = transforms.Compose([
        transforms.Resize(resize_to, interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t > 0.5).float()),
    ])

    # --- Load image and mask ---
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    image_tensor = tf_image(image).unsqueeze(0).to(device)  # (1,3,H,W)
    mask_tensor = tf_mask(mask).unsqueeze(0).to(device)     # (1,1,H,W)

    # --- Load generator and weights ---
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_ckpt_path, map_location=device))
    generator.eval()

    with torch.no_grad():
        masked_input = image_tensor * (1.0 - mask_tensor)
        gen_input = torch.cat([masked_input, mask_tensor], dim=1)
        output = generator(gen_input, image_tensor, mask_tensor)

    # --- Plotting ---
    img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    mask_np = mask_tensor.squeeze().cpu().numpy()
    out_np  = output.squeeze().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(out_np)
    axes[2].set_title("Inpainted")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()



run_inference_and_plot(
    generator_ckpt_path="/mnt/shared/dils/projects/microplastic/code/gan/model/generator.pth",
    image_path="/mnt/shared/dils/projects/microplastic/data/c1/imgs/sample.jpg",
    mask_path="/mnt/shared/dils/projects/microplastic/data/c1/masks_dilated/sample.png"
)
