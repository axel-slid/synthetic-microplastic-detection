# %% --------------------------------------------------------------------------------

from __future__ import annotations

import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Helper transforms (synchronized image/mask)
# ------------------------------------------------------------------------------

resize_to = (512, 512)

class JointTransform:
    def __init__(self, image_size: Tuple[int, int]):
        self.resize = transforms.Resize(image_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.resize(image)
        mask = self.resize(mask)

        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        if random.random() < 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        if random.random() < 0.5:
            angle = 0
            scale = random.uniform(0.9, 1.1)
            translate = [0, 0]
            image = transforms.functional.affine(image, angle=angle, translate=translate, scale=scale, shear=0, fill=0, interpolation=InterpolationMode.BILINEAR)
            mask = transforms.functional.affine(mask, angle=angle, translate=translate, scale=scale, shear=0, fill=0, interpolation=InterpolationMode.NEAREST)

        image = self.normalize(self.to_tensor(image))
        mask = (self.to_tensor(mask) > 0.5).float()
        return image, mask

joint_transform = JointTransform(resize_to)

# ------------------------------------------------------------------------------
# 1. Models
# ------------------------------------------------------------------------------

class Generator(nn.Module):
    """U-Net-style encoder/decoder for in-painting masked regions."""

    def __init__(self) -> None:
        super().__init__()

        def conv(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.ReLU(inplace=True),
            )

        def deconv(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.ReLU(inplace=True),
            )

        self.encoder = nn.Sequential(
            conv(4, 64),
            conv(64, 128),
            conv(128, 256),
            conv(256, 512),
        )

        self.decoder = nn.Sequential(
            deconv(512, 256),
            deconv(256, 128),
            deconv(128, 64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),  # output ∈ [-1,1]
        )

    def forward(
        self,
        x: torch.Tensor,
        original: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon * mask + original * (1.0 - mask)


class Discriminator(nn.Module):
    """70×70 PatchGAN discriminator."""

    def __init__(self) -> None:
        super().__init__()

        def block(in_c: int, out_c: int, *, stride: int = 2) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.model = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# ------------------------------------------------------------------------------
# 2. Dataset
# ------------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        assert len(self.images) == len(self.masks), "Image/Mask count mismatch."

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, mask = joint_transform(image, mask)

        return image, mask, os.path.basename(img_path), os.path.basename(mask_path)

# ------------------------------------------------------------------------------
# 3. Visual utilities
# ------------------------------------------------------------------------------

def save_sanity_check_figure(
    dataloader: DataLoader,
    output_path: str,
    num_samples: int = 3,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images, masks, img_names, mask_names = next(iter(dataloader))

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6), squeeze=False)

    for i in range(num_samples):
        img_disp = images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
        axes[0, i].imshow(img_disp)
        axes[0, i].set_title(img_names[i], fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(masks[i].squeeze().numpy(), cmap="gray")
        axes[1, i].set_title(mask_names[i], fontsize=8)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

def visualize_predictions(
    generator: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    output_dir: str,
    samples: int = 6,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()

    with torch.no_grad():
        images, masks, img_names, mask_names = next(iter(dataloader))
        images, masks = images.to(device), masks.to(device)

        masked_images = images * (1.0 - masks)
        gen_in = torch.cat([masked_images, masks], dim=1)
        generated = generator(gen_in, images, masks).cpu()

    rows, cols = 3, samples
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)

    for i in range(samples):
        axes[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        axes[0, i].set_title(f"GT\n{img_names[i]}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(masks[i].squeeze().cpu().numpy(), cmap="gray")
        axes[1, i].set_title(f"Mask\n{mask_names[i]}", fontsize=8)
        axes[1, i].axis("off")

        axes[2, i].imshow(generated[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[2, i].set_title("Generated", fontsize=8)
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"epoch_{epoch:03d}.png"), dpi=200)
    plt.close(fig)
    generator.train()

# ------------------------------------------------------------------------------
# 4. Training loop
# ------------------------------------------------------------------------------

def train(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    optimizer_G: optim.Optimizer,
    optimizer_D: optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    device: torch.device,
) -> None:
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for images, masks, *_ in loop:
            images, masks = images.to(device), masks.to(device)

            masked = images * (1.0 - masks)
            gen_in = torch.cat([masked, masks], dim=1)

            optimizer_G.zero_grad()
            generated = generator(gen_in, images, masks)
            pred_fake = discriminator(generated)
            g_loss = criterion(pred_fake, torch.ones_like(pred_fake))
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            pred_real = discriminator(images)
            real_loss = criterion(pred_real, torch.ones_like(pred_real))

            pred_fake_detached = discriminator(generated.detach())
            fake_loss = criterion(pred_fake_detached, torch.zeros_like(pred_fake_detached))

            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()

            loop.set_postfix(G=f"{g_loss.item():.4f}", D=f"{d_loss.item():.4f}")

        if (epoch + 1) % 10 == 0:
            visualize_predictions(
                generator,
                dataloader,
                device,
                epoch + 1,
                output_dir="/mnt/shared/dils/projects/microplastic/code/gan/figures",
            )

# ------------------------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(42)
    random.seed(42)

    batch_size: int = 16
    lr: float = 2e-4
    num_epochs: int = 500

    image_dir = "/mnt/shared/dils/projects/microplastic/data/c1/imgs"
    mask_dir = "/mnt/shared/dils/projects/microplastic/data/c1/masks_dilated"
    figures_dir = "/mnt/shared/dils/projects/microplastic/code/gan/figures"

    generator_ckpt = "/mnt/shared/dils/projects/microplastic/code/gan/model/generator.pth"
    discriminator_ckpt = "/mnt/shared/dils/projects/microplastic/code/gan/model/discriminator.pth"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    dataset = CustomDataset(image_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    save_sanity_check_figure(
        dataloader,
        output_path=os.path.join(figures_dir, "sanity_check_inputs.png"),
        num_samples=3,
    )
    print("[Info] Sanity-check figure saved.")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    train(generator, discriminator, dataloader, optimizer_G, optimizer_D, criterion, num_epochs, device)

    os.makedirs(os.path.dirname(generator_ckpt), exist_ok=True)
    torch.save(generator.state_dict(), generator_ckpt)
    torch.save(discriminator.state_dict(), discriminator_ckpt)
    print("[Info] Training complete – weights saved.")

if __name__ == "__main__":
    main()

# %% --------------------------------------------------------------------------------
