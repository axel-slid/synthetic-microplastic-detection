# Microplastic Synthetic Data Augmentation Pipeline

End-to-end pipeline for training generative inpainting models that synthesise
microplastic images, then evaluating whether those synthetic images improve
downstream segmentation performance.

---

## Repository layout

```
microplastic/
├── data/
│   ├── c1/                          real annotated images (training source)
│   │   ├── imgs/
│   │   ├── masks/
│   │   └── masks_dilated/           created by step 1
│   ├── c2/                          unlabelled background images
│   │   └── imgs/
│   ├── c3/                          held-out test cohort
│   │   ├── imgs/
│   │   └── masks/
│   ├── splits/                      created by step 4
│   │   ├── baseline/
│   │   ├── gan/ | sd/ | lama/ | mat/
│   └── ...
│
├── src/
│   ├── generators/
│   │   ├── gan.py                   U-Net GAN (PatchGAN discriminator)
│   │   ├── stable_diffusion.py      SD-2 inpainting fine-tuned on c1
│   │   ├── lama.py                  LaMa: Fourier Convolution inpainting (ICLR 2022)
│   │   └── mat.py                   MAT: Mask-Aware Transformer (CVPR 2022)
│   └── segmentation/
│       ├── model.py                 DeepLabV3-ResNet50 training
│       └── metrics.py               pixel accuracy + IoU comparison
│
├── scripts/                         numbered pipeline steps
│   ├── 01_preprocess.py
│   ├── 02_train_generator.py
│   ├── 03_generate_synthetic.py
│   ├── 04_prepare_data.py
│   ├── 05_train_segmentation.py
│   └── 06_evaluate.py
│
├── checkpoints/                     saved model weights (created at runtime)
├── outputs/                         comparison figures (created at runtime)
└── experiments/
    └── reader_study/
        └── generate.py
```

---

## Pipeline overview

```
data/c1 (real) ──► Step 1: dilate masks
                       │
                       ▼
              Step 2: train generator  ◄── choose: gan | sd | lama | mat
                       │
                       ▼
data/c2 (unlabelled) ─► Step 3: generate synthetic images
                       │
                       ▼
              Step 4: prepare train/val split  (real + synthetic)
                       │
                       ▼
              Step 5: train segmentation model
                       │
                       ▼
data/c3 (test) ──────► Step 6: evaluate & compare models
```

---

## Requirements

```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install opencv-python pillow matplotlib tqdm
pip install xformers          # optional, faster SD attention
pip install openpyxl          # reader study only
```

---

## Step 1 — Preprocess: dilate masks

Dilates c1 annotation masks by 4 pixels so the inpainting region covers the
full extent of each microplastic fibre.

```bash
python scripts/01_preprocess.py \
    --src  data/c1/masks \
    --dst  data/c1/masks_dilated \
    --kernel_size 3 \
    --iterations  4
```

**Output:** `data/c1/masks_dilated/`

---

## Step 2 — Train a generative model

All four models share the same interface.  Pick one (or run all four to
compare synthesis quality).

### GAN (U-Net + PatchGAN)

```bash
python scripts/02_train_generator.py \
    --model      gan \
    --image_dir  data/c1/imgs \
    --mask_dir   data/c1/masks_dilated \
    --output_dir checkpoints/gan \
    --epochs     500 \
    --batch_size 16 \
    --lr         2e-4 \
    --device     cuda
```

**Checkpoints saved to:** `checkpoints/gan/generator.pth`, `checkpoints/gan/discriminator.pth`
**Training previews:** `checkpoints/gan/figures/epoch_*.png`

---

### Stable Diffusion (fine-tuned SD-2 inpainting)

Downloads `stabilityai/stable-diffusion-2-inpainting` from HuggingFace on first run.

```bash
python scripts/02_train_generator.py \
    --model          sd \
    --image_dir      data/c1/imgs \
    --mask_dir       data/c1/masks_dilated \
    --output_dir     checkpoints/sd \
    --epochs         100 \
    --batch_size     1 \
    --lr             1e-5 \
    --grad_accum     4 \
    --mixed_precision fp16 \
    --device         cuda
```

**Final UNet saved to:** `checkpoints/sd/unet_final/`
**Per-epoch snapshots:** `checkpoints/sd/epoch-{N}/unet/`

---

### LaMa (Fourier Convolution inpainting — ICLR 2022)

Uses Fast Fourier Convolution blocks to capture global structure.
Recommended over GAN for large masks and complex textures.

```bash
python scripts/02_train_generator.py \
    --model      lama \
    --image_dir  data/c1/imgs \
    --mask_dir   data/c1/masks_dilated \
    --output_dir checkpoints/lama \
    --epochs     200 \
    --batch_size 8 \
    --lr         1e-4 \
    --ffc_blocks 9 \
    --lambda_rec 10.0 \
    --device     cuda
```

**Checkpoints saved to:** `checkpoints/lama/generator.pth`

---

### MAT (Mask-Aware Transformer — CVPR 2022)

Transformer bottleneck with masked self-attention.  Style injection via AdaIN
ensures inpainted regions match the surrounding tissue appearance.
Requires more GPU memory than GAN/LaMa; reduce `--batch_size` if needed.

```bash
python scripts/02_train_generator.py \
    --model      mat \
    --image_dir  data/c1/imgs \
    --mask_dir   data/c1/masks_dilated \
    --output_dir checkpoints/mat \
    --epochs     200 \
    --batch_size 4 \
    --lr         1e-4 \
    --style_dim  256 \
    --embed_dim  512 \
    --num_heads  8 \
    --depth      6 \
    --lambda_rec 10.0 \
    --device     cuda
```

**Checkpoints saved to:** `checkpoints/mat/generator.pth`

---

## Step 3 — Generate synthetic images

Takes unlabelled c2 images, randomly selects a c1 mask, and inpaints synthetic
microplastics.  Produces both the generated image and its corresponding mask.

### GAN

```bash
python scripts/03_generate_synthetic.py \
    --model      gan \
    --checkpoint checkpoints/gan/generator.pth \
    --image_dir  data/c2/imgs \
    --mask_dir   data/c1/masks_dilated \
    --output_dir data/c2/gen_gan \
    --num_images 10000 \
    --device     cuda
```

**Output:** `data/c2/gen_gan/` (images), `data/c2/gen_gan_masks/` (masks)

---

### Stable Diffusion

```bash
python scripts/03_generate_synthetic.py \
    --model           sd \
    --checkpoint      checkpoints/sd/unet_final \
    --image_dir       data/c2/imgs \
    --mask_dir        data/c1/masks_dilated \
    --output_dir      data/c2/gen_sd \
    --num_images      10000 \
    --inference_steps 50 \
    --device          cuda
```

**Output:** `data/c2/gen_sd/` (images), `data/c2/gen_sd_masks/` (masks)

---

### LaMa

```bash
python scripts/03_generate_synthetic.py \
    --model      lama \
    --checkpoint checkpoints/lama/generator.pth \
    --image_dir  data/c2/imgs \
    --mask_dir   data/c1/masks_dilated \
    --output_dir data/c2/gen_lama \
    --num_images 10000 \
    --device     cuda
```

**Output:** `data/c2/gen_lama/` (images), `data/c2/gen_lama_masks/` (masks)

---

### MAT

```bash
python scripts/03_generate_synthetic.py \
    --model      mat \
    --checkpoint checkpoints/mat/generator.pth \
    --image_dir  data/c2/imgs \
    --mask_dir   data/c1/masks_dilated \
    --output_dir data/c2/gen_mat \
    --num_images 10000 \
    --device     cuda
```

**Output:** `data/c2/gen_mat/` (images), `data/c2/gen_mat_masks/` (masks)

---

## Step 4 — Prepare train/val data splits

Creates the train/val directory structure expected by the segmentation trainer.
Run once for the baseline (no generated data), then once per generative model.

### Baseline split (real data only)

```bash
python scripts/04_prepare_data.py \
    --real_imgs  data/c1/imgs \
    --real_masks data/c1/masks \
    --output_dir data/splits/baseline \
    --real_only
```

### GAN-augmented split

```bash
python scripts/04_prepare_data.py \
    --real_imgs  data/c1/imgs \
    --real_masks data/c1/masks \
    --gen_imgs   data/c2/gen_gan \
    --gen_masks  data/c2/gen_gan_masks \
    --output_dir data/splits/gan
```

### SD-augmented split

```bash
python scripts/04_prepare_data.py \
    --real_imgs  data/c1/imgs \
    --real_masks data/c1/masks \
    --gen_imgs   data/c2/gen_sd \
    --gen_masks  data/c2/gen_sd_masks \
    --output_dir data/splits/sd
```

### LaMa-augmented split

```bash
python scripts/04_prepare_data.py \
    --real_imgs  data/c1/imgs \
    --real_masks data/c1/masks \
    --gen_imgs   data/c2/gen_lama \
    --gen_masks  data/c2/gen_lama_masks \
    --output_dir data/splits/lama
```

### MAT-augmented split

```bash
python scripts/04_prepare_data.py \
    --real_imgs  data/c1/imgs \
    --real_masks data/c1/masks \
    --gen_imgs   data/c2/gen_mat \
    --gen_masks  data/c2/gen_mat_masks \
    --output_dir data/splits/mat
```

---

## Step 5 — Train segmentation models

Trains DeepLabV3-ResNet50 for binary microplastic segmentation.
Run once per data split.

### Baseline

```bash
python scripts/05_train_segmentation.py \
    --data_root        data/splits/baseline \
    --output_dir       checkpoints/seg_baseline \
    --epochs           100 \
    --samples_per_epoch 10000 \
    --batch_size       4 \
    --lr               1e-4 \
    --device           cuda
```

### GAN-augmented

```bash
python scripts/05_train_segmentation.py \
    --data_root        data/splits/gan \
    --output_dir       checkpoints/seg_gan \
    --epochs           100 \
    --samples_per_epoch 10000 \
    --batch_size       4 \
    --lr               1e-4 \
    --device           cuda
```

### SD-augmented

```bash
python scripts/05_train_segmentation.py \
    --data_root        data/splits/sd \
    --output_dir       checkpoints/seg_sd \
    --epochs           100 \
    --samples_per_epoch 10000 \
    --batch_size       4 \
    --lr               1e-4 \
    --device           cuda
```

### LaMa-augmented

```bash
python scripts/05_train_segmentation.py \
    --data_root        data/splits/lama \
    --output_dir       checkpoints/seg_lama \
    --epochs           100 \
    --samples_per_epoch 10000 \
    --batch_size       4 \
    --lr               1e-4 \
    --device           cuda
```

### MAT-augmented

```bash
python scripts/05_train_segmentation.py \
    --data_root        data/splits/mat \
    --output_dir       checkpoints/seg_mat \
    --epochs           100 \
    --samples_per_epoch 10000 \
    --batch_size       4 \
    --lr               1e-4 \
    --device           cuda
```

**Best checkpoint saved to:** `checkpoints/seg_<model>/best_model.pth`
**Per-epoch previews:** `checkpoints/seg_<model>/viz/epoch_*.jpg`

---

## Step 6 — Evaluate and compare models

Computes pixel accuracy and IoU on the held-out test set (c3) and saves
side-by-side comparison figures.

### Baseline vs GAN

```bash
python scripts/06_evaluate.py \
    --model_a    checkpoints/seg_baseline/best_model.pth \
    --model_b    checkpoints/seg_gan/best_model.pth \
    --label_a    "Baseline" \
    --label_b    "GAN" \
    --data_root  data/c3 \
    --output_dir outputs/comparison_gan
```

### Baseline vs Stable Diffusion

```bash
python scripts/06_evaluate.py \
    --model_a    checkpoints/seg_baseline/best_model.pth \
    --model_b    checkpoints/seg_sd/best_model.pth \
    --label_a    "Baseline" \
    --label_b    "Stable Diffusion" \
    --data_root  data/c3 \
    --output_dir outputs/comparison_sd
```

### Baseline vs LaMa

```bash
python scripts/06_evaluate.py \
    --model_a    checkpoints/seg_baseline/best_model.pth \
    --model_b    checkpoints/seg_lama/best_model.pth \
    --label_a    "Baseline" \
    --label_b    "LaMa" \
    --data_root  data/c3 \
    --output_dir outputs/comparison_lama
```

### Baseline vs MAT

```bash
python scripts/06_evaluate.py \
    --model_a    checkpoints/seg_baseline/best_model.pth \
    --model_b    checkpoints/seg_mat/best_model.pth \
    --label_a    "Baseline" \
    --label_b    "MAT" \
    --data_root  data/c3 \
    --output_dir outputs/comparison_mat
```

**Printed metrics:** Pixel accuracy + Mean IoU for each model
**Saved figures:** `outputs/comparison_<model>/*.png`

---

## Model architecture notes

### GAN
- Generator: simple 4-level U-Net encoder/decoder (BCE adversarial loss)
- Discriminator: 70x70 PatchGAN
- Input: 4-channel (RGB + mask); output composited back onto original

### Stable Diffusion
- Base: `stabilityai/stable-diffusion-2-inpainting`
- Fine-tunes UNet weights only; VAE and CLIP text encoder frozen
- Text prompt fixed to microplastic description during training
- Inference: null-text DDPM sampling

### LaMa (ICLR 2022)
- Key innovation: Fast Fourier Convolution (FFC) — splits feature maps into
  local (spatial conv) and global (spectral / FFT) pathways
- Generator: CNN encoder → 9 FFCResBlocks → CNN decoder
- Discriminator: two-scale PatchGAN with spectral normalisation
- Loss: hinge adversarial + L1 reconstruction in masked region

### MAT (CVPR 2022)
- Key innovation: masked self-attention — transformer tokens attend only to
  unmasked (visible) positions, preventing information leakage
- Style encoder extracts a global style vector from visible pixels
- AdaIN injects style into each decoder block for appearance consistency
- Generator: CNN encoder → 6 transformer blocks → CNN decoder
- Loss: hinge adversarial + L1 reconstruction in masked region

---

## Running the full pipeline (example — LaMa)

```bash
# 1. Dilate masks
python scripts/01_preprocess.py \
    --src data/c1/masks --dst data/c1/masks_dilated

# 2. Train LaMa generator
python scripts/02_train_generator.py \
    --model lama --image_dir data/c1/imgs \
    --mask_dir data/c1/masks_dilated \
    --output_dir checkpoints/lama --epochs 200

# 3. Generate 10k synthetic images
python scripts/03_generate_synthetic.py \
    --model lama --checkpoint checkpoints/lama/generator.pth \
    --image_dir data/c2/imgs --mask_dir data/c1/masks_dilated \
    --output_dir data/c2/gen_lama --num_images 10000

# 4a. Baseline split (real only)
python scripts/04_prepare_data.py \
    --real_imgs data/c1/imgs --real_masks data/c1/masks \
    --output_dir data/splits/baseline --real_only

# 4b. LaMa-augmented split
python scripts/04_prepare_data.py \
    --real_imgs data/c1/imgs --real_masks data/c1/masks \
    --gen_imgs data/c2/gen_lama --gen_masks data/c2/gen_lama_masks \
    --output_dir data/splits/lama

# 5a. Train baseline segmentation
python scripts/05_train_segmentation.py \
    --data_root data/splits/baseline \
    --output_dir checkpoints/seg_baseline

# 5b. Train LaMa-augmented segmentation
python scripts/05_train_segmentation.py \
    --data_root data/splits/lama \
    --output_dir checkpoints/seg_lama

# 6. Compare on test set
python scripts/06_evaluate.py \
    --model_a checkpoints/seg_baseline/best_model.pth \
    --model_b checkpoints/seg_lama/best_model.pth \
    --label_a "Baseline" --label_b "LaMa" \
    --data_root data/c3 --output_dir outputs/comparison_lama
```
