#!/usr/bin/env python3
"""
Smoke test for SD training pipeline.
Loads all model components from local cache, runs one forward + backward pass,
and one optimiser step. Exits 0 on success, non-zero on failure.

Usage:
    CUDA_VISIBLE_DEVICES=1 python tests/test_sd_smoke.py
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

# 1. CUDA check
assert torch.cuda.is_available(), "CUDA not available"
gpu  = torch.cuda.get_device_properties(0)
free = gpu.total_memory - torch.cuda.memory_allocated(0)
print(f"[GPU]  {gpu.name} | total {gpu.total_memory/1e9:.1f} GB | free ~{free/1e9:.1f} GB")
assert gpu.total_memory > 20e9, f"GPU too small ({gpu.total_memory/1e9:.1f} GB); need >20 GB for SD"

# 2. HF cache check
from huggingface_hub import file_download
cache = file_download.HUGGINGFACE_HUB_CACHE
print(f"[HF]   cache = {cache}")
model_dir = os.path.join(cache, "models--stabilityai--stable-diffusion-2-inpainting")
assert os.path.isdir(model_dir), f"SD model cache not found: {model_dir}"
snaps = os.listdir(os.path.join(model_dir, "snapshots"))
assert snaps, "No snapshots in SD model cache"
print(f"[HF]   snapshot: {snaps[0]}")

# 3. Load components
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
kw = dict(local_files_only=True)
BASE = "stabilityai/stable-diffusion-2-inpainting"
print("[Load] scheduler ..."); scheduler   = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler", **kw)
print("[Load] vae ...");       vae          = AutoencoderKL.from_pretrained(BASE, subfolder="vae", **kw)
print("[Load] unet ...");      unet         = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", **kw)
print("[Load] tokenizer ...");  tokenizer   = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer", **kw)
print("[Load] text_encoder ..."); text_enc  = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder", **kw)

# 4. To GPU fp16
device = torch.device("cuda")
vae.to(device, dtype=torch.float16).requires_grad_(False).eval()
text_enc.to(device, dtype=torch.float16).requires_grad_(False).eval()
unet.to(device, dtype=torch.float16).train()
alloc = torch.cuda.memory_allocated() / 1e9
print(f"[VRAM] after load: {alloc:.2f} GB / {gpu.total_memory/1e9:.1f} GB")
assert alloc < gpu.total_memory * 0.85 / 1e9, f"OOM risk after model load ({alloc:.2f} GB used)"

# 5. Forward + backward
B = 1
img  = torch.randn(B, 3, 512, 512, device=device, dtype=torch.float16)
mask = (torch.rand(B, 1, 64, 64, device=device) > 0.5).to(torch.float16)
with torch.no_grad():
    lat  = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor
    mimg = img * (1 - F.interpolate(mask, size=(512, 512)))
    mlat = vae.encode(mimg).latent_dist.sample() * vae.config.scaling_factor
noise = torch.randn_like(lat)
t     = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
noisy = scheduler.add_noise(lat, noise, t)
inp   = torch.cat([noisy, mask, mlat], dim=1)
toks  = tokenizer([""] * B, return_tensors="pt", padding="max_length",
                  max_length=tokenizer.model_max_length, truncation=True).input_ids.to(device)
with torch.no_grad():
    enc = text_enc(toks).last_hidden_state.to(torch.float16)
with torch.autocast("cuda"):
    pred = unet(inp, t, encoder_hidden_states=enc).sample
    loss = F.mse_loss(pred.float(), noise.float())
print(f"[Pass] loss = {loss.item():.4f}")
assert math.isfinite(loss.item()), "loss is NaN/Inf"
loss.backward()
print(f"[VRAM] after backward: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# 6. Optimiser step
opt = torch.optim.AdamW(unet.parameters(), lr=1e-5)
opt.step(); opt.zero_grad()
print("[PASS] Smoke test passed — SD pipeline is ready.")
