"""CI download driver (Ubuntu) — no aria2/proxy needed, plain HF snapshot_download with hf_transfer."""
import os
from huggingface_hub import snapshot_download

# Same allow-lists as local; HF on GH Actions is fast enough without aria2.
TARGETS = [
    ("runwayml/stable-diffusion-v1-5",
     "scratch/runwayml_stable-diffusion-v1-5",
     ["model_index.json", "scheduler/*.json", "tokenizer/*",
      "text_encoder/config.json", "text_encoder/*.fp16.bin",
      "vae/config.json", "vae/*.fp16.bin",
      "unet/config.json", "unet/*.fp16.bin"]),
    ("runwayml/stable-diffusion-inpainting",
     "scratch/runwayml_stable-diffusion-inpainting",
     ["model_index.json", "scheduler/*.json", "tokenizer/*",
      "text_encoder/config.json", "text_encoder/*.fp16.bin",
      "vae/config.json", "vae/*.fp16.bin",
      "unet/config.json", "unet/*.fp16.bin"]),
    ("latent-consistency/lcm-lora-sdv1-5",
     "scratch/latent-consistency_lcm-lora-sdv1-5",
     ["*.safetensors", "*.json", "*.md"]),
]

for repo, dest, allow in TARGETS:
    print(f"=== {repo} → {dest} ===", flush=True)
    snapshot_download(repo, local_dir=dest, allow_patterns=allow)
    print(f"✅ done", flush=True)
print("🎉 all sources fetched")
