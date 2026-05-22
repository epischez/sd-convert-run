"""Convert sd-1.5-inpainting → CoreML (9-channel UNet, NO LoRA)."""
import os, sys, shutil, subprocess, platform

workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(workspace_dir)
print(f"📂 Workspace: {workspace_dir}", flush=True)

inpaint_model_dir = os.path.join(workspace_dir, "scratch/runwayml_stable-diffusion-inpainting")
restaged_dir      = os.path.join(workspace_dir, "scratch/temp_restaged_inpaint")
coreml_out_dir    = os.path.join(workspace_dir, "scratch/temp_coreml_inpaint")
final_models_dir  = os.path.join(workspace_dir, "sd_models_inpaint")

if not os.path.isdir(inpaint_model_dir):
    sys.exit(f"❌ Missing source: {inpaint_model_dir}")

# Sanity-check 9-channel UNet
import json
unet_cfg = os.path.join(inpaint_model_dir, "unet", "config.json")
if os.path.exists(unet_cfg):
    with open(unet_cfg) as f:
        cfg = json.load(f)
    in_ch = cfg.get("in_channels", "?")
    print(f"🔎 UNet in_channels = {in_ch} (expect 9)", flush=True)
    if in_ch != 9:
        sys.exit(f"❌ Wrong source — expected 9-channel UNet, got {in_ch}")

print("\n=== Re-stage fp16 → fp32 .bin layout for torch2coreml ===", flush=True)
import torch
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    inpaint_model_dir, torch_dtype=torch.float32, variant="fp16",
    safety_checker=None, feature_extractor=None, requires_safety_checker=False,
)
pipe.to(dtype=torch.float32)
if os.path.exists(restaged_dir):
    shutil.rmtree(restaged_dir)
print(f"💾 save_pretrained → {restaged_dir}", flush=True)
pipe.save_pretrained(restaged_dir)
del pipe
import gc; gc.collect()

print("\n=== torch2coreml ===", flush=True)
if os.path.exists(coreml_out_dir):
    shutil.rmtree(coreml_out_dir)

is_mac = platform.system() == "Darwin"
cmd = [sys.executable, "-m", "python_coreml_stable_diffusion.torch2coreml",
       "--convert-unet", "--chunk-unet",
       "--convert-text-encoder",
       "--convert-vae-decoder", "--convert-vae-encoder",
       "--model-version", restaged_dir,
       "--attention-implementation", "ORIGINAL",
       "-o", coreml_out_dir]
if is_mac:
    cmd.append("--bundle-resources-for-swift-cli")
env = dict(os.environ)
ml_sd = os.path.join(workspace_dir, "ml-stable-diffusion")
env["PYTHONPATH"] = ml_sd + os.pathsep + env.get("PYTHONPATH", "")
if is_mac:
    env["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"
print(f"🚀 {' '.join(cmd)}", flush=True)
subprocess.run(cmd, env=env, check=True)

print("\n=== Stage final dir ===", flush=True)
if os.path.exists(final_models_dir):
    shutil.rmtree(final_models_dir)
os.makedirs(final_models_dir, exist_ok=True)

import glob, requests
if is_mac:
    src = os.path.join(coreml_out_dir, "Resources")
    for name in ("TextEncoder.mlmodelc", "UnetChunk1.mlmodelc", "UnetChunk2.mlmodelc",
                 "VAEEncoder.mlmodelc", "VAEDecoder.mlmodelc", "vocab.json", "merges.txt"):
        s = os.path.join(src, name); d = os.path.join(final_models_dir, name)
        if os.path.exists(s):
            if os.path.isdir(s): shutil.copytree(s, d)
            else: shutil.copy2(s, d)
            print(f"✅ {name}", flush=True)
else:
    rename_map = [
        ("text_encoder.mlpackage",  "TextEncoder.mlpackage"),
        ("unet_chunk1.mlpackage",   "UnetChunk1.mlpackage"),
        ("unet_chunk2.mlpackage",   "UnetChunk2.mlpackage"),
        ("vae_encoder.mlpackage",   "VAEEncoder.mlpackage"),
        ("vae_decoder.mlpackage",   "VAEDecoder.mlpackage"),
    ]
    for src_glob, dst_name in rename_map:
        matches = glob.glob(os.path.join(coreml_out_dir, f"*_{src_glob}"))
        if not matches:
            print(f"⚠️  missing {src_glob}", flush=True); continue
        shutil.copytree(matches[0], os.path.join(final_models_dir, dst_name))
        print(f"✅ {dst_name}", flush=True)
    print("📥 Downloading tokenizer vocab/merges…", flush=True)
    for fn, url in [
        ("vocab.json", "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json"),
        ("merges.txt", "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt"),
    ]:
        with open(os.path.join(final_models_dir, fn), "wb") as f:
            f.write(requests.get(url, timeout=60).content)
        print(f"✅ {fn}", flush=True)

zip_path = os.path.join(workspace_dir, "sd_models_inpaint.zip")
if os.path.exists(zip_path): os.remove(zip_path)
shutil.make_archive(zip_path.replace(".zip",""), "zip", final_models_dir)
print(f"\n🎉 Inpaint bundle ready: {zip_path}", flush=True)
