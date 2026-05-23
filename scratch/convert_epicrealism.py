"""Convert epiCRealism pureEvolution V3 → CoreML.
Runs on GitHub Action Linux (no Xcode → produces .mlpackage).
"""
import os
import sys
import shutil
import subprocess
import argparse
import platform

workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(workspace_dir)
print(f"📂 Workspace: {workspace_dir}", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("--attention-implementation", default="ORIGINAL", choices=["ORIGINAL", "SPLIT_EINSUM"])
parser.add_argument("--resolution", default="768x768", choices=["768x768", "512x768", "768x512", "512x512"])
args = parser.parse_args()

# Map resolution to latent dimensions (resolution / 8)
res_map = {
    "768x768": (96, 96),
    "512x768": (64, 96),
    "768x512": (96, 64),
    "512x512": (64, 64),
}
latent_h, latent_w = res_map[args.resolution]

model_id = "scratch/jzli_epiCRealism-v3"
coreml_out_dir = os.path.join(workspace_dir, "scratch/temp_coreml_epicrealism")
final_models_dir = os.path.join(workspace_dir, f"epicrealism_pureEvolutionV3_{args.attention_implementation.lower()}_{args.resolution}")

if os.path.exists(coreml_out_dir):
    shutil.rmtree(coreml_out_dir)

is_mac = platform.system() == "Darwin"
cmd = [sys.executable, "-m", "python_coreml_stable_diffusion.torch2coreml",
       "--convert-unet", "--chunk-unet",
       "--convert-text-encoder",
       "--convert-vae-decoder",
       "--convert-vae-encoder",
       "--model-version", model_id,
       "--attention-implementation", args.attention_implementation,
       "--latent-h", str(latent_h),
       "--latent-w", str(latent_w),
       "-o", coreml_out_dir]

if is_mac:
    cmd.append("--bundle-resources-for-swift-cli")

env = dict(os.environ)
ml_sd = os.path.join(workspace_dir, "ml-stable-diffusion")
env["PYTHONPATH"] = ml_sd + os.pathsep + env.get("PYTHONPATH", "")
if is_mac:
    env["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"

print(f"🚀 Running: {' '.join(cmd)}", flush=True)
subprocess.run(cmd, env=env, check=True)

print("\n=== Step 2: Stage final dir ===", flush=True)
if os.path.exists(final_models_dir):
    shutil.rmtree(final_models_dir)
os.makedirs(final_models_dir, exist_ok=True)

rename_map = [
    ("text_encoder.mlpackage",  "TextEncoder.mlpackage"),
    ("unet_chunk1.mlpackage",   "UnetChunk1.mlpackage"),
    ("unet_chunk2.mlpackage",   "UnetChunk2.mlpackage"),
    ("vae_decoder.mlpackage",   "VAEDecoder.mlpackage"),
    ("vae_encoder.mlpackage",   "VAEEncoder.mlpackage"),
]

import glob
for src_glob, dst_name in rename_map:
    matches = glob.glob(os.path.join(coreml_out_dir, f"*_{src_glob}"))
    if not matches:
        print(f"⚠️  missing {src_glob}", flush=True)
        continue
    s = matches[0]
    d = os.path.join(final_models_dir, dst_name)
    shutil.copytree(s, d)
    print(f"✅ {dst_name} ← {os.path.basename(s)}", flush=True)

# Tokenizer files
print("📥 Fetching tokenizer vocab/merges…", flush=True)
import requests
for fn, url in [
    ("vocab.json",  "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json"),
    ("merges.txt",  "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt"),
]:
    with open(os.path.join(final_models_dir, fn), "wb") as f:
        f.write(requests.get(url, timeout=60).content)
    print(f"✅ {fn}", flush=True)

# Zip for upload
zip_name = f"epicrealism_pureEvolutionV3_{args.attention_implementation.lower()}_{args.resolution}"
zip_path = os.path.join(workspace_dir, f"{zip_name}.zip")
if os.path.exists(zip_path):
    os.remove(zip_path)
shutil.make_archive(zip_path.replace(".zip", ""), "zip", final_models_dir)
print(f"\n🎉 epiCRealism bundle ready: {zip_path}", flush=True)
