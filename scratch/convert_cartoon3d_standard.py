import os
import sys
import shutil
import subprocess

# Ensure we're in the right working directory
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(workspace_dir)
print(f"📂 Workspace directory set to: {workspace_dir}")

# Create scratch/temp dirs if not exists
temp_fused_dir = os.path.join(workspace_dir, "scratch/temp_fused_cartoon3d_standard")
temp_coreml_dir = os.path.join(workspace_dir, "scratch/temp_coreml_cartoon3d_standard")
os.makedirs("scratch", exist_ok=True)

print("\n=== Step 1: Loading Cartoon3D Standard Model ===")
try:
    from diffusers import StableDiffusionPipeline
    import torch
except ImportError:
    print("❌ Error: Missing required Python packages. Run: pip install diffusers transformers peft accelerate torch")
    sys.exit(1)

model_id = "Yntec/DisneyPixar"

print(f"📥 Loading base model from Hugging Face: {model_id}")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32
)

# Select computation device
device = "cpu"

print(f"💾 Saving model locally to: {temp_fused_dir}")
pipe.save_pretrained(temp_fused_dir)

# Clean memory
del pipe
import gc
gc.collect()

print("\n=== Step 2: Converting model to CoreML ===")
if os.path.exists(temp_coreml_dir):
    shutil.rmtree(temp_coreml_dir)

# Construct conversion command for T2I pipeline
is_mac = (sys.platform == "darwin")
cmd = [
    sys.executable, "-m", "python_coreml_stable_diffusion.torch2coreml",
    "--convert-text-encoder",
    "--convert-unet",
    "--chunk-unet",
    "--convert-vae-decoder",
    "--convert-vae-encoder", # VAE Encoder converted to support I2I
    "--model-version", temp_fused_dir,
    "--attention-implementation", "ORIGINAL",
    "--compute-unit", "ALL",
    "--latent-h", "64",
    "--latent-w", "64",
    "-o", temp_coreml_dir,
]

if is_mac:
    cmd.append("--bundle-resources-for-swift-cli")

print(f"🚀 Executing conversion command: {' '.join(cmd)}")
env = dict(os.environ)
ml_sd_dir = os.path.join(workspace_dir, "ml-stable-diffusion")
if "PYTHONPATH" in env:
    env["PYTHONPATH"] = ml_sd_dir + os.pathsep + env["PYTHONPATH"]
else:
    env["PYTHONPATH"] = ml_sd_dir

if is_mac:
    env["DEVELOPER_DIR"] = "/Applications/Xcode.app/Contents/Developer"

subprocess.run(cmd, env=env, check=True)

print("\n=== Step 3: Packaging CoreML models ===")
output_zip = os.path.join(workspace_dir, "sd_models_cartoon3d_standard.zip")
if os.path.exists(output_zip):
    os.remove(output_zip)

if is_mac:
    compiled_resources_dir = os.path.join(temp_coreml_dir, "Resources")
    print(f"📦 Zipping compiled .mlmodelc assets from {compiled_resources_dir}...")
    shutil.make_archive(output_zip.replace(".zip", ""), 'zip', compiled_resources_dir)
else:
    import requests
    print("📥 Linux environment: Manually downloading tokenizer vocab.json and merges.txt...")
    vocab_url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json"
    merges_url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt"
    
    with open(os.path.join(temp_coreml_dir, "vocab.json"), "wb") as f:
        f.write(requests.get(vocab_url).content)
    with open(os.path.join(temp_coreml_dir, "merges.txt"), "wb") as f:
        f.write(requests.get(merges_url).content)
        
    print(f"📦 Zipping uncompiled .mlpackage assets from {temp_coreml_dir}...")
    shutil.make_archive(output_zip.replace(".zip", ""), 'zip', temp_coreml_dir)

# Clean up temporary folders to save disk space
print("🧹 Cleaning up temporary directories...")
if os.path.exists(temp_fused_dir):
    shutil.rmtree(temp_fused_dir)
if os.path.exists(temp_coreml_dir):
    shutil.rmtree(temp_coreml_dir)

print(f"\n🎉 CoreML Model conversion completed! Archive saved to: {output_zip}")
