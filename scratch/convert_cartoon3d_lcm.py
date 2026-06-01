import os
import sys
import shutil
import subprocess
import requests

# Ensure we're in the right working directory
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(workspace_dir)
print(f"📂 Workspace directory set to: {workspace_dir}")

# Create scratch/temp dirs if not exists
temp_fused_dir = os.path.join(workspace_dir, "scratch/temp_fused_cartoon3d_lcm")
temp_coreml_dir = os.path.join(workspace_dir, "scratch/temp_coreml_cartoon3d")
os.makedirs("scratch", exist_ok=True)

print("\n=== Step 1: Merging Yntec/DisneyPixar with LCM-LoRA ===")
try:
    from diffusers import StableDiffusionPipeline
    import torch
except ImportError:
    print("❌ Error: Missing required Python packages. Run: pip install diffusers transformers peft accelerate torch")
    sys.exit(1)

model_id = "Yntec/DisneyPixar"
lora_id = "latent-consistency/lcm-lora-sdv1-5"

print(f"📥 Loading base model from Hugging Face: {model_id}")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32
)

# Select computation device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and os.environ.get("GITHUB_ACTIONS") != "true":
    device = "mps"
else:
    device = "cpu"

print(f"⚙️ Using device: {device}")
pipe.to(device)

print(f"📥 Loading LCM-LoRA weights from Hugging Face: {lora_id}")
pipe.load_lora_weights(lora_id)

print(f"🧪 Fusing LoRA weights into UNet on {device}...")
pipe.fuse_lora()
pipe.unload_lora_weights()

print("🧹 Moving model back to CPU and keeping float32 precision...")
pipe.to(device="cpu", dtype=torch.float32)

print(f"💾 Saving fused model locally to: {temp_fused_dir}")
pipe.save_pretrained(temp_fused_dir)

# Clean memory
del pipe
import gc
gc.collect()

print("\n=== Step 2: Converting fused model to CoreML ===")
if os.path.exists(temp_coreml_dir):
    shutil.rmtree(temp_coreml_dir)

# Patch ml-stable-diffusion to prevent text_encoder quantization and ensure unet chunks are quantized
patch_file = os.path.join(workspace_dir, "ml-stable-diffusion/python_coreml_stable_diffusion/torch2coreml.py")
if os.path.exists(patch_file):
    print(f"🔧 Patching {patch_file} to exclude text encoder from quantization and include UNet chunks...")
    with open(patch_file, "r") as f:
        content = f.read()
    
    target_str = 'for model_name in ["text_encoder", "text_encoder_2", "unet", "refiner", "control-unet"]:'
    replacement_str = 'for model_name in ["unet", "unet_chunk1", "unet_chunk2", "refiner", "refiner_chunk1", "refiner_chunk2", "control-unet", "control-unet_chunk1", "control-unet_chunk2"]:'
    
    if target_str in content:
        content = content.replace(target_str, replacement_str)
        with open(patch_file, "w") as f:
            f.write(content)
        print("✅ Patch applied successfully!")
    else:
        if replacement_str in content:
            print("ℹ️ Patch already applied.")
        else:
            print("⚠️ Warning: Quantization target loop pattern not found in torch2coreml.py.")
else:
    print(f"⚠️ Warning: Could not find {patch_file} to patch.")

# Construct conversion command for T2I pipeline
is_mac = (sys.platform == "darwin")
cmd = [
    sys.executable, "-m", "python_coreml_stable_diffusion.torch2coreml",
    "--convert-text-encoder",
    "--convert-unet",
    "--chunk-unet",
    "--convert-vae-decoder",
    "--convert-vae-encoder",
    "--model-version", temp_fused_dir,
    "--attention-implementation", "ORIGINAL",
    "--compute-unit", "ALL",
    "--latent-h", "64",
    "--latent-w", "64",
    "-o", temp_coreml_dir,
]

if is_mac:
    # Xcode compilers are only available on macOS
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
output_zip = os.path.join(workspace_dir, "sd_models_cartoon3d.zip")
if os.path.exists(output_zip):
    os.remove(output_zip)

if is_mac:
    compiled_resources_dir = os.path.join(temp_coreml_dir, "Resources")
    print(f"📦 Zipping compiled .mlmodelc assets from {compiled_resources_dir}...")
    shutil.make_archive(output_zip.replace(".zip", ""), 'zip', compiled_resources_dir)
    
    # Deploy compiled chunked models
    unet1_src = os.path.join(compiled_resources_dir, "UnetChunk1.mlmodelc")
    unet2_src = os.path.join(compiled_resources_dir, "UnetChunk2.mlmodelc")
    text_encoder_src = os.path.join(compiled_resources_dir, "TextEncoder.mlmodelc")
    vae_decoder_src = os.path.join(compiled_resources_dir, "VAEDecoder.mlmodelc")
    vae_encoder_src = os.path.join(compiled_resources_dir, "VAEEncoder.mlmodelc")
    
    dest_dirs = [
        os.path.join(workspace_dir, "sd_models/cartoon3d_lcm"),
        os.path.join(workspace_dir, "MLImageKitDemo/sd_models/cartoon3d_lcm")
    ]
    
    for d_dir in dest_dirs:
        os.makedirs(d_dir, exist_ok=True)
        # Copy components
        for src, name in [
            (unet1_src, "UnetChunk1.mlmodelc"),
            (unet2_src, "UnetChunk2.mlmodelc"),
            (text_encoder_src, "TextEncoder.mlmodelc"),
            (vae_decoder_src, "VAEDecoder.mlmodelc"),
            (vae_encoder_src, "VAEEncoder.mlmodelc")
        ]:
            if os.path.exists(src):
                dest_path = os.path.join(d_dir, name)
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.copytree(src, dest_path)
                print(f"✅ Copied {name} to {dest_path}")
                
        # Copy vocab.json and merges.txt
        for f_name in ["vocab.json", "merges.txt"]:
            src_f = os.path.join(compiled_resources_dir, f_name)
            if os.path.exists(src_f):
                dest_f = os.path.join(d_dir, f_name)
                if os.path.exists(dest_f):
                    os.remove(dest_f)
                shutil.copy(src_f, dest_f)
                print(f"✅ Copied {f_name} to {dest_f}")
else:
    # On Linux (Google Colab / Ubuntu Runner), we have .mlpackage folders.
    # We must manually download vocab.json and merges.txt since bundle-resources-for-swift-cli was skipped.
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
