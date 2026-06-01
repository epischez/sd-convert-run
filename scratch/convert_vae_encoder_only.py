import os
import sys
import shutil
import subprocess

# Ensure we're in the right working directory
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(workspace_dir)
print(f"📂 Workspace directory set to: {workspace_dir}")

temp_coreml_dir = os.path.join(workspace_dir, "scratch/temp_coreml_vae_encoder")
os.makedirs("scratch", exist_ok=True)

model_id = "Yntec/DisneyPixar"

print("\n=== Step 1: Converting VAE Encoder to CoreML ===")
if os.path.exists(temp_coreml_dir):
    shutil.rmtree(temp_coreml_dir)

is_mac = (sys.platform == "darwin")
cmd = [
    sys.executable, "-m", "python_coreml_stable_diffusion.torch2coreml",
    "--convert-vae-encoder",
    "--model-version", model_id,
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

print("\n=== Step 2: Packaging VAE Encoder ===")
output_zip = os.path.join(workspace_dir, "vae_encoder_only.zip")
if os.path.exists(output_zip):
    os.remove(output_zip)

if is_mac:
    compiled_resources_dir = os.path.join(temp_coreml_dir, "Resources")
    vae_encoder_src = os.path.join(compiled_resources_dir, "VAEEncoder.mlmodelc")
    
    dest_dir = os.path.join(workspace_dir, "sd_models/cartoon3d_lcm")
    os.makedirs(dest_dir, exist_ok=True)
    
    if os.path.exists(vae_encoder_src):
        dest_path = os.path.join(dest_dir, "VAEEncoder.mlmodelc")
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.copytree(vae_encoder_src, dest_path)
        print(f"✅ Copied VAEEncoder.mlmodelc to {dest_path}")
        
    print(f"📦 Zipping compiled VAEEncoder.mlmodelc from {compiled_resources_dir}...")
    shutil.make_archive(output_zip.replace(".zip", ""), 'zip', compiled_resources_dir)
else:
    print(f"📦 Zipping uncompiled .mlpackage assets from {temp_coreml_dir}...")
    shutil.make_archive(output_zip.replace(".zip", ""), 'zip', temp_coreml_dir)

# Clean up temporary folders
print("🧹 Cleaning up temporary directories...")
if os.path.exists(temp_coreml_dir):
    shutil.rmtree(temp_coreml_dir)

print(f"\n🎉 VAE Encoder conversion completed! Archive saved to: {output_zip}")
