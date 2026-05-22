"""Convert base SD 1.5 + LCM-LoRA → CoreML for T2I.
Runs on Linux (no Xcode → produces .mlpackage; locally we recompile to .mlmodelc).
"""
import os, sys, shutil, subprocess, platform

workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(workspace_dir)
print(f"📂 Workspace: {workspace_dir}", flush=True)

base_model_dir   = os.path.join(workspace_dir, "scratch/runwayml_stable-diffusion-v1-5")
lcm_lora_dir     = os.path.join(workspace_dir, "scratch/latent-consistency_lcm-lora-sdv1-5")
fused_dir        = os.path.join(workspace_dir, "scratch/temp_fused_t2i_lcm")
coreml_out_dir   = os.path.join(workspace_dir, "scratch/temp_coreml_t2i")
final_models_dir = os.path.join(workspace_dir, "sd_models_t2i")

for d in (base_model_dir, lcm_lora_dir):
    if not os.path.isdir(d):
        sys.exit(f"❌ Missing source: {d}. Run dl_models_ci.py first.")

print("\n=== Step 1: Load base + fuse LCM-LoRA ===", flush=True)
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_dir, torch_dtype=torch.float32, variant="fp16",
    safety_checker=None, feature_extractor=None, requires_safety_checker=False,
)
print(f"📥 Loading LCM-LoRA: {lcm_lora_dir}", flush=True)
pipe.load_lora_weights(lcm_lora_dir)
print("🧪 Fusing", flush=True)
pipe.fuse_lora()
pipe.unload_lora_weights()
pipe.to(dtype=torch.float32)

if os.path.exists(fused_dir):
    shutil.rmtree(fused_dir)
print(f"💾 save_pretrained → {fused_dir}", flush=True)
pipe.save_pretrained(fused_dir)
del pipe
import gc; gc.collect()

print("\n=== Step 2: torch2coreml ===", flush=True)
if os.path.exists(coreml_out_dir):
    shutil.rmtree(coreml_out_dir)

is_mac = platform.system() == "Darwin"
cmd = [sys.executable, "-m", "python_coreml_stable_diffusion.torch2coreml",
       "--convert-unet", "--chunk-unet",
       "--convert-text-encoder",
       "--convert-vae-decoder",
       "--model-version", fused_dir,
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

print("\n=== Step 3: Stage final dir ===", flush=True)
# On Linux: ship .mlpackage; locally we recompile to .mlmodelc later.
# On macOS: --bundle-resources-for-swift-cli already produced Resources/*.mlmodelc.
if os.path.exists(final_models_dir):
    shutil.rmtree(final_models_dir)
os.makedirs(final_models_dir, exist_ok=True)

want_pkg = [
    "Stable_Diffusion_version__Users_runner_work_*_text_encoder.mlpackage",
    "Stable_Diffusion_version__Users_runner_work_*_unet_chunk1.mlpackage",
    "Stable_Diffusion_version__Users_runner_work_*_unet_chunk2.mlpackage",
    "Stable_Diffusion_version__Users_runner_work_*_vae_decoder.mlpackage",
]
import glob, requests
if is_mac:
    src = os.path.join(coreml_out_dir, "Resources")
    for name in ("TextEncoder.mlmodelc", "UnetChunk1.mlmodelc", "UnetChunk2.mlmodelc",
                 "VAEDecoder.mlmodelc", "vocab.json", "merges.txt"):
        s = os.path.join(src, name); d = os.path.join(final_models_dir, name)
        if os.path.exists(s):
            if os.path.isdir(s): shutil.copytree(s, d)
            else: shutil.copy2(s, d)
            print(f"✅ {name}", flush=True)
else:
    # rename .mlpackage to friendly names
    rename_map = [
        ("text_encoder.mlpackage",  "TextEncoder.mlpackage"),
        ("unet_chunk1.mlpackage",   "UnetChunk1.mlpackage"),
        ("unet_chunk2.mlpackage",   "UnetChunk2.mlpackage"),
        ("vae_decoder.mlpackage",   "VAEDecoder.mlpackage"),
    ]
    # the actual file names from torch2coreml are prefixed with the model_version path
    for src_glob, dst_name in rename_map:
        matches = glob.glob(os.path.join(coreml_out_dir, f"*_{src_glob}"))
        if not matches:
            print(f"⚠️  missing {src_glob}", flush=True); continue
        s = matches[0]
        d = os.path.join(final_models_dir, dst_name)
        shutil.copytree(s, d)
        print(f"✅ {dst_name} ← {os.path.basename(s)}", flush=True)
    # tokenizer files
    print("📥 Downloading tokenizer vocab/merges…", flush=True)
    for fn, url in [
        ("vocab.json",  "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json"),
        ("merges.txt",  "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt"),
    ]:
        with open(os.path.join(final_models_dir, fn), "wb") as f:
            f.write(requests.get(url, timeout=60).content)
        print(f"✅ {fn}", flush=True)

# zip for upload
zip_path = os.path.join(workspace_dir, "sd_models_t2i.zip")
if os.path.exists(zip_path): os.remove(zip_path)
shutil.make_archive(zip_path.replace(".zip",""), "zip", final_models_dir)
print(f"\n🎉 T2I bundle ready: {zip_path}", flush=True)
