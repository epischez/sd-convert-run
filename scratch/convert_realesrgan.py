"""
Convert RealESRGAN_x4plus (RRDBNet) → CoreML mlpackage → mlmodelc.

Fixed I/O shape: 512×512 RGB → 2048×2048 RGB, FP16, mlprogram, iOS17.

Outputs under scratch/:
  - RealESRGAN_x4plus.pth          (downloaded weights)
  - RealESRGAN_x4plus.mlpackage    (CoreML mlpackage)
  - RealESRGAN_x4plus.mlmodelc     (compiled, requires macOS + xcrun)
"""
import os, sys, shutil, subprocess, urllib.request

workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(workspace_dir)
print(f"📂 Workspace: {workspace_dir}")

weights_url   = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
weights_path  = os.path.join(workspace_dir, "scratch", "RealESRGAN_x4plus.pth")
mlpackage_out = os.path.join(workspace_dir, "scratch", "RealESRGAN_x4plus.mlpackage")
mlmodelc_out  = os.path.join(workspace_dir, "scratch", "RealESRGAN_x4plus.mlmodelc")

# ───── Step 1: Download weights
if not os.path.exists(weights_path):
    print(f"\n=== Step 1: Downloading RealESRGAN_x4plus weights (~67MB) ===")
    print(f"🔗 {weights_url}")
    urllib.request.urlretrieve(weights_url, weights_path)
    print(f"✅ Saved to {weights_path}")
else:
    print(f"\n=== Step 1: Weights already present at {weights_path} ===")

# ───── Step 2: Define RRDBNet inline
print("\n=== Step 2: Building RRDBNet & loading weights ===")
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                 num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
state = torch.load(weights_path, map_location="cpu")
key = "params_ema" if "params_ema" in state else ("params" if "params" in state else None)
if key is None:
    sys.exit(f"❌ Unrecognized checkpoint keys: {list(state.keys())[:10]}")
net.load_state_dict(state[key], strict=True)
net.eval()
print(f"✅ Loaded RRDBNet weights via key '{key}'")

# Wrapper: input [0,1] → output [0,255] (CoreML ImageType expects 0..255 on output)
class WrappedRealESRGAN(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        out = self.m(x)
        out = torch.clamp(out, 0.0, 1.0) * 255.0
        return out

wrapped = WrappedRealESRGAN(net).eval()

# ───── Step 3: Trace
print("\n=== Step 3: Tracing model at 512×512 ===")
example = torch.zeros(1, 3, 512, 512)
with torch.no_grad():
    traced = torch.jit.trace(wrapped, example)
print("✅ Trace complete")

# ───── Step 4: Convert to CoreML
print("\n=== Step 4: Converting to CoreML (FP16, mlprogram, iOS17) ===")
import coremltools as ct

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(
        name="input",
        shape=(1, 3, 512, 512),
        color_layout=ct.colorlayout.RGB,
        scale=1.0 / 255.0,
    )],
    outputs=[ct.ImageType(
        name="output",
        color_layout=ct.colorlayout.RGB,
    )],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17,
    convert_to="mlprogram",
)
mlmodel.short_description = "Real-ESRGAN x4plus, 512→2048 RGB, FP16"
mlmodel.author = "xinntao (Real-ESRGAN)"
mlmodel.license = "BSD-3-Clause"

if os.path.exists(mlpackage_out):
    shutil.rmtree(mlpackage_out)
mlmodel.save(mlpackage_out)
print(f"✅ Saved mlpackage: {mlpackage_out}")

# ───── Step 5: Compile to .mlmodelc (requires macOS + xcrun)
print("\n=== Step 5: Compiling mlpackage → mlmodelc ===")
xcrun_available = shutil.which("xcrun") is not None
if not xcrun_available:
    print("⚠️ xcrun not found — skipping compile step (Linux runner).")
    print("   Ship the .mlpackage and compile via MLModel.compileModel(at:) at runtime.")
    sys.exit(0)

tmp_compile_dir = os.path.join(workspace_dir, "scratch", "_realesrgan_compiled_tmp")
if os.path.exists(tmp_compile_dir):
    shutil.rmtree(tmp_compile_dir)
os.makedirs(tmp_compile_dir)
subprocess.run([
    "xcrun", "coremlcompiler", "compile", mlpackage_out, tmp_compile_dir
], check=True)

src = os.path.join(tmp_compile_dir, "RealESRGAN_x4plus.mlmodelc")
if not os.path.isdir(src):
    sys.exit(f"❌ Expected compiled output at {src}")
if os.path.exists(mlmodelc_out):
    shutil.rmtree(mlmodelc_out)
shutil.copytree(src, mlmodelc_out)
shutil.rmtree(tmp_compile_dir)
print(f"\n🎉 SR CoreML bundle ready: {mlmodelc_out}")
