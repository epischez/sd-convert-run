import os
import sys
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct
import shutil
import subprocess

# Globally override torch.autograd.profiler.record_function to bypass JIT tracer recording profiler ops
class DummyRecordFunction:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args, **kwargs):
        pass

torch.autograd.profiler.record_function = DummyRecordFunction
if hasattr(torch, 'profiler'):
    torch.profiler.record_function = DummyRecordFunction

# 1. Ensure MAT repo is cloned
mat_repo_path = os.path.abspath("scratch/MAT_repo")
if not os.path.exists(mat_repo_path):
    print(f"Cloning MAT repository into {mat_repo_path}...")
    os.makedirs("scratch", exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/fenglinglwb/MAT.git", mat_repo_path], check=True)

# Add MAT repo to sys.path so we can import legacy loader and generator
sys.path.insert(0, mat_repo_path)

# Monkey patch profiled_function to bypass JIT tracing errors with profiler ops
import torch_utils.misc as misc
misc.profiled_function = lambda fn: fn

# 2. Monkey patch upfirdn2d and bias_act BEFORE importing networks to bypass MKL/CUDA code
# Also monkey patch torch.full to support numpy scalars (e.g. np.float32) in newer NumPy versions
import torch_utils.ops.bias_act as bias_act
import torch_utils.ops.upfirdn2d as upfirdn2d

orig_torch_full = torch.full
def patched_torch_full(size, fill_value, *args, **kwargs):
    if hasattr(fill_value, 'item'):
        fill_value = fill_value.item()
    return orig_torch_full(size, fill_value, *args, **kwargs)
torch.full = patched_torch_full

def pure_pytorch_bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None, impl='cuda'):
    # Reshape bias to align with dim
    if b is not None:
        shape = [1] * x.ndim
        shape[dim] = b.shape[0]
        x = x + b.view(shape)
        
    # Activation
    if act == 'relu':
        x = F.relu(x)
    elif act == 'lrelu':
        alpha = alpha if alpha is not None else 0.2
        x = F.leaky_relu(x, negative_slope=alpha)
    elif act == 'tanh':
        x = torch.tanh(x)
    elif act == 'sigmoid':
        x = torch.sigmoid(x)
    elif act == 'swish':
        x = torch.sigmoid(x) * x
        
    if gain is not None and gain != 1:
        x = x * gain
        
    if clamp is not None and clamp >= 0:
        x = torch.clamp(x, -clamp, clamp)
        
    return x

bias_act.bias_act = pure_pytorch_bias_act

def pure_pytorch_upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    if isinstance(up, int):
        upx, upy = up, up
    else:
        upx, upy = up[0], up[1]
        
    if isinstance(down, int):
        downx, downy = down, down
    else:
        downx, downy = down[0], down[1]
        
    if isinstance(padding, int):
        padx0, padx1, pady0, pady1 = padding, padding, padding, padding
    elif len(padding) == 2:
        padx0, padx1, pady0, pady1 = padding[0], padding[0], padding[1], padding[1]
    else:
        padx0, padx1, pady0, pady1 = padding[0], padding[1], padding[2], padding[3]
        
    batch_size, num_channels, in_height, in_width = x.shape
    
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
        
    # Upsample by inserting zeros (sequential rank-4 operations to avoid CoreML rank-5 limitation)
    if upy > 1:
        x = x.reshape(batch_size * num_channels, in_height, 1, in_width)
        x = torch.nn.functional.pad(x, [0, 0, 0, upy - 1])
        x = x.reshape(batch_size, num_channels, in_height * upy, in_width)
        in_height = in_height * upy
    if upx > 1:
        x = x.reshape(batch_size * num_channels, in_height, in_width, 1)
        x = torch.nn.functional.pad(x, [0, upx - 1])
        x = x.reshape(batch_size, num_channels, in_height, in_width * upx)
        
    # Pad or crop
    pad_l = max(padx0, 0)
    pad_r = max(padx1, 0)
    pad_t = max(pady0, 0)
    pad_b = max(pady1, 0)
    if pad_l > 0 or pad_r > 0 or pad_t > 0 or pad_b > 0:
        x = torch.nn.functional.pad(x, [pad_l, pad_r, pad_t, pad_b])
        
    crop_l = max(-padx0, 0)
    crop_r = max(-padx1, 0)
    crop_t = max(-pady0, 0)
    crop_b = max(-pady1, 0)
    if crop_l > 0 or crop_r > 0 or crop_t > 0 or crop_b > 0:
        h_end = x.shape[2] - crop_b
        w_end = x.shape[3] - crop_r
        x = x[:, :, crop_t:h_end, crop_l:w_end]
        
    # Filter
    f = f * (gain ** (f.ndim / 2.0))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))
        
    # Convolve
    if f.ndim == 1:
        # Separable
        f_w = f.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1)
        x = torch.nn.functional.conv2d(x, f_w, groups=num_channels, padding=(0, 0))
        f_h = f.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1)
        x = torch.nn.functional.conv2d(x, f_h, groups=num_channels, padding=(0, 0))
    else:
        # 2D Filter
        f_2d = f.unsqueeze(0).unsqueeze(0).repeat(num_channels, 1, 1, 1)
        x = torch.nn.functional.conv2d(x, f_2d, groups=num_channels, padding=(0, 0))
        
    # Downsample
    if downx > 1 or downy > 1:
        x = x[:, :, ::downy, ::downx]
        
    return x

upfirdn2d.upfirdn2d = pure_pytorch_upfirdn2d

# Monkey patch conv2d_resample to:
# 1. Cast groups to int for JIT tracer compatibility.
# 2. Bypass transpose convolution (when up > 1) because CoreML does not support dynamic weights for transpose conv.
#    Instead, it falls back to the reference path which uses upfirdn2d upsampling + standard conv2d.
import torch_utils.ops.conv2d_resample as conv2d_resample
def patched_conv2d_resample(x, w, f=None, up=1, down=1, padding=0, groups=1, flip_weight=True, flip_filter=False):
    from torch_utils.ops.conv2d_resample import _get_weight_shape, _get_filter_size, _parse_padding, _conv2d_wrapper
    import torch_utils.ops.upfirdn2d as upfirdn2d
    
    if hasattr(groups, 'item'):
        groups = int(groups.item())
    elif not isinstance(groups, int):
        try:
            groups = int(groups)
        except Exception:
            groups = 1
            
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and (x.ndim == 4)
    assert isinstance(w, torch.Tensor) and (w.ndim == 4) and (w.dtype == x.dtype)
    assert f is None or (isinstance(f, torch.Tensor) and f.ndim in [1, 2] and f.dtype == torch.float32)
    assert isinstance(up, int) and (up >= 1)
    assert isinstance(down, int) and (down >= 1)
    assert isinstance(groups, int) and (groups >= 1)
    out_channels, in_channels_per_group, kh, kw = _get_weight_shape(w)
    fw, fh = _get_filter_size(f)
    px0, px1, py0, py1 = _parse_padding(padding)

    # Adjust padding to account for up/downsampling.
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2

    # Fast path: 1x1 convolution with downsampling only => downsample first, then convolve.
    if kw == 1 and kh == 1 and (down > 1 and up == 1):
        x = upfirdn2d.upfirdn2d(x=x, f=f, down=down, padding=[px0,px1,py0,py1], flip_filter=flip_filter)
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        return x

    # Fast path: 1x1 convolution with upsampling only => convolve first, then upsample.
    if kw == 1 and kh == 1 and (up > 1 and down == 1):
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        x = upfirdn2d.upfirdn2d(x=x, f=f, up=up, padding=[px0,px1,py0,py1], gain=up**2, flip_filter=flip_filter)
        return x

    # Fast path: downsampling only => use strided convolution.
    if down > 1 and up == 1:
        x = upfirdn2d.upfirdn2d(x=x, f=f, padding=[px0,px1,py0,py1], flip_filter=flip_filter)
        x = _conv2d_wrapper(x=x, w=w, stride=down, groups=groups, flip_weight=flip_weight)
        return x

    # [CoreML Bypass] Transpose conv (up > 1) fast path is omitted here.
    # It will fall through to the reference implementation below.

    # Fast path: no up/downsampling, padding supported by the underlying implementation => use plain conv2d.
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and px0 >= 0 and py0 >= 0:
            return _conv2d_wrapper(x=x, w=w, padding=[py0,px0], groups=groups, flip_weight=flip_weight)

    # Fallback: Generic reference implementation.
    x = upfirdn2d.upfirdn2d(x=x, f=(f if up > 1 else None), up=up, padding=[px0,px1,py0,py1], gain=up**2, flip_filter=flip_filter)
    x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
    if down > 1:
        x = upfirdn2d.upfirdn2d(x=x, f=f, down=down, flip_filter=flip_filter)
    return x

conv2d_resample.conv2d_resample = patched_conv2d_resample

# Now import generator safely
from networks.mat import Generator, SwinTransformerBlock, window_partition, window_reverse

def patched_swin_forward(self, x, x_size, mask=None):
    H, W = x_size
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x
    x = x.view(B, H, W, C)
    if mask is not None:
        mask = mask.view(B, H, W, 1)

    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        if mask is not None:
            shifted_mask = torch.roll(mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x
        if mask is not None:
            shifted_mask = mask

    # partition windows
    x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
    if mask is not None:
        mask_windows = window_partition(shifted_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
    else:
        mask_windows = None

    # Use registered attn_mask directly to avoid tracing calculate_mask slice assignments
    attn_windows, mask_windows = self.attn(x_windows, mask_windows, mask=self.attn_mask)

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
    if mask is not None:
        mask_windows = mask_windows.view(-1, self.window_size, self.window_size, 1)
        shifted_mask = window_reverse(mask_windows, self.window_size, H, W)

    shifted_x = shifted_x.view(B, H * W, C)
    if mask is not None:
        shifted_mask = shifted_mask.view(B, H * W, 1)

    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x.view(B, H, W, C), shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, H * W, C)
        if mask is not None:
            mask = torch.roll(shifted_mask.view(B, H, W, 1), shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            mask = mask.view(B, H * W, 1)
    else:
        x = shifted_x
        if mask is not None:
            mask = shifted_mask

    # Split weights of self.fuse to bypass cat + FullyConnectedLayer ANE compiler crash
    w = self.fuse.weight * self.fuse.weight_gain
    w1 = w[:, :self.dim]
    w2 = w[:, self.dim:]
    
    x1 = shortcut.matmul(w1.t())
    x2 = x.matmul(w2.t())
    x_sum = x1 + x2
    
    b = self.fuse.bias
    if b is not None and self.fuse.bias_gain != 1:
        b = b * self.fuse.bias_gain
        
    import torch_utils.ops.bias_act as bias_act
    x = bias_act.bias_act(x_sum, b, act=self.fuse.activation, dim=x_sum.ndim-1)
    
    x = self.mlp(x)
    return x, mask

SwinTransformerBlock.forward = patched_swin_forward

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    print(f"Downloading {url} to {dest_path}...")
    
    def report(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, read_so_far * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({read_so_far}/{total_size} bytes)")
        else:
            sys.stdout.write(f"\rDownloaded: {read_so_far} bytes")
        sys.stdout.flush()
        
    urllib.request.urlretrieve(url, dest_path, report)
    print("\nDownload complete!")

# Wrapper module for CoreML conversion
class MATCoreMLWrapper(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        # z is z_dim=512, c is c_dim=0.
        # Register fixed tensors for z and c
        self.register_buffer('fixed_z', torch.zeros(1, 512))
        self.register_buffer('fixed_c', torch.zeros(1, 0))

    def forward(self, image, mask):
        # image shape: (1, 3, 512, 512), values in range [-1.0, 1.0] (from CoreML input scale & bias)
        # mask shape: (1, 1, 512, 512), values in range [0.0, 1.0] (from CoreML input scale & bias)
        
        # Run MAT generator
        # noise_mode='const' makes it deterministic (using fixed noise tensors)
        out = self.generator(
            image,
            mask,
            self.fixed_z,
            self.fixed_c,
            truncation_psi=1.0,
            noise_mode='const'
        )
        
        # out shape: (1, 3, 512, 512), range: [-1.0, 1.0]
        # Denormalize output to [0, 255] for ct.ImageType output
        output_image = (out + 1.0) * 127.5
        return output_image

def main():
    # Make sure output directories exist
    os.makedirs("scratch", exist_ok=True)
    
    # Download weights
    weight_url = "https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth"
    weight_path = "scratch/Places_512_FullData_G.pth"
    download_file(weight_url, weight_path)
    
    # Initialize Generator architecture
    print("Initializing Generator architecture...")
    G = Generator(
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=512,
        img_channels=3,
        synthesis_kwargs={'use_noise': False}
    ).eval().requires_grad_(False)
    
    # Load model state dict
    print(f"Loading checkpoint from {weight_path}...")
    state_dict = torch.load(weight_path, map_location='cpu')
    G.load_state_dict(state_dict)
    
    # Create Wrapper
    wrapper = MATCoreMLWrapper(G).eval()
    
    # Prepare sample inputs for tracing
    sample_image = torch.zeros(1, 3, 512, 512)
    sample_mask = torch.zeros(1, 1, 512, 512)
    
    print("Tracing model with PyTorch JIT...")
    import traceback
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, (sample_image, sample_mask))
    except Exception as e:
        print("❌ Tracing failed with exception:")
        traceback.print_exc()
        sys.exit(1)
        
    print("Converting to CoreML model...")
    # Configure input and output types
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, 512, 512),
        scale=2.0 / 255.0,
        bias=[-1.0, -1.0, -1.0],
        color_layout=ct.colorlayout.RGB
    )
    
    mask_input = ct.ImageType(
        name="mask",
        shape=(1, 1, 512, 512),
        scale=1.0 / 255.0,
        bias=[0.0],
        color_layout=ct.colorlayout.GRAYSCALE
    )
    
    output_image = ct.ImageType(
        name="output_image",
        color_layout=ct.colorlayout.RGB
    )
    
    # Perform conversion
    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input, mask_input],
        outputs=[output_image],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=None, # defaults to float16
        skip_model_load=True
    )
    
    # Set model metadata
    mlmodel.author = "MAT Authors (CVPR 2022) / CoreML conversion by Antigravity"
    mlmodel.license = "Academic / Open-source"
    mlmodel.short_description = "Mask-Aware Transformer (MAT) for Large Hole Image Inpainting"
    mlmodel.input_description["image"] = "Input image patch to be inpainted (512x512, RGB)"
    mlmodel.input_description["mask"] = "Input binary mask patch (512x512, Grayscale, 1 for hole, 0 for keep)"
    mlmodel.output_description["output_image"] = "Inpainted result image (512x512, RGB)"
    
    # Save the mlpackage
    mlpackage_path = "scratch/MAT.mlpackage"
    if os.path.exists(mlpackage_path):
        shutil.rmtree(mlpackage_path)
    mlmodel.save(mlpackage_path)
    print(f"✅ CoreML model saved to {mlpackage_path}")
    
    # Compile the model to .mlmodelc if xcrun is available (macOS)
    if shutil.which("xcrun"):
        compiled_path = "scratch/MAT.mlmodelc"
        if os.path.exists(compiled_path):
            shutil.rmtree(compiled_path)
            
        print(f"Compiling {mlpackage_path} to {compiled_path}...")
        try:
            cmd = [
                "xcrun", "coremlcompiler", "compile",
                mlpackage_path,
                "scratch/"
            ]
            subprocess.run(cmd, check=True)
            print(f"✅ Compilation complete: {compiled_path}")
        except Exception as e:
            print(f"⚠️ Compilation failed: {e}")
    else:
        print("ℹ️ 'xcrun' not found. Skipping compilation to .mlmodelc. The iOS app can compile the .mlpackage on-the-fly.")

if __name__ == "__main__":
    main()
