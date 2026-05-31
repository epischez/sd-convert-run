import os
import sys
import coremltools as ct
import torch
import torch.nn as nn

# A wrapper module around LaMa which handles some input/output pre- and post-processing
class CoreMLaMa(nn.Module):
    def __init__(self, lama):
        super(CoreMLaMa, self).__init__()
        self.lama = lama

    def forward(self, image, mask):
        # image shape: (1, 3, 800, 800)
        # mask shape: (1, 1, 800, 800)
        normalized_mask = ((mask > 0) * 1).byte()
        lama_out = self.lama(image, normalized_mask)
        output = torch.clamp(lama_out * 255, min=0, max=255)
        return output

def main():
    os.makedirs("scratch", exist_ok=True)
    
    # We import iopaint dynamically after pip installs it in the action
    from iopaint.model.lama import LaMa
    print("Initializing LaMa model...")
    model_manager = LaMa("cpu")
    lama_inpaint_model = model_manager.model
    model = CoreMLaMa(lama_inpaint_model).eval()

    size = (800, 800)
    image_shape=(1, 3, size[1], size[0])
    mask_shape=(1, 1, size[1], size[0])

    print("Scripting CoreMLaMa...")
    jit_model = torch.jit.script(model)

    print("Converting model to CoreML...")
    coreml_model = ct.convert(
        jit_model,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        inputs=[
            ct.ImageType(name="image",
                         shape=image_shape,
                         scale=1/255.0),
            ct.ImageType(
                name="mask",
                shape=mask_shape,
                color_layout=ct.colorlayout.GRAYSCALE)
        ],
        outputs=[ct.ImageType(name="output")],
        skip_model_load=True
    )

    coreml_model_file_name = "scratch/LaMa.mlpackage"
    print(f"Saving model to {coreml_model_file_name}...")
    if os.path.exists(coreml_model_file_name):
        import shutil
        shutil.rmtree(coreml_model_file_name)
    coreml_model.save(coreml_model_file_name)
    print("Done converting!")

    # Compile the model to .mlmodelc if xcrun is available (macOS)
    import shutil
    import subprocess
    if shutil.which("xcrun"):
        compiled_path = "scratch/LaMa.mlmodelc"
        if os.path.exists(compiled_path):
            shutil.rmtree(compiled_path)
            
        print(f"Compiling {coreml_model_file_name} to {compiled_path}...")
        try:
            cmd = [
                "xcrun", "coremlcompiler", "compile",
                coreml_model_file_name,
                "scratch/"
            ]
            subprocess.run(cmd, check=True)
            print(f"✅ Compilation complete: {compiled_path}")
        except Exception as e:
            print(f"⚠️ Compilation failed: {e}")
            sys.exit(1)
    else:
        print("xcrun not found, skipping compilation.")

if __name__ == "__main__":
    main()
