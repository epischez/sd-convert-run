# SD Dual Convert (T2I + Inpaint, CoreML)

Ephemeral CI repo used to convert Stable Diffusion 1.5 variants to Apple CoreML for the [MLImageKitDemo](https://github.com/epischez/MLImageKitDemo) iOS plugin:

- **T2I**: `runwayml/stable-diffusion-v1-5` + `latent-consistency/lcm-lora-sdv1-5` (LCM 4-step inference)
- **Inpaint**: `runwayml/stable-diffusion-inpainting` (no LoRA — pure inpaint UNet, DPM 20-step)

Triggered manually via `gh workflow run sd_convert.yml -f kind=both`. Artifacts (`sd_models_t2i.zip` / `sd_models_inpaint.zip`) contain `.mlpackage` bundles + tokenizer; recompile locally to `.mlmodelc` via `xcrun coremlcompiler`.

Public repo because the source workflow needs to clone Apple's `ml-stable-diffusion` and pull HF artifacts; nothing private is committed here.
