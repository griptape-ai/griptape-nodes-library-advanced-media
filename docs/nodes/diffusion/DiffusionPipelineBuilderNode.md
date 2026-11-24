---
description: Loads a diffusion model from your Hugging Face cache, applies LoRAs and memory settings, and keeps the pipeline in memory for the Generate nodes.
when_to_use: Use it once per model setup, before any Generate node, to load a Flux, Qwen, WAN or other diffusion model and reuse it across runs.
---

## When to use it

You want to run a diffusion model on your own machine, for example to make concept frames with Flux or to animate a frame with WAN. The builder loads the model. A Generate node, such as [Generate Image](../image/GenerateImageNode.md), makes the media. Connect the builder's `pipeline` output to the Generate node's `pipeline` input.

Build once and connect several Generate nodes to the same builder. Each one reuses the loaded model.

The `pipeline` output is a reference to the loaded model, not an image. It changes whenever you change a setting on the builder, such as the provider, the model, a LoRA or a memory option. The builder loads the new configuration the next time it runs.

Only one pipeline stays in memory at a time. Building a different configuration unloads the previous one. If a workflow alternates between two builders, each run reloads its model. [Clear Pipeline Cache](../utils/ClearPipelineCacheNode.md) unloads it on demand.

The builder only offers models that are already downloaded to your Hugging Face cache. Use the model's download button to open the Model Manager, or follow the [Hugging Face guide](https://docs.griptapenodes.com/en/stable/guides/integrations/hugging_face/). If the model is missing or its download hasn't finished, the run stops with a message that names it.

## Example

You want a depth pass from a plate without leaving the workflow.

1. Add **Diffusion Pipeline Builder** and set `provider` to `Marigold`.
1. Set `pipeline_type` to `MarigoldDepthPipeline` and pick a downloaded `model`.
1. Add **Generate Media (Diffusion Pipeline)** and connect `pipeline` from the builder.
1. Connect the plate to the new `image` input and run the workflow. The colorized depth map is in `output_image`.

The `qwen_edit_2509_camera_control` template shows a builder with two LoRAs feeding a Generate Media node that runs once for each prompt in a loop.

## Parameters

### provider

The model family. Changing it replaces the provider-specific parameters (`model`, text encoders and so on) and resets `pipeline_type` to the first choice for that provider. Connections on parameters that exist for both providers are kept.

| Provider | `pipeline_type` choices | Produces |
|----------|-------------------------|----------|
| `Flux` | `FluxPipeline`, `FluxFillPipeline`, `FluxKontextPipeline`, `FluxImg2ImgPipeline`, `FluxControlNetPipeline`, `FluxUpscalePipeline` | Image |
| `Flux2` | `Flux2Pipeline`, `Flux2KleinPipeline` | Image |
| `Allegro` | `AllegroPipeline` | Video |
| `Amused` | `AmusedPipeline`, `AmusedImg2ImgPipeline`, `AmusedInpaintPipeline` | Image |
| `AudioLDM` | `AudioLDMPipeline` | Audio |
| `DepthCrafter` | `DepthCrafterVideoDiffusionPipeline` | Video |
| `GLM-Image` | `GlmImagePipeline` | Image |
| `LTX-2` | `LTX2Pipeline`, `LTX2ImageToVideoPipeline` | Video |
| `Marigold` | `MarigoldDepthPipeline`, `MarigoldNormalsPipeline`, `MarigoldIntrinsicsPipeline` | Image (intrinsics gives `albedo`, `component_1` and `component_2`) |
| `Qwen` | `QwenImagePipeline`, `QwenImageImg2ImgPipeline`, `QwenImageEditPipeline`, `QwenImageEditInpaintPipeline`, `QwenImageEditPlusPipeline`, `QwenImageInpaintPipeline`, `QwenImageUpscalePipeline`, `QwenImageControlNetPipeline`, `QwenImageLayeredPipeline` | Image (layered gives a list of images) |
| `Stable Diffusion` | `StableDiffusionPipeline`, `StableDiffusion3Pipeline`, `StableDiffusionAttendAndExcitePipeline`, `StableDiffusionDiffEditPipeline` | Image (DiffEdit also gives `mask_image`) |
| `WAN` | `WanPipeline`, `WanAnimatePipeline`, `WanImageToVideoPipeline`, `WanVideoToVideoPipeline`, `WanVacePipeline` | Video |
| `Wuerstchen` | `WuerstchenPipeline` | Image |
| `Z-Image` | `ZImagePipeline` | Image |
| `Custom` | Every `pipeline_type` above in one searchable list | Depends on the type |

`Custom` lists every model in your Hugging Face cache, not only the ones the other providers suggest, and turns off the compatibility checks. A model that doesn't match the pipeline type fails when the builder runs.

### pipeline_type

Which pipeline of the provider to build. It decides the inputs the Generate node shows, such as an `input_image` for an image-to-video type. Changing it rebuilds the provider-specific parameters.

### loras

Connect one or more [Load LoRA](../lora/LoadLora.md) nodes. The LoRAs are fused into the model when it is built, so adding, removing or reweighting one loads the pipeline again on the next run. If two connected LoRAs point to the same file, the later weight is used.

### memory_optimization_strategy

`Manual` (the default) shows the memory settings below and applies exactly what you choose. `Automatic` hides them and picks settings from the memory the device has free. See [Memory and performance](../../guides/memory-and-performance.md).

### attention_slicing

Shown in `Manual` mode. Computes attention in slices to lower peak memory. See [attention_slicing](../../guides/memory-and-performance.md#attention_slicing).

### vae_slicing

Shown in `Manual` mode. Decodes the image in slices at the end of a run. See [vae_slicing](../../guides/memory-and-performance.md#vae_slicing).

### transformer_layerwise_casting

Shown in `Manual` mode. Stores transformer weights in fp8 and upcasts one layer at a time. See [transformer_layerwise_casting](../../guides/memory-and-performance.md#transformer_layerwise_casting).

### cpu_offload_strategy

Shown in `Manual` mode. Moves parts of the pipeline between the GPU and system memory. See [cpu_offload_strategy](../../guides/memory-and-performance.md#cpu_offload_strategy).

### quantization_mode

Shown in `Manual` mode. Compresses the model weights to fp8, int8 or int4 when the pipeline is built. See [quantization_mode](../../guides/memory-and-performance.md#quantization_mode).
