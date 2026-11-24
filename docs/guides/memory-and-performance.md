---
title: Memory and performance
description: How the Diffusion Pipeline Builder uses GPU memory, what each memory setting does, and what to try when a run is slow or runs out of memory.
when_to_use: Read it when a diffusion run fails with an out-of-memory error, runs slower than expected, or you need to choose between Manual and Automatic memory settings.
---

The [Diffusion Pipeline Builder](../nodes/diffusion/DiffusionPipelineBuilderNode.md) has a `memory_optimization_strategy` parameter. It is `Manual` by default. In `Manual` mode every memory setting starts off and the whole pipeline is moved to the GPU, so a model larger than your free GPU memory runs out of memory until you turn something on. If you aren't sure what your GPU can hold, switch to `Automatic`, except for the pipelines listed under [Automatic mode](#automatic-mode).

The upstream [Diffusers memory guide](https://huggingface.co/docs/diffusers/main/en/optimization/memory) explains the underlying techniques.

## Automatic mode

`Automatic` hides the manual settings and chooses from the memory the device has free when the pipeline is built. When the model fits, it only turns on slicing and moves the pipeline to the GPU. It never uses quantization.

On a CUDA GPU it works down this list and stops as soon as the pipeline fits. The fit check allows 20 percent headroom over the model's size.

1. Turns on VAE slicing, and attention slicing if less than 8 GB of GPU memory is free.
1. Moves the whole pipeline to the GPU if it fits.
1. Otherwise stores the transformer weights in fp8 (layerwise casting) and checks again. This step is skipped for pre-quantized models and for pipelines that don't support it, such as Z-Image. A pipeline that has no transformer fails here with an error when memory is short: Stable Diffusion 1.x, Wuerstchen, AudioLDM, Marigold and DepthCrafter. Use `Manual` for those pipelines.
1. Otherwise turns on model offload if the largest single component fits in free memory, and sequential offload if it doesn't.
1. If the pipeline still doesn't fit, it logs a warning and runs anyway.

Pipelines that load with a device map, such as GLM-Image, only get the slicing options. The loader decides where their parts live.

On Apple Silicon it moves the pipeline to the GPU if it fits in the memory macOS recommends for it. If it doesn't fit, it turns on VAE slicing and leaves the pipeline on the CPU, which is slow.

## Manual settings

`Manual` applies your choices in this order: quantization, attention slicing, VAE slicing, layerwise casting, then offload or placement on the GPU. Pipelines that load with a device map skip the offload and placement step.

### attention_slicing

Computes attention in slices instead of all at once, which lowers peak memory during denoising and makes it slower. It has no effect on a pipeline that doesn't support it. Automatic mode turns it on for CUDA GPUs with less than 8 GB free.

### vae_slicing

Splits a batch into single items for the final decode. These nodes generate one image or clip per run, so it often changes little. Automatic mode turns it on for every run on a CUDA GPU.

### transformer_layerwise_casting

Stores the transformer weights in fp8 (`float8_e4m3fn`) and converts one layer at a time to bfloat16 while it runs. The weights take about half the memory of bfloat16 and each step costs a little extra time.

It only applies to pipelines with a transformer, so it does nothing on pipelines built on a UNet. It is skipped for pre-quantized models and for Z-Image. In those cases the build continues and the engine log notes the skip.

### cpu_offload_strategy

Decides where the parts of the pipeline live between steps.

| Choice | What happens |
|--------|--------------|
| `None` | The whole pipeline is moved to the GPU. No offload overhead, the most GPU memory. |
| `Model` | One whole component, such as the text encoder, transformer or VAE, sits on the GPU while it works. The rest wait in system memory. |
| `Sequential` | Components are streamed to the GPU a layer at a time. The least GPU memory and the slowest. |

If the pipeline doesn't support the offload mode you pick, the node logs a warning and leaves the pipeline on the CPU. System memory has to hold whatever isn't on the GPU.

### quantization_mode

Compresses the pipeline weights with `optimum-quanto` to `fp8`, `int8` or `int4`. Fewer bits use less memory and can cost more quality, and `int4` weights take roughly a quarter of the space of bfloat16. All components with weights are quantized, including the text encoders and the VAE, except layers named `proj_out`.

Quantization runs every time the pipeline is built, so you pay for it again after [Clear Pipeline Cache](../nodes/utils/ClearPipelineCacheNode.md) or an engine restart. It is skipped for models that are already quantized.

### If you run out of memory

Add one setting at a time. This is the order Automatic mode follows, with quantization added at the end.

1. `vae_slicing`, then `attention_slicing`.
1. `transformer_layerwise_casting`.
1. `cpu_offload_strategy` set to `Model`, then `Sequential`.
1. `quantization_mode`, starting at `fp8` and moving to `int8` or `int4` if needed.

## The pipeline cache

The library keeps one built pipeline in memory. Reuse and eviction follow from that.

- Several Generate nodes connected to one builder share the loaded model. This is the fastest setup.
- Running a builder with a different configuration unloads the pipeline that was loaded. Workflows that alternate between two builders reload a model on each switch.
- Changing any builder setting, including a LoRA or its weight, makes a new configuration. The model loads again the next time the builder runs.
- A Generate node whose pipeline has gone missing rebuilds it through its connected builder.
- After an out-of-memory error during generation, the node unloads the pipeline so the next run starts from a clean state. A failed build removes the partly built pipeline.
- [Clear Pipeline Cache](../nodes/utils/ClearPipelineCacheNode.md) unloads it on demand.

The Diffusers library keeps its own pipeline in memory, separate from this one. Using both in one session can hold two copies of the same weights.

## Previews and intermediate files

The library settings under `advanced_media_library` control the intermediates folder.

- `enable_image_preview_intermediates` is off by default. Turning it on shows intermediate images while an image generates, and slows the run down.
- Many image pipelines write a black placeholder image to the intermediates folder at the start of a run, whatever the preview setting.
- `enable_directory_cleanup` is off by default. Turn it on to delete the oldest files once the folder passes `max_directory_size_gb` (5 by default).

## Troubleshooting

### The model isn't in the list or won't load

The builder loads models from your Hugging Face cache and only offers the ones already downloaded. Download the model from the Model Manager, then use the refresh button next to `model`. If the run stops because a download hasn't finished, wait for the Model Manager to complete it. Account and token setup is in the [Hugging Face guide](https://docs.griptapenodes.com/en/stable/guides/integrations/hugging_face/).

### A Generate node can't find its pipeline

Connect a builder to the `pipeline` input and run the workflow. The node rebuilds a missing pipeline through the connected builder. If the builder's own run failed, fix that error first.

### Out-of-memory errors

Follow the order under [If you run out of memory](#if-you-run-out-of-memory), or switch to `Automatic` (not for Stable Diffusion 1.x, Wuerstchen, AudioLDM, Marigold or DepthCrafter). Generating at a smaller `width` and `height` also lowers memory use.

### The builder fails with a model that doesn't match

With the `Custom` provider all compatibility checks are off, so a model that doesn't match the pipeline type fails when the builder runs. Use a named provider to get checked choices.
