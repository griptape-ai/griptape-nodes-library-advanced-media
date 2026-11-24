---
description: Points to a LoRA file on disk and gives it a weight, for the loras input of a Diffusion Pipeline Builder.
when_to_use: Use it when you have a trained LoRA, such as a show's look or a character, that you want a local diffusion pipeline to apply.
---

## When to use it

You have a LoRA file on disk and want the model to use it. Set `file_path`, choose a `weight`, and connect `loras` to the `loras` input of a [Diffusion Pipeline Builder](../diffusion/DiffusionPipelineBuilderNode.md). Add one Load LoRA node per LoRA. The builder's `loras` input takes any number of them.

The node doesn't load anything itself. It outputs the path and weight, and the builder reads the file when it builds the pipeline. The LoRA is fused into the model at that point, so changing the file or the weight loads the pipeline again on the next run.

The library also has ready-made nodes for specific Flux and WAN LoRAs in the LoRA category. Use Load LoRA for files you trained or downloaded yourself.

## Example

A studio has trained a LoRA on its look-development stills and wants frames in that style.

1. Add **Load LoRA** and set `file_path` to the `.safetensors` file.
1. Set `weight` to 0.8 to soften the effect.
1. Connect `loras` to the `loras` input of the builder and run the builder.
1. Generate with a [Generate Image](../image/GenerateImageNode.md) node.

## Parameters

### file_path

The node stops with an error if no file exists at the path. The picker offers `.safetensors`, `.sft`, `.pt`, `.bin`, `.json` and `.lora` files, but the builder reads the file with the safetensors loader, so the file must be in safetensors format whatever its extension.

### weight

From 0 to 1, default 1. It scales how strongly the LoRA affects the result.

### loras

The path and weight, ready to connect to a builder.
