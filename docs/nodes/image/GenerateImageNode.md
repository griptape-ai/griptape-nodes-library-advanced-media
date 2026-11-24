---
description: Generates an image from a Diffusion Pipeline Builder pipeline, such as Flux, Qwen or Stable Diffusion, running on your own machine.
when_to_use: Use it to make concept frames, edits or other still images with a locally built diffusion pipeline.
---

## When to use it

You want a still image from a model that runs on your own GPU. Build the model once with a [Diffusion Pipeline Builder](../diffusion/DiffusionPipelineBuilderNode.md), then connect its `pipeline` output here. The builder can offer text-to-image, image-to-image, inpainting, editing, ControlNet and upscaling types, and this node shows the inputs that type needs.

The node is identical to [Generate Media](../diffusion/DiffusionPipelineRuntimeNode.md) and only differs in name and category. Connected to a video or audio pipeline, it still gives that pipeline's video or audio output.

## Example

You are exploring the look of a creature for a sequence and want a handful of concept frames.

1. Build a `Flux` pipeline with `FluxPipeline` and a downloaded model.
1. Connect the builder to **Generate Image (Diffusion Pipeline)**.
1. Write the creature description in `prompt` and keep the default 1024 by 1024 size.
1. Turn on `randomize_seed` and run the node several times. Each run gives a new frame in `output_image`.

If a run fails with an out-of-memory error, see [Memory and performance](../../guides/memory-and-performance.md).
