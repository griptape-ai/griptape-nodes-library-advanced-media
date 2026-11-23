---
description: Generates a video clip from a Diffusion Pipeline Builder pipeline, such as WAN, LTX-2 or Allegro, running on your own machine.
when_to_use: Use it to animate a still, generate a short shot or produce a depth video with a locally built diffusion pipeline.
---

## When to use it

You want a video clip from a model that runs on your own GPU. Build the model once with a [Diffusion Pipeline Builder](../diffusion/DiffusionPipelineBuilderNode.md) using the `WAN`, `LTX-2`, `Allegro` or `DepthCrafter` provider, then connect its `pipeline` output here. The node shows the inputs that pipeline type needs, such as `input_image` for WAN image to video, and gives the clip in `output_video`.

The node is identical to [Generate Media](../diffusion/DiffusionPipelineRuntimeNode.md) and only differs in name and category.

Video pipelines are the heaviest the library runs. [Memory and performance](../../guides/memory-and-performance.md) covers the settings that make them fit on a smaller GPU.

## Example

You have a concept frame for a shot and want to see how it could move before anyone blocks it.

1. Build a `WAN` pipeline with `WanImageToVideoPipeline`.
1. Connect the builder to **Generate Video (Diffusion Pipeline)**.
1. Connect the concept frame to `input_image` and describe the camera and action in `prompt`.
1. Leave `num_frames` at its default of 81 for a first test and run the node.
1. Preview `output_video`.

WAN image to video needs a `width` and `height` that are multiples of 16.
