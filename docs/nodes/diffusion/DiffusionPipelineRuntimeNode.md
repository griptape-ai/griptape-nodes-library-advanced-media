---
description: Runs a pipeline from a Diffusion Pipeline Builder and produces an image, video or audio clip, with inputs that match the pipeline.
when_to_use: Use it to generate from any builder pipeline, or when a workflow needs one generate node that works for image, video and audio pipelines alike.
---

## When to use it

You have a [Diffusion Pipeline Builder](DiffusionPipelineBuilderNode.md) and want to generate with it. Connect the builder's `pipeline` output to this node's `pipeline` input.

The node has no generation inputs until a pipeline is connected. Then it adds the inputs and outputs that pipeline needs. A WAN image-to-video pipeline adds `input_image`, `prompt` and `num_frames` and gives `output_video`. A Marigold depth pipeline adds `image` and gives `output_image`. If you disconnect the pipeline, those inputs and outputs are removed. If you swap in a different pipeline, connections to inputs that exist on both are kept.

[Generate Image](../image/GenerateImageNode.md), [Generate Video](../video/GenerateVideoNode.md) and [Generate Audio](../audio/GenerateAudioNode.md) are the same node under different names and categories, so they behave identically.

The node needs the pipeline to be loaded. If it isn't, for example after [Clear Pipeline Cache](../utils/ClearPipelineCacheNode.md), the node loads it again through the connected builder before generating.

## Example

You have a Qwen image-edit pipeline built and want a set of camera-angle variations of a concept frame for a layout review.

1. Connect the builder's `pipeline` to this node.
1. Connect the concept frame to the new image input and write the camera move in `prompt`.
1. Turn on `randomize_seed` to get a different variation on each run, or leave it off and change `seed` by hand.
1. Run the node. The result is in `output_image`.

## Parameters

### pipeline

Accepts the output of a Diffusion Pipeline Builder. Setting it decides which other parameters the node shows.

### width and height

Most pipelines add them, including image-to-image, edit, inpaint and image-to-video types. Marigold, DepthCrafter, Qwen layered and the Flux and Qwen upscale types take the size from their input and have neither. Audio pipelines have neither. The default is 1024 for each on most image pipelines. WAN image to video rejects a width or height that isn't a multiple of 16.

### seed and randomize_seed

Most pipelines add them. DepthCrafter doesn't. `seed` defaults to 42. With `randomize_seed` on, the node writes a new random value into `seed` on every run and the field can't be edited.

### output_image, output_video, output_audio

Which of these appears depends on the pipeline. The [builder's provider table](DiffusionPipelineBuilderNode.md#provider) lists what each provider produces.

### progress and logs

`progress` fills as denoising steps complete. `logs` records each step with its time. Cancel stops the run after the current step.
