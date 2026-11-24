---
description: Generates an audio clip from a text prompt with an AudioLDM pipeline, running on your own machine.
when_to_use: Use it to make temporary sound effects or ambience for an animatic or a review cut.
---

## When to use it

You want a short sound from a text description and don't need a licensed or recorded asset yet. Build an `AudioLDM` pipeline with a [Diffusion Pipeline Builder](../diffusion/DiffusionPipelineBuilderNode.md), connect its `pipeline` output here, and describe the sound in `prompt`. The clip is in `output_audio`.

The node is identical to [Generate Media](../diffusion/DiffusionPipelineRuntimeNode.md) and only differs in name and category. Audio pipelines have no `width` or `height`.

## Example

You are cutting an animatic and need placeholder ambience for a cavern shot.

1. Build an `AudioLDM` pipeline and pick a downloaded model.
1. Connect the builder to **Generate Audio (Diffusion Pipeline)**.
1. Set `prompt` to "water dripping in a large stone cave, distant echo" and `negative_prompt` to "music, speech".
1. Set `audio_length_in_s` to the length of the shot. It defaults to 5 seconds.
1. Run the node and drop `output_audio` into the cut.
