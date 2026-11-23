---
description: Unloads every diffusion pipeline this library holds in memory and frees the GPU memory they used.
when_to_use: Use it when you need the GPU back for other work, or to force the next generation to load its model from scratch.
---

## When to use it

The library keeps the last pipeline it built in memory so repeated generations are fast. That model keeps using GPU memory until something replaces it. Run this node when another node or application needs that memory.

The node reports how many pipelines it removed in its status result, for example "Cleared 1 pipeline(s) from cache." It runs even when nothing is loaded.

After a clear, a [Diffusion Pipeline Builder](../diffusion/DiffusionPipelineBuilderNode.md) runs again the next time it is needed, and a [Generate](../diffusion/DiffusionPipelineRuntimeNode.md) node loads its pipeline through the connected builder if it finds it missing.

This node only affects the Advanced Media Library. The Diffusers library keeps a separate cache with its own Clear Pipeline Cache node.

## Example

A workflow generates concept frames with Flux, and a render job on the same GPU starts when it finishes. Put **Clear Pipeline Cache** after the last Generate node so the Flux model's GPU memory is free for the render.
