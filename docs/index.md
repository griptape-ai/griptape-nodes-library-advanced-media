---
title: "Advanced Media Library"
description: Nodes that run diffusion models and image and video analysis models on your own machine, for generation, depth, edges, pose, masks and LoRAs.
when_to_use: Use it for local image, video and audio generation and for pre-processing passes such as depth, edges, pose and masks.
---

The Advanced Media Library runs its models **locally**, on your own GPU, with no cloud service in the loop. It covers three kinds of work.

- **Generation.** Diffusion pipelines for images, video and audio, built once and reused across runs.
- **Pre-processing.** Depth, edge, pose and mask passes for images and video, such as the passes you feed to a ControlNet or a comp.
- **LoRAs.** Nodes that apply your own LoRA files or ready-made Flux and WAN LoRAs to a diffusion pipeline.

The library also has a node that translates text and a template workflow, `qwen_edit_2509_camera_control`, which changes the camera angle of an image with Qwen-Image-Edit-2509.

- **Requirements**: an NVIDIA GPU (CUDA) or an Apple Silicon Mac (MPS). Without one of them the nodes refuse to run, though you can still edit workflows. A Hugging Face account and access token are needed to download models, see
    [Hugging Face Models](https://docs.griptapenodes.com/en/stable/guides/integrations/hugging_face/)

## Installation

See the [Libraries guide](https://docs.griptapenodes.com/en/stable/guides/libraries/) for install, update and troubleshooting help.

Models are not bundled. Download the ones you need to your Hugging Face cache, and the nodes list them once they are there.

## Nodes

### Diffusion pipelines

The builder loads a model once. The Generate nodes use it. See [Memory and performance](guides/memory-and-performance.md) for the memory settings and for what to do when a run is slow or runs out of memory.

- [Diffusion Pipeline Builder](nodes/diffusion/DiffusionPipelineBuilderNode.md)
- [Generate Media (Diffusion Pipeline)](nodes/diffusion/DiffusionPipelineRuntimeNode.md)
- [Generate Image (Diffusion Pipeline)](nodes/image/GenerateImageNode.md)
- [Generate Video (Diffusion Pipeline)](nodes/video/GenerateVideoNode.md)
- [Generate Audio (Diffusion Pipeline)](nodes/audio/GenerateAudioNode.md)
- [Clear Pipeline Cache](nodes/utils/ClearPipelineCacheNode.md)

### LoRA

- [Load LoRA](nodes/lora/LoadLora.md)
- Flux: Flux LoRA: Lumatales-FL, Flux LoRA: Micro Landscape on Mobile Phone, Flux LoRA: Miniature World, Flux ICEdit LoRA: RiverZ Normal
- WAN: Kijai CausVid 1.3B, Kijai CausVid 14B

### Image

- Depth: Depth Anything V2 Image
- Edges: Detect Edges with OpenCV, Morphology, Detect Edges with Anyline
- Pose: OpenPose Image Detection
- Segmentation: Image Mask via G-DINO + SAM2

### Video

- Depth: Depth Anything V2 Video
- Pose: OpenPose Video Detection
- Segmentation: Video Mask via G-DINO + SAM2
- WAN auxiliary inputs: WAN First Frame Aux, WAN First+Last Frame Aux, WAN Last Frame Aux, WAN Random Frames Aux, WAN Static Mask Aux

### Text

- Translate Text (Google TranslateGemma)

## Using it with the Diffusers library

The Diffusers library also runs Diffusers pipelines, with nodes for building them in stages. You can load both libraries.

- Both read models from the same Hugging Face cache, so a model you download once is available to both.
- Each keeps its own pipeline in memory. If you use both in one session, two copies of the same weights can sit in GPU memory. Use one library per session when memory is tight.
- Both pin versions of `diffusers`, `transformers` and `torch`. Compare the pins in each library's `pyproject.toml` before installing both.
- Both have nodes called Load LoRA and Clear Pipeline Cache. Each Clear Pipeline Cache only releases its own library's pipelines.
