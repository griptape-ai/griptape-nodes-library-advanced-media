import logging
from typing import Any

from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from griptape.artifacts import ImageUrlArtifact
from PIL.Image import Image
from pillow_nodes_library.utils import (  # type: ignore[reportMissingImports]
    image_artifact_to_pil,
)
from utils.image_utils import load_image_from_url_artifact

from diffusers_nodes_library.common.parameters.diffusion.qwen.common import qwen_latents_to_image_pil
from diffusers_nodes_library.common.parameters.diffusion.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode

logger = logging.getLogger("diffusers_nodes_library")


class QwenInpaintPipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    """Runtime parameters for QwenImageInpaintPipeline and QwenImageEditInpaintPipeline."""

    def __init__(self, node: BaseNode):
        super().__init__(node)

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Source image to be inpainted",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="mask_image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Inpainting mask (white areas will be repainted, black areas preserved)",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="The prompt to guide inpainting",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="negative_prompt",
                default_value="",
                type="str",
                tooltip="The prompt not to guide inpainting",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="true_cfg_scale",
                default_value=1.0,
                type="float",
                tooltip="True classifier-free guidance is enabled when true_cfg_scale > 1 and negative_prompt is provided",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="strength",
                default_value=0.6,
                type="float",
                tooltip="Extent of transformation (0=minimal, 1=maximum)",
                ui_options={"slider": {"min_val": 0.0, "max_val": 1.0}, "step": 0.01},
            )
        )
        self._node.add_parameter(
            Parameter(
                name="padding_mask_crop",
                default_value=None,
                type="int",
                tooltip="Margin size for cropping to masked area (advanced parameter)",
                ui_options={"hidden": True},
            )
        )

    def _remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("image")
        self._node.remove_parameter_element_by_name("mask_image")
        self._node.remove_parameter_element_by_name("prompt")
        self._node.remove_parameter_element_by_name("negative_prompt")
        self._node.remove_parameter_element_by_name("true_cfg_scale")
        self._node.remove_parameter_element_by_name("strength")
        self._node.remove_parameter_element_by_name("padding_mask_crop")

    def get_image_pil(self) -> Image:
        """Get the source image as a PIL Image in RGB mode."""
        input_image_artifact = self._node.get_parameter_value("image")
        if isinstance(input_image_artifact, ImageUrlArtifact):
            input_image_artifact = load_image_from_url_artifact(input_image_artifact)
        input_image_pil = image_artifact_to_pil(input_image_artifact)
        return input_image_pil.convert("RGB")

    def get_mask_image_pil(self) -> Image:
        """Get the mask image as a PIL Image in grayscale (L) mode."""
        mask_image_artifact = self._node.get_parameter_value("mask_image")
        if isinstance(mask_image_artifact, ImageUrlArtifact):
            mask_image_artifact = load_image_from_url_artifact(mask_image_artifact)
        mask_image_pil = image_artifact_to_pil(mask_image_artifact)
        return mask_image_pil.convert("L")

    def _get_pipe_kwargs(self) -> dict:
        """Assemble all parameters for the pipeline call."""
        kwargs = {
            "image": self.get_image_pil(),
            "mask_image": self.get_mask_image_pil(),
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "true_cfg_scale": self._node.get_parameter_value("true_cfg_scale"),
            "strength": self._node.get_parameter_value("strength"),
        }

        # Only include padding_mask_crop if it's not None
        padding_mask_crop = self._node.get_parameter_value("padding_mask_crop")
        if padding_mask_crop is not None:
            kwargs["padding_mask_crop"] = padding_mask_crop

        return kwargs

    def latents_to_image_pil(self, pipe: DiffusionPipeline, latents: Any) -> Image:
        """Convert latents to PIL image using Qwen-specific logic."""
        return qwen_latents_to_image_pil(pipe, latents, self.get_height(), self.get_width())
