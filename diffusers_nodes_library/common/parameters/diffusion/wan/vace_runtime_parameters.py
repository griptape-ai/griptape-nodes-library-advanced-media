import logging

import diffusers  # type: ignore[reportMissingImports]
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from PIL import Image  # type: ignore[reportMissingImports]

from diffusers_nodes_library.common.parameters.diffusion.wan.base_runtime_parameters import (
    WanVideoPipelineRuntimeParametersBase,
)
from pillow_nodes_library.utils import (  # type: ignore[reportMissingImports]
    image_artifact_to_pil,
)
from utils.image_utils import load_image_from_url_artifact

logger = logging.getLogger("diffusers_nodes_library")


class WanVacePipelineRuntimeParameters(WanVideoPipelineRuntimeParametersBase):
    def __init__(self, node: BaseNode):
        super().__init__(node)

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="Prompt for video generation",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="negative_prompt",
                default_value="",
                type="str",
                tooltip="Negative prompt (optional)",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="num_frames",
                default_value=81,
                type="int",
                tooltip="Number of frames to generate (model-specific)",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="guidance_scale",
                default_value=5.0,
                type="float",
                tooltip="CFG guidance scale (higher = more prompt adherence)",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="input_video",
                default_value=None,
                type="VideoUrlArtifact",
                tooltip="Input video for video-to-video generation (optional)",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="mask",
                default_value=None,
                type="VideoUrlArtifact",
                tooltip="Mask video for video-to-video generation (required when input_video is provided)",
            )
        )
        self._node.add_parameter(
            ParameterList(
                name="reference_frames",
                default_value=[],
                type="ImageArtifact",
                tooltip="Reference frames to guide video generation (optional)",
            )
        )

    def add_output_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="output_video",
                output_type="VideoUrlArtifact",
                tooltip="Generated video",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def _remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("prompt")
        self._node.remove_parameter_element_by_name("negative_prompt")
        self._node.remove_parameter_element_by_name("num_frames")
        self._node.remove_parameter_element_by_name("guidance_scale")
        self._node.remove_parameter_element_by_name("input_video")
        self._node.remove_parameter_element_by_name("mask")
        self._node.remove_parameter_element_by_name("reference_frames")

    def remove_output_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("output_video")

    def _get_pipe_kwargs(self) -> dict:
        return {
            "video": self.get_input_video_pil_frames(),
            "mask": self.get_mask_pil_frames(),
            "reference_images": self.get_reference_frames_pil(),
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "num_frames": self.get_num_frames(),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
        }

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []

        # Validate video and mask are provided together or neither are provided
        input_video = self.get_input_video()
        mask = self.get_mask()
        prompt = self._node.get_parameter_value("prompt")

        if (input_video is None) != (mask is None):
            if input_video is None:
                errors.append(
                    ValueError(
                        "Mask is provided but input_video is missing. Both video and mask are required together."
                    )
                )
            else:
                errors.append(
                    ValueError(
                        "Input video is provided but mask is missing. Both video and mask are required together."
                    )
                )

        # Ensure there's some input for generation
        if input_video is None and not prompt.strip():
            errors.append(ValueError("Must provide either a prompt or both input_video and mask for video generation"))

        # Validate dimensions are divisible by 16
        width = self.get_width()
        height = self.get_height()
        if width % 16 != 0:
            errors.append(ValueError(f"Width ({width}) must be divisible by 16"))
        if height % 16 != 0:
            errors.append(ValueError(f"Height ({height}) must be divisible by 16"))

        return errors or None

    def get_num_frames(self) -> int:
        return int(self._node.get_parameter_value("num_frames"))

    def get_input_video(self) -> VideoUrlArtifact | None:
        return self._node.get_parameter_value("input_video")

    def get_mask(self) -> VideoUrlArtifact | None:
        return self._node.get_parameter_value("mask")

    def get_reference_frames(self) -> list:
        return self._node.get_parameter_value("reference_frames") or []

    def get_input_video_pil_frames(self) -> list[Image.Image] | None:
        input_video = self.get_input_video()
        if input_video is None:
            return None
        return self._video_artifact_to_pil_frames(input_video)

    def get_mask_pil_frames(self) -> list[Image.Image] | None:
        mask = self.get_mask()
        if mask is None:
            return None
        return [pil_frame.convert("L") for pil_frame in self._video_artifact_to_pil_frames(mask)]

    def get_reference_frames_pil(self) -> list[Image.Image] | None:
        """Get reference frames as a list of PIL Images."""
        reference_frames = self.get_reference_frames()
        if not reference_frames:
            return None

        pil_images = []
        for frame_artifact in reference_frames:
            image_artifact = frame_artifact
            if isinstance(image_artifact, ImageUrlArtifact):
                image_artifact = load_image_from_url_artifact(image_artifact)
            pil_image = image_artifact_to_pil(image_artifact)
            pil_image = pil_image.convert("RGB")
            pil_images.append(pil_image)

        return pil_images if pil_images else None

    def _video_artifact_to_pil_frames(self, video_artifact: VideoUrlArtifact) -> list[Image.Image]:
        """Convert a VideoUrlArtifact to a list of PIL Image frames."""
        if video_artifact is None:
            return []

        # Use diffusers loading utilities to convert video URL to frames
        return diffusers.utils.load_video(video_artifact.value)
