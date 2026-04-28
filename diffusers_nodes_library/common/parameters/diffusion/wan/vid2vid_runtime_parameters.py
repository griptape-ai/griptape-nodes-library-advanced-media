import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode

from diffusers_nodes_library.common.parameters.diffusion.wan.base_runtime_parameters import (
    WanVideoPipelineRuntimeParametersBase,
)

logger = logging.getLogger("diffusers_nodes_library")


class WanVideoToVideoPipelineRuntimeParameters(WanVideoPipelineRuntimeParametersBase):
    def __init__(self, node: BaseNode):
        super().__init__(node)

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="input_video",
                input_types=["VideoArtifact", "VideoUrlArtifact"],
                type="VideoUrlArtifact",
                tooltip="Input video for video-to-video generation",
            )
        )
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
                name="guidance_scale",
                default_value=5.0,
                type="float",
                tooltip="CFG guidance scale (higher = more prompt adherence)",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="strength",
                default_value=0.8,
                type="float",
                tooltip="Higher strength leads to more differences between original image and generated video.",
                ui_options={"slider": {"min_val": 0.0, "max_val": 1.0}, "step": 0.01},
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
        self._node.remove_parameter_element_by_name("input_video")
        self._node.remove_parameter_element_by_name("prompt")
        self._node.remove_parameter_element_by_name("negative_prompt")
        self._node.remove_parameter_element_by_name("guidance_scale")
        self._node.remove_parameter_element_by_name("strength")

    def remove_output_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("output_video")

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []

        # Validate input video is provided
        input_video = self.get_input_video()
        if input_video is None:
            errors.append(ValueError("Input video is required for video-to-video generation"))

        # Validate dimensions are divisible by 16
        width = self.get_width()
        height = self.get_height()
        if width % 16 != 0:
            errors.append(ValueError(f"Width ({width}) must be divisible by 16"))
        if height % 16 != 0:
            errors.append(ValueError(f"Height ({height}) must be divisible by 16"))

        return errors or None

    def _get_pipe_kwargs(self) -> dict:
        return {
            "video": self._video_artifact_to_pil_frames(self.get_input_video()),
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
            "strength": self._node.get_parameter_value("strength"),
        }

    def get_input_video(self) -> Any:
        return self._node.get_parameter_value("input_video")
