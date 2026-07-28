import logging
import tempfile
from pathlib import Path
from typing import Any

import diffusers  # type: ignore[reportMissingImports]
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.files.project_file import ProjectFileDestination

from diffusers_nodes_library.common.parameters.diffusion.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)

logger = logging.getLogger("diffusers_nodes_library")

# Frame rate recommended by the Allegro pipeline documentation.
ALLEGRO_FPS = 15


class AllegroPipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    """Runtime parameters for the Allegro text-to-video pipeline.

    Allegro is a video model with a 3D VAE, so it produces a video output and
    decodes latents differently from the image pipelines the base class targets.
    """

    def __init__(self, node: BaseNode):
        super().__init__(node)

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="Prompt",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="negative_prompt",
                default_value="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                type="str",
                tooltip="Negative prompt (optional)",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="num_frames",
                default_value=40,
                type="int",
                tooltip="Number of frames to generate",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="guidance_scale",
                default_value=7.5,
                type="float",
                tooltip="CFG guidance scale",
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

    def remove_output_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("output_video")

    def publish_output_image_preview_placeholder(self) -> None:
        # Video pipelines don't use image placeholders.
        pass

    def _get_pipe_kwargs(self) -> dict:
        return {
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "num_frames": self._node.get_parameter_value("num_frames"),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
        }

    def _process_pipeline_output(self, pipe: Any, callback_on_step_end: Any) -> None:
        """Process Allegro video pipeline output.

        Allegro returns ``.frames`` (not ``.images``), so the base image
        implementation cannot be used.
        """
        output = pipe(
            **self.get_pipe_kwargs(),
            output_type="pil",
            callback_on_step_end=callback_on_step_end,
        )
        frames = output.frames[0]

        # Export frames to video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file_obj:
            temp_file = Path(temp_file_obj.name)

        try:
            diffusers.utils.export_to_video(frames, str(temp_file), fps=ALLEGRO_FPS)
            self.publish_output_video(temp_file)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def latents_to_video_mp4(self, pipe: Any, latents: Any) -> Path:
        """Convert latents to video frames and export as an MP4 file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file_obj:
            temp_file = Path(temp_file_obj.name)

        try:
            # Decode latents to video using the Allegro 3D VAE.
            latents = latents.to(pipe.vae.dtype)
            video = pipe.decode_latents(latents)
            video = pipe.video_processor.postprocess_video(video=video, output_type="pil")[0]

            # Export frames to video
            diffusers.utils.export_to_video(video, str(temp_file), fps=ALLEGRO_FPS)
        except Exception:
            # Clean up on error
            if temp_file.exists():
                temp_file.unlink()
            raise
        else:
            return temp_file

    def publish_output_image_preview_latents(self, pipe: Any, latents: Any) -> None:
        """Publish a preview video from latents during generation.

        Overrides the base image implementation, which relies on
        ``pipe._unpack_latents`` and other Flux-style image-pipeline methods
        that Allegro does not provide.
        """
        preview_video_path = None
        try:
            preview_video_path = self.latents_to_video_mp4(pipe, latents)
            dest = ProjectFileDestination.from_situation(filename="preview_video.mp4", situation="save_node_output")
            saved = dest.write_bytes(preview_video_path.read_bytes())
            self._node.publish_update_to_parameter("output_video", VideoUrlArtifact(saved.location))
        except Exception as e:
            logger.warning("Failed to generate video preview from latents: %s", e)
        finally:
            if preview_video_path is not None and preview_video_path.exists():
                preview_video_path.unlink()

    def publish_output_video(self, video_path: Path) -> None:
        dest = ProjectFileDestination.from_situation(
            filename=f"output_video{video_path.suffix}", situation="save_node_output"
        )
        saved = dest.write_bytes(video_path.read_bytes())
        video_artifact = VideoUrlArtifact(saved.location)
        self._node.publish_update_to_parameter("output_video", video_artifact)
        self._node.parameter_output_values["output_video"] = video_artifact
