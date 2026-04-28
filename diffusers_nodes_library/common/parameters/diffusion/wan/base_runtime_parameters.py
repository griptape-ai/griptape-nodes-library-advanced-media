import logging
import tempfile
from pathlib import Path
from typing import Any

import diffusers  # type: ignore[reportMissingImports]
import PIL.Image
import torch  # type: ignore[reportMissingImports]
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.files.project_file import ProjectFileDestination

from diffusers_nodes_library.common.parameters.diffusion.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from utils.video_utils import load_video_frames_from_url_artifact

logger = logging.getLogger("diffusers_nodes_library")


class WanVideoPipelineRuntimeParametersBase(DiffusionPipelineRuntimeParameters):
    """Base class for WAN video pipeline runtime parameters.

    Provides common video output handling for all WAN video pipelines.
    """

    def __init__(self, node: BaseNode):
        super().__init__(node)

    def _video_artifact_to_pil_frames(self, video_artifact: VideoUrlArtifact | None) -> list[PIL.Image.Image]:
        """Convert a VideoUrlArtifact to a list of PIL Image frames."""
        if video_artifact is None:
            return []
        return load_video_frames_from_url_artifact(video_artifact)

    def publish_output_image_preview_placeholder(self) -> None:
        """Override to publish video placeholder instead of image placeholder."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            black_frame = PIL.Image.new("RGB", (320, 240), color="black")
            diffusers.utils.export_to_video([black_frame], str(temp_path), fps=1)
            dest = ProjectFileDestination.from_situation(filename="placeholder_video.mp4", situation="save_node_output")
            saved = dest.write_bytes(temp_path.read_bytes())
            self._node.publish_update_to_parameter("output_video", VideoUrlArtifact(saved.location))
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _process_pipeline_output(self, pipe: Any, callback_on_step_end: Any) -> None:
        """Override to handle video output instead of image output.

        Subclasses can override _get_pipeline_call_kwargs() to customize
        the kwargs passed to the pipeline call.
        """
        pipe_kwargs = self._get_pipeline_call_kwargs(pipe)

        output = pipe(
            **pipe_kwargs,
            output_type="pil",
            callback_on_step_end=callback_on_step_end,
        )
        frames = output.frames[0]

        # Export frames to video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            diffusers.utils.export_to_video(frames, str(temp_path), fps=16)
            self.publish_output_video(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _get_pipeline_call_kwargs(self, pipe: Any) -> dict:  # noqa: ARG002
        """Get kwargs to pass to the pipeline call.

        Override this method if you need to customize kwargs based on the pipe
        (e.g., for image-to-video pipelines that need to process the image).
        Default implementation just calls get_pipe_kwargs().
        """
        return self.get_pipe_kwargs()

    def latents_to_video_mp4(self, pipe: Any, latents: Any) -> Path:
        """Convert latents to video frames and export as MP4 file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file_obj:
            temp_file = Path(temp_file_obj.name)

        try:
            # Convert latents to video frames using VAE decode
            latents = latents.to(pipe.vae.dtype)

            # Apply latents normalization as per the WAN pipeline
            latents_mean = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean

            # Decode latents to video using VAE
            video = pipe.vae.decode(latents, return_dict=False)[0]
            frames = pipe.video_processor.postprocess_video(video, output_type="pil")[0]

            # Export frames to video
            diffusers.utils.export_to_video(frames, str(temp_file), fps=16)
        except Exception:
            # Clean up on error
            if temp_file.exists():
                temp_file.unlink()
            raise
        else:
            return temp_file

    def publish_output_video_preview_latents(self, pipe: Any, latents: Any) -> None:
        """Publish a preview video from latents during generation."""
        preview_video_path = None
        try:
            preview_video_path = self.latents_to_video_mp4(pipe, latents)
            dest = ProjectFileDestination.from_situation(filename="preview_video.mp4", situation="save_node_output")
            saved = dest.write_bytes(preview_video_path.read_bytes())
            self._node.publish_update_to_parameter("output_video", VideoUrlArtifact(saved.location))
        except Exception as e:
            logger.warning("Failed to generate video preview from latents: %s", e)
        finally:
            # Clean up temporary file
            if preview_video_path is not None and preview_video_path.exists():
                preview_video_path.unlink()

    def publish_output_video(self, video_path: Path) -> None:
        dest = ProjectFileDestination.from_situation(
            filename=f"output_video{video_path.suffix}", situation="save_node_output"
        )
        saved = dest.write_bytes(video_path.read_bytes())
        video_artifact = VideoUrlArtifact(saved.location)
        self._node.publish_update_to_parameter("output_video", video_artifact)
