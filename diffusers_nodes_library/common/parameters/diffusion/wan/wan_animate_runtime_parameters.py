import logging
import tempfile
from pathlib import Path
from typing import Any

import diffusers  # type: ignore[reportMissingImports]
import torch  # type: ignore[reportMissingImports]
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape.loaders import ImageLoader
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.files.project_file import ProjectFileDestination
from PIL import Image  # type: ignore[reportMissingImports]

from diffusers_nodes_library.common.parameters.diffusion.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from pillow_nodes_library.utils import (  # type: ignore[reportMissingImports]
    image_artifact_to_pil,
)

logger = logging.getLogger("diffusers_nodes_library")


class WanAnimatePipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    def __init__(self, node: BaseNode):
        super().__init__(node)

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Reference character image",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="pose_video",
                input_types=["VideoArtifact", "VideoUrlArtifact"],
                type="VideoUrlArtifact",
                tooltip="Video defining body pose/motion",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="face_video",
                input_types=["VideoArtifact", "VideoUrlArtifact"],
                type="VideoUrlArtifact",
                tooltip="Video defining facial expressions",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="background_video",
                input_types=["VideoArtifact", "VideoUrlArtifact"],
                type="VideoUrlArtifact",
                default_value=None,
                tooltip="Background video (optional)",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="mask_video",
                input_types=["VideoArtifact", "VideoUrlArtifact"],
                type="VideoUrlArtifact",
                default_value=None,
                tooltip="Mask video (optional)",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="Text prompt for video generation",
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
                name="mode",
                default_value="animate",
                type="str",
                tooltip="Animation mode: 'animate' or 'replace'",
                ui_options={"options": ["animate", "replace"]},
            )
        )
        self._node.add_parameter(
            Parameter(
                name="num_frames",
                default_value=81,
                type="int",
                tooltip="Number of frames to generate",
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
        self._node.remove_parameter_element_by_name("image")
        self._node.remove_parameter_element_by_name("pose_video")
        self._node.remove_parameter_element_by_name("face_video")
        self._node.remove_parameter_element_by_name("background_video")
        self._node.remove_parameter_element_by_name("mask_video")
        self._node.remove_parameter_element_by_name("prompt")
        self._node.remove_parameter_element_by_name("negative_prompt")
        self._node.remove_parameter_element_by_name("mode")
        self._node.remove_parameter_element_by_name("num_frames")
        self._node.remove_parameter_element_by_name("guidance_scale")

    def remove_output_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("output_video")

    def _get_pipe_kwargs(self) -> dict:
        return {
            "image": self.get_input_image_pil(),
            "pose_video": self._video_artifact_to_pil_frames(self.get_pose_video()),
            "face_video": self._video_artifact_to_pil_frames(self.get_face_video()),
            "background_video": self._get_optional_video_frames(self.get_background_video()),
            "mask_video": self._get_optional_video_frames(self.get_mask_video()),
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "mode": self._node.get_parameter_value("mode"),
            "num_frames": self.get_num_frames(),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
            "output_type": "pil",
        }

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []

        # Validate required inputs
        if self.get_input_image() is None:
            errors.append(ValueError("Reference character image is required"))
        if self.get_pose_video() is None:
            errors.append(ValueError("Pose video is required"))
        if self.get_face_video() is None:
            errors.append(ValueError("Face video is required"))

        # Validate dimensions are divisible by 16
        width = self.get_width()
        height = self.get_height()
        if width % 16 != 0:
            errors.append(ValueError(f"Width ({width}) must be divisible by 16"))
        if height % 16 != 0:
            errors.append(ValueError(f"Height ({height}) must be divisible by 16"))

        # Validate mode
        mode = self._node.get_parameter_value("mode")
        if mode not in ("animate", "replace"):
            errors.append(ValueError(f"Mode must be 'animate' or 'replace', got '{mode}'"))

        return errors or None

    def get_num_frames(self) -> int:
        return int(self._node.get_parameter_value("num_frames"))

    def get_input_image(self) -> ImageUrlArtifact | None:
        return self._node.get_parameter_value("image")

    def get_input_image_pil(self) -> Image.Image:
        input_image_artifact = self.get_input_image()
        if isinstance(input_image_artifact, ImageUrlArtifact):
            input_image_artifact = ImageLoader().parse(input_image_artifact.to_bytes())
        input_image_pil = image_artifact_to_pil(input_image_artifact)
        return input_image_pil.convert("RGB")

    def get_pose_video(self) -> VideoUrlArtifact | None:
        return self._node.get_parameter_value("pose_video")

    def get_face_video(self) -> VideoUrlArtifact | None:
        return self._node.get_parameter_value("face_video")

    def get_background_video(self) -> VideoUrlArtifact | None:
        return self._node.get_parameter_value("background_video")

    def get_mask_video(self) -> VideoUrlArtifact | None:
        return self._node.get_parameter_value("mask_video")

    def _video_artifact_to_pil_frames(self, video_artifact: VideoUrlArtifact) -> list[Image.Image]:
        """Convert a VideoUrlArtifact to a list of PIL Image frames."""
        if video_artifact is None:
            return []
        return diffusers.utils.load_video(video_artifact.value)

    def _get_optional_video_frames(self, video_artifact: VideoUrlArtifact | None) -> list[Image.Image] | None:
        """Convert an optional video artifact to PIL frames, returning None if not provided."""
        if video_artifact is None:
            return None
        return self._video_artifact_to_pil_frames(video_artifact)

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
        """Publish the final output video."""
        dest = ProjectFileDestination.from_situation(
            filename=f"output_video{video_path.suffix}", situation="save_node_output"
        )
        saved = dest.write_bytes(video_path.read_bytes())
        self._node.parameter_output_values["output_video"] = VideoUrlArtifact(saved.location)
