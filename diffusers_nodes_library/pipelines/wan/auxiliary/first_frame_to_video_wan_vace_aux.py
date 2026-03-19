import logging
import tempfile
from pathlib import Path

import diffusers  # type: ignore[reportMissingImports]
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from PIL import Image  # type: ignore[reportMissingImports]

from pillow_nodes_library.utils import (  # type: ignore[reportMissingImports]
    image_artifact_to_pil,
)
from utils.image_utils import load_image_from_url_artifact

logger = logging.getLogger("diffusers_nodes_library")


class FirstFrameToVideoWanVaceAux(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="input_image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Input image to use as the first frame",
            )
        )
        self.add_parameter(
            Parameter(
                name="num_frames",
                default_value=81,
                input_types=["int"],
                type="int",
                tooltip="Number of frames to generate (must match WAN VACE pipeline)",
                ui_options={"slider": {"min_val": 9, "max_val": 161}, "step": 8},
            )
        )
        self.add_parameter(
            Parameter(
                name="width",
                default_value=832,
                input_types=["int"],
                type="int",
                tooltip="Output video width (must be divisible by 16)",
            )
        )
        self.add_parameter(
            Parameter(
                name="height",
                default_value=480,
                input_types=["int"],
                type="int",
                tooltip="Output video height (must be divisible by 16)",
            )
        )
        self.add_parameter(
            Parameter(
                name="output_video",
                output_type="VideoUrlArtifact",
                tooltip="Generated video with input image as first frame",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_mask",
                output_type="VideoUrlArtifact",
                tooltip="Generated mask video for first frame conditioning",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._output_video_file = ProjectFileParameter(
            node=self,
            name="output_video_file",
            default_filename="output_video.mp4",
        )
        self._output_video_file.add_parameter()

        self._output_mask_file = ProjectFileParameter(
            node=self,
            name="output_mask_file",
            default_filename="output_mask.mp4",
        )
        self._output_mask_file.add_parameter()

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []

        input_image = self.get_parameter_value("input_image")
        if input_image is None:
            errors.append(ValueError("Input image is required"))

        width = int(self.get_parameter_value("width"))
        height = int(self.get_parameter_value("height"))

        if width % 16 != 0:
            errors.append(ValueError(f"Width ({width}) must be divisible by 16"))
        if height % 16 != 0:
            errors.append(ValueError(f"Height ({height}) must be divisible by 16"))

        return errors or None

    def process(self) -> AsyncResult | None:
        yield lambda: self._process()

    def _process(self) -> AsyncResult | None:
        input_image = self.get_parameter_value("input_image")
        num_frames = int(self.get_parameter_value("num_frames"))
        width = int(self.get_parameter_value("width"))
        height = int(self.get_parameter_value("height"))

        # Convert input image to PIL, handling ImageUrlArtifact
        if isinstance(input_image, ImageUrlArtifact):
            input_image = load_image_from_url_artifact(input_image)
        pil_image = image_artifact_to_pil(input_image)

        # Resize image to target dimensions and ensure RGB mode
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Create video frames - first frame is the input image, rest are black
        frames = []
        frames.append(pil_image)  # First frame is the input image

        # Create black frames for the rest
        black_frame = Image.new("RGB", (width, height), (0, 0, 0))
        frames.extend([black_frame] * (num_frames - 1))

        # Create mask frames - black for first frame (preserve), white for rest (generate)
        mask_frames = []
        white_frame = Image.new("RGB", (width, height), (255, 255, 255))  # Generate other frames
        black_frame = Image.new("RGB", (width, height), (0, 0, 0))  # Preserve first frame

        mask_frames.append(black_frame)  # First frame preserved
        mask_frames.extend([white_frame] * (num_frames - 1))  # Other frames to be generated

        # Export videos
        video_path = self._export_frames_to_video(frames)
        mask_path = self._export_frames_to_video(mask_frames)

        try:
            # Save video
            video_dest = self._output_video_file.build_file()
            video_saved = video_dest.write_bytes(video_path.read_bytes())
            self.set_parameter_value("output_video", VideoUrlArtifact(video_saved.location))
            self.parameter_output_values["output_video"] = VideoUrlArtifact(video_saved.location)

            # Save mask
            mask_dest = self._output_mask_file.build_file()
            mask_saved = mask_dest.write_bytes(mask_path.read_bytes())
            self.set_parameter_value("output_mask", VideoUrlArtifact(mask_saved.location))
            self.parameter_output_values["output_mask"] = VideoUrlArtifact(mask_saved.location)

        finally:
            # Clean up temporary files
            if video_path.exists():
                video_path.unlink()
            if mask_path.exists():
                mask_path.unlink()

    def _export_frames_to_video(self, frames: list[Image.Image]) -> Path:
        """Export PIL frames to MP4 video file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            diffusers.utils.export_to_video(frames, str(temp_path), fps=16)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
        else:
            return temp_path
