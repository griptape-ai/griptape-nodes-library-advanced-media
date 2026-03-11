import logging
import tempfile
from pathlib import Path

import diffusers  # type: ignore[reportMissingImports]
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from PIL import Image  # type: ignore[reportMissingImports]

from pillow_nodes_library.utils import (  # type: ignore[reportMissingImports]
    image_artifact_to_pil,
)
from utils.image_utils import load_image_from_url_artifact

logger = logging.getLogger("diffusers_nodes_library")


class RandomFramesToVideoWanVaceAux(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            ParameterList(
                name="input_images",
                default_value=[],
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Images to insert at random frame positions",
            )
        )
        self.add_parameter(
            Parameter(
                name="frame_indices",
                default_value="0,40,80",
                input_types=["str"],
                type="str",
                tooltip="Comma-separated frame indices where images should be placed (e.g., '0,40,80')",
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
                tooltip="Generated video with images at specified frame indices",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_mask",
                output_type="VideoUrlArtifact",
                tooltip="Generated mask video for random frame conditioning",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._output_video_file = ProjectFileParameter(
            node=self, name="output_video_file", default_filename="output_video.mp4",
        )
        self._output_video_file.add_parameter()

        self._output_mask_file = ProjectFileParameter(
            node=self, name="output_mask_file", default_filename="output_mask.mp4",
        )
        self._output_mask_file.add_parameter()

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []

        input_images = self.get_parameter_value("input_images")
        if not input_images:
            errors.append(ValueError("At least one input image is required"))

        frame_indices_str = self.get_parameter_value("frame_indices")
        try:
            frame_indices = [int(idx.strip()) for idx in frame_indices_str.split(",")]
            num_frames = int(self.get_parameter_value("num_frames"))

            if len(frame_indices) != len(input_images):
                errors.append(
                    ValueError(
                        f"Number of frame indices ({len(frame_indices)}) must match number of input images ({len(input_images)})"
                    )
                )

            errors.extend(
                ValueError(f"Frame index {idx} is out of range (0-{num_frames - 1})")
                for idx in frame_indices
                if idx < 0 or idx >= num_frames
            )
        except ValueError as e:
            errors.append(ValueError(f"Invalid frame indices format: {e}"))

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
        input_images = self.get_parameter_value("input_images")
        frame_indices_str = self.get_parameter_value("frame_indices")
        num_frames = int(self.get_parameter_value("num_frames"))
        width = int(self.get_parameter_value("width"))
        height = int(self.get_parameter_value("height"))

        # Parse frame indices
        frame_indices = [int(idx.strip()) for idx in frame_indices_str.split(",")]

        # Convert input images to PIL and resize, handling ImageUrlArtifact
        pil_images = []
        for input_img in input_images:
            img = input_img
            if isinstance(img, ImageUrlArtifact):
                img = load_image_from_url_artifact(img)
            pil_img = image_artifact_to_pil(img)
            pil_img = pil_img.resize((width, height), Image.Resampling.LANCZOS)
            # Ensure RGB mode
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

        # Create video frames - black frames with specified images at indices
        frames = []
        black_frame = Image.new("RGB", (width, height), (0, 0, 0))

        # Create mapping of frame index to image
        frame_to_image = dict(zip(frame_indices, pil_images, strict=False))

        for i in range(num_frames):
            if i in frame_to_image:
                frames.append(frame_to_image[i])
            else:
                frames.append(black_frame)

        # Create mask frames - black for specified frames (preserve), white for others (generate)
        mask_frames = []
        white_frame = Image.new("RGB", (width, height), (255, 255, 255))  # Generate frames
        black_frame = Image.new("RGB", (width, height), (0, 0, 0))  # Preserve frames

        for i in range(num_frames):
            if i in frame_indices:
                mask_frames.append(black_frame)  # Preserve specified frames
            else:
                mask_frames.append(white_frame)  # Generate other frames

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
