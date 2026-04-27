import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2  # type: ignore[reportMissingImports]
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.files.file import File


def download_video_to_temp_file(video_url_artifact: VideoUrlArtifact) -> Path:
    """Download a video from a VideoUrlArtifact to a temporary file.

    Args:
        video_url_artifact: The VideoUrlArtifact containing the video URL

    Returns:
        Path to the temporary file containing the downloaded video.
        The caller is responsible for cleaning up this file.

    Raises:
        ValueError: If video download fails with descriptive error message
    """
    url = video_url_artifact.value

    # Extract suffix from URL, defaulting to .mp4
    parsed_url = urlparse(url)
    url_path = Path(unquote(parsed_url.path))
    suffix = url_path.suffix or ".mp4"

    # TODO(#60): Use ProjectFileDestination with temp files situation
    fd, temp_path_str = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    temp_path = Path(temp_path_str)

    try:
        video_bytes = File(url).read_bytes()
        temp_path.write_bytes(video_bytes)
    except Exception as err:
        temp_path.unlink(missing_ok=True)
        details = f"Failed to download video at '{url}'.\nError: {err}"
        raise ValueError(details) from err

    return temp_path


def load_video_frames_from_url_artifact(video_url_artifact: VideoUrlArtifact) -> list:
    """Load a VideoUrlArtifact as a list of PIL frames.

    Resolves Griptape path macros (e.g. `{inputs}/foo.mp4`) and HTTP URLs via
    `File(...)`, then decodes frames with `diffusers.utils.load_video`.
    """
    import diffusers.utils  # type: ignore[reportMissingImports]

    temp_path = download_video_to_temp_file(video_url_artifact)
    try:
        return diffusers.utils.load_video(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


def get_video_fps(video_path: Path, default_fps: float = 30.0) -> float:
    """Get the FPS (frames per second) of a video file using OpenCV.

    Args:
        video_path: Path to the video file
        default_fps: Default FPS to return if unable to determine from video

    Returns:
        The video's FPS, or default_fps if unable to determine
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return default_fps

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return default_fps
        return fps
    finally:
        cap.release()
