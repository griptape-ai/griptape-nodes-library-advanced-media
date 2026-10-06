import importlib
import sys
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

# The node imports the model stack at module level. Whatever isn't installed is
# stubbed so validation can be tested without torch, diffusers, transformers, or sam2.
MODEL_STACK_MODULES = (
    "cv2",
    "diffusers",
    "diffusers.pipelines",
    "diffusers.pipelines.pipeline_utils",
    "imageio",
    "sam2",
    "sam2.build_sam",
    "torch",
    "transformers",
)


@pytest.fixture(scope="module")
def detector_class() -> Iterator[type]:
    modules_before = set(sys.modules)
    with pytest.MonkeyPatch.context() as monkeypatch:
        for module_name in MODEL_STACK_MODULES:
            try:
                importlib.import_module(module_name)
            except ImportError:
                monkeypatch.setitem(sys.modules, module_name, MagicMock())
        module = importlib.import_module("dino_sam2_library.dino_sam_2_video_detector")
        yield module.DinoSam2VideoDetector
    # Drop modules imported against the stubs so later tests import them for real.
    for module_name in set(sys.modules) - modules_before:
        sys.modules.pop(module_name, None)


def test_validate_before_node_run_reports_missing_input_video(detector_class: type) -> None:
    node = detector_class(name="Video Mask")

    errors = node.validate_before_node_run()

    assert errors
    assert any("input_video" in str(error) for error in errors)


def test_validate_before_node_run_reports_empty_input_video(detector_class: type) -> None:
    node = detector_class(name="Video Mask")
    node.set_parameter_value("input_video", VideoUrlArtifact(""))

    errors = node.validate_before_node_run()

    assert errors
    assert any("input_video" in str(error) for error in errors)


def test_validate_before_node_run_accepts_input_video(detector_class: type) -> None:
    node = detector_class(name="Video Mask")
    node.set_parameter_value("input_video", VideoUrlArtifact("https://example.com/clip.mp4"))

    assert not node.validate_before_node_run()
