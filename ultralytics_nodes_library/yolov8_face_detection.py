import logging
from typing import Any

from griptape_nodes.exe_types.core_types import (
    DeprecationMessage,
    NodeMessageResult,
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.retained_mode.retained_mode import RetainedMode as cmd  # noqa: N813
from griptape_nodes.traits.button import ButtonDetailsMessagePayload

logger = logging.getLogger("ultralytics_nodes_library")

_NEW_LIBRARY_NAME = "Griptape Nodes Ultralytics Library"
_NEW_LIBRARY_URL = "https://github.com/griptape-ai/griptape-nodes-library-ultralytics"
_MIGRATION_MESSAGE = (
    "This node has moved to a separate library.\n"
    f"Install {_NEW_LIBRARY_NAME} from {_NEW_LIBRARY_URL} "
    "and click the button to migrate this node and its connections. "
    "This stub will be removed in advanced-media v0.72.0."
)


class YOLOv8FaceDetection(ControlNode):
    """Deprecated stub. The real implementation moved to griptape-nodes-library-ultralytics.

    Parameters are preserved so `migrate_parameter` can rewire existing connections onto
    the replacement node. Running this node raises RuntimeError.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.migrate_message = DeprecationMessage(
            value=_MIGRATION_MESSAGE,
            button_text="Create YOLOv8 Face Detection Node",
            migrate_function=self._migrate,
        )
        self.add_node_element(self.migrate_message)

        self.add_parameter(
            Parameter(
                name="input_image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Input image for face detection",
            )
        )
        self.add_parameter(
            Parameter(
                name="confidence_threshold",
                default_value=0.5,
                input_types=["float"],
                type="float",
                tooltip="Minimum confidence score for a detection to be kept",
            )
        )
        self.add_parameter(
            Parameter(
                name="dilation",
                default_value=0.0,
                input_types=["float"],
                type="float",
                tooltip="Percentage to expand detected bounding boxes",
            )
        )
        self.add_parameter(
            Parameter(
                name="detected_faces",
                output_type="list",
                tooltip="List of detected faces with bounding boxes and confidence scores",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": True},
            )
        )

    def _migrate(self, button: Any, button_details: ButtonDetailsMessagePayload) -> NodeMessageResult | None:  # noqa: ARG002
        new_node_name = f"{self.name}_migrated"

        new_node_result = cmd.create_node_relative_to(
            reference_node_name=self.name,
            new_node_type="YOLOv8FaceDetection",
            new_node_name=new_node_name,
            specific_library_name=_NEW_LIBRARY_NAME,
            offset_side="top_right",
            offset_y=-50,
            swap=True,
            match_size=True,
        )

        if isinstance(new_node_result, str):
            new_node = new_node_result
        else:
            logger.error("Failed to create node: %s", new_node_result)
            return None

        cmd.migrate_parameter(self.name, new_node, "exec_in", "exec_in")
        cmd.migrate_parameter(self.name, new_node, "exec_out", "exec_out")
        cmd.migrate_parameter(self.name, new_node, "input_image", "input_image")
        cmd.migrate_parameter(self.name, new_node, "confidence_threshold", "confidence_threshold")
        cmd.migrate_parameter(self.name, new_node, "dilation", "dilation")
        cmd.migrate_parameter(self.name, new_node, "detected_faces", "detected_faces")

        return None

    def validate_before_node_run(self) -> list[Exception] | None:
        return [RuntimeError(f"{_MIGRATION_MESSAGE} See: {_NEW_LIBRARY_URL}")]

    def process(self) -> None:
        raise RuntimeError(f"{_MIGRATION_MESSAGE} See: {_NEW_LIBRARY_URL}")
