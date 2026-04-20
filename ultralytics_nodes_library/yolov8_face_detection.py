import logging

from griptape_nodes.exe_types.core_types import ParameterMessage
from griptape_nodes.exe_types.node_types import ControlNode

logger = logging.getLogger("ultralytics_nodes_library")

_NEW_LIBRARY_URL = "https://github.com/griptape-ai/griptape-nodes-library-ultralytics"
_MIGRATION_MESSAGE = (
    "This node has moved to a separate library. "
    "Install griptape-nodes-library-ultralytics and re-add the node from there. "
    "This stub will be removed in advanced-media v0.72.0."
)


class YOLOv8FaceDetection(ControlNode):
    """Deprecated stub. The real implementation moved to griptape-nodes-library-ultralytics.

    This class is retained only so existing workflows deserialize and can be visually
    migrated. Any attempt to run it raises RuntimeError.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_node_element(
            ParameterMessage(
                name="migration_notice",
                variant="warning",
                title="Node Moved",
                value=_MIGRATION_MESSAGE,
                button_text="Open griptape-nodes-library-ultralytics",
                button_link=_NEW_LIBRARY_URL,
            )
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        return [RuntimeError(f"{_MIGRATION_MESSAGE} See: {_NEW_LIBRARY_URL}")]

    def process(self) -> None:
        raise RuntimeError(f"{_MIGRATION_MESSAGE} See: {_NEW_LIBRARY_URL}")
