import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.retained_mode.events.object_events import ClearAllObjectStateRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.settings import LIBRARIES_TO_REGISTER_KEY
from griptape_nodes.utils import install_file_url_support

logger = logging.getLogger(__name__)

# Install file:// URL support for httpx/requests in tests
install_file_url_support()

LIBRARY_ROOT = Path(__file__).parents[2]


def get_workflows() -> list[str]:
    """Get all workflow templates for this library."""
    workflows_dir = LIBRARY_ROOT / "workflows" / "templates"
    return [
        str(f)
        for f in workflows_dir.iterdir()
        if f.is_file() and f.suffix == ".py" and not f.name.startswith("__")
    ]


load_dotenv()


@pytest.fixture(scope="session")
def griptape_nodes() -> GriptapeNodes:
    """Initialize GriptapeNodes before tests and clean up afterwards."""
    return GriptapeNodes()


@pytest_asyncio.fixture(scope="session")
async def workflow_executor() -> AsyncGenerator[LocalWorkflowExecutor, Any]:
    """Create and manage a single LocalWorkflowExecutor for all tests."""
    async with LocalWorkflowExecutor() as executor:
        yield executor


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_library(griptape_nodes: GriptapeNodes) -> AsyncGenerator[None, Any]:
    """Set up this library for testing and restore original state afterwards."""
    config_manager = griptape_nodes.ConfigManager()

    # Save the original libraries state
    original_libraries = config_manager.get_config_value(key=LIBRARIES_TO_REGISTER_KEY, default=[])

    # Set this library for testing
    config_manager.set_config_value(
        key=LIBRARIES_TO_REGISTER_KEY,
        value=[str(LIBRARY_ROOT / "griptape_nodes_library.json")],
    )

    yield  # Run all tests

    # Restore original libraries state
    config_manager.set_config_value(
        key=LIBRARIES_TO_REGISTER_KEY,
        value=original_libraries,
    )


@pytest_asyncio.fixture(autouse=True)
async def clear_state_before_each_test(griptape_nodes: GriptapeNodes) -> AsyncGenerator[None, Any]:
    """Clear all object state before each test to ensure clean starting conditions."""
    clear_request = ClearAllObjectStateRequest(i_know_what_im_doing=True)
    await griptape_nodes.ahandle_request(clear_request)

    griptape_nodes.ConfigManager()._set_log_level("DEBUG")

    yield  # Run the test

    # Clean up after test
    clear_request = ClearAllObjectStateRequest(i_know_what_im_doing=True)
    await griptape_nodes.ahandle_request(clear_request)


@pytest.mark.parametrize("workflow_path", get_workflows())
@pytest.mark.asyncio
async def test_workflow_runs(workflow_path: str, workflow_executor: LocalWorkflowExecutor) -> None:
    """Simple test to check if the workflow runs without errors."""
    await workflow_executor.arun(workflow_name="main", flow_input={}, workflow_path=workflow_path)
