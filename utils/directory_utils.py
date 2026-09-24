"""Directory utilities for the advanced media library."""

import logging
from pathlib import Path
from typing import Any

from griptape_nodes.retained_mode.events.config_events import (
    GetConfigValueRequest,
    GetConfigValueResultSuccess,
    GetWorkspaceRequest,
    GetWorkspaceResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.os_manager import OSManager

logger = logging.getLogger("griptape_nodes")


def get_config_value(category_and_key: str, default: Any = None) -> Any:
    """Read a configuration value over the request bus.

    Args:
        category_and_key: Configuration key in "category.key" form.
        default: Returned when the key is absent or holds no value.

    Note:
        The config manager accessor is refused while a node is executing, because the config
        belongs to the orchestrator. The request answers from either process.
    """
    result = GriptapeNodes.handle_request(GetConfigValueRequest(category_and_key=category_and_key))
    if not isinstance(result, GetConfigValueResultSuccess):
        return default
    return result.value


def get_workspace_path() -> Path:
    """Get the absolute workspace directory over the request bus."""
    result = GriptapeNodes.handle_request(GetWorkspaceRequest())
    if not isinstance(result, GetWorkspaceResultSuccess):
        msg = "Attempted to locate the workspace directory. Failed because the engine did not report one."
        raise RuntimeError(msg)
    return Path(result.workspace_path)


def cleanup_static_files_subdirectory(directory_name: str) -> None:
    """Trim one subdirectory of the static files directory to its configured size limit.

    Args:
        directory_name: Directory name relative to the static files directory.

    Note:
        `cleanup_directory_if_needed` is a static method that only touches the filesystem, and no
        os_events request wraps it, so it is called on the class rather than through the facade.
    """
    cleanup_enabled = get_config_value("advanced_media_library.enable_directory_cleanup")
    if not cleanup_enabled:
        return

    static_files_directory = get_config_value("static_files_directory", default="staticfiles")
    path = get_workspace_path() / static_files_directory / directory_name

    max_size_gb = get_config_value("advanced_media_library.max_directory_size_gb")
    OSManager.cleanup_directory_if_needed(full_directory_path=path, max_size_gb=max_size_gb)


def check_cleanup_intermediates_directory() -> None:
    """Check if directory cleanup is enabled and perform cleanup if needed.

    This function checks the configuration to see if directory cleanup is enabled
    for the advanced media library. If enabled, it will clean up the intermediates
    directory by removing the oldest files until the directory size is below the
    configured maximum size threshold.

    The function uses the following configuration values:
    - advanced_media_library.enable_directory_cleanup: Boolean to enable/disable cleanup
    - advanced_media_library.max_directory_size_gb: Maximum directory size in GB
    - advanced_media_library.temp_folder_name: Name of the intermediates directory
    - static_files_directory: Base directory for static files

    Note:
        This function is typically called before saving new intermediate files
        to ensure sufficient space is available.
    """
    # Perform cleanup if needed before saving new file
    cleanup_static_files_subdirectory(get_intermediates_directory_path())


def get_intermediates_directory_path() -> str:
    """Get the configured intermediates directory name for the advanced media library.

    This function retrieves the directory name where intermediate files (such as
    preview images during AI generation) are stored. The directory name is
    configured via the 'advanced_media_library.temp_folder_name' setting.

    Returns:
        str: The configured intermediates directory name, or "intermediates" if not configured.
            This is a directory name (not a full path) that will be used relative to
            the static files directory.

    Note:
        If the configuration value is not found, a warning is logged and the default
        "intermediates" directory name is returned.
    """
    # Get configured temp folder name, default to "intermediates"
    temp_folder_name = get_config_value("advanced_media_library.temp_folder_name")
    if temp_folder_name is None:
        logger.warning(
            "Configuration value 'advanced_media_library.temp_folder_name' not found, using default 'intermediates'"
        )
        temp_folder_name = "intermediates"
    return temp_folder_name
