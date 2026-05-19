"""Advanced library hook for the Advanced Media Library.

The earliest module in this library that the engine imports. We use it to claim
`sys.modules['torch']` (and the torch.* submodules we care about) for THIS
library's venv before any node module gets imported. Once a torch package object
is in `sys.modules`, every subsequent `import torch` in the engine process gets
that same object — even if another library's venv site-packages was placed on
`sys.path` first.

If a different library has already imported torch, this module logs the
mismatch loudly so the conflict is visible in the engine logs.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary

if TYPE_CHECKING:
    from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logger = logging.getLogger("advanced_media_library")

_TORCH_PACKAGES = ("torch", "torchvision", "torchaudio")


def _log_torch_state(label: str) -> None:
    for pkg in _TORCH_PACKAGES:
        mod = sys.modules.get(pkg)
        if mod is None:
            logger.warning("[torch-probe %s] %s not yet in sys.modules", label, pkg)
            continue
        version = getattr(mod, "__version__", "<unknown>")
        file = getattr(mod, "__file__", "<unknown>")
        logger.warning("[torch-probe %s] %s=%s file=%s", label, pkg, version, file)


def _claim_torch_for_this_venv() -> None:
    """Import torch (and friends) FIRST so they resolve from this library's venv.

    `_add_library_paths_to_sys_path` has already prepended this library's venv
    site-packages to `sys.path`. Importing torch here pins
    `sys.modules['torch']` to this venv's torch before any node module runs.
    """
    _log_torch_state("before-claim")

    import torch  # noqa: F401  pyright: ignore[reportMissingImports]
    import torchvision  # noqa: F401  pyright: ignore[reportMissingImports]
    import torchaudio  # noqa: F401  pyright: ignore[reportMissingImports]

    _log_torch_state("after-claim")


_claim_torch_for_this_venv()


class AdvancedMediaLibrary(AdvancedNodeLibrary):
    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:  # noqa: ARG002
        _log_torch_state(f"before-nodes:{library_data.name}")

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:  # noqa: ARG002
        _log_torch_state(f"after-nodes:{library_data.name}")
