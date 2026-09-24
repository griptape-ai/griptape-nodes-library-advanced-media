from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import diffusers  # type: ignore[reportMissingImports]

import logging
from abc import ABC, abstractmethod

from griptape_nodes.exe_types.node_types import BaseNode

logger = logging.getLogger("diffusers_nodes_library")


class DiffusionPipelineTypePipelineParameters(ABC):
    # Every concrete subclass must declare the name of the pipeline class it builds. The name
    # has to be available on the orchestrator, which has no diffusers, so it cannot be read off
    # `pipeline_class.__name__`: `pipeline_name` feeds the config hash that the builder node
    # stamps into its output during `after_value_set`.
    PIPELINE_NAME: ClassVar[str]

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        self._node = node
        self._list_all_models = list_all_models

    @abstractmethod
    def add_input_parameters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_input_parameters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_config_kwargs(self) -> dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def pipeline_class(self) -> type[diffusers.DiffusionPipeline]:
        raise NotImplementedError

    @property
    def pipeline_name(self) -> str:
        return self.PIPELINE_NAME

    @abstractmethod
    def validate_before_node_run(self) -> list[Exception] | None:
        raise NotImplementedError

    def validate_in_execution_environment(self) -> list[Exception] | None:
        """Checks that need the real diffusers classes, run where the pipeline is built."""
        return None

    @abstractmethod
    def build_pipeline(self) -> diffusers.DiffusionPipeline:
        raise NotImplementedError

    def is_prequantized(self) -> bool:
        """Return True if the model is already quantized (e.g., bnb-4bit).

        Pre-quantized models should not have layerwise casting or additional
        quantization applied.
        """
        return False

    def supports_layerwise_casting(self) -> bool:
        """Return True if the pipeline's transformer supports layerwise casting.

        Some transformers (e.g., ZImage) check weight dtype before calling modules,
        which is incompatible with layerwise casting hooks that cast weights during
        the forward pass.
        """
        return True

    def requires_device_map(self) -> bool:
        """Return True if the pipeline requires device_map during loading.

        Some pipelines (e.g., GLM-Image) have components that must be loaded with
        accelerate's device_map to properly materialize weights. When True:
        - build_pipeline() should use device_map parameter
        - optimize_diffusion_pipeline() should skip .to(device) and CPU offload calls
        """
        return False
