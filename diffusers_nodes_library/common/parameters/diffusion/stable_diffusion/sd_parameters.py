from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import diffusers  # type: ignore[reportMissingImports]

import logging

from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from diffusers_nodes_library.common.parameters.diffusion.pipeline_type_parameters import (
    DiffusionPipelineTypePipelineParameters,
)

logger = logging.getLogger("diffusers_nodes_library")


class StableDiffusionPipelineParameters(DiffusionPipelineTypePipelineParameters):
    PIPELINE_NAME: ClassVar[str] = "StableDiffusionPipeline"

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._huggingface_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "CompVis/stable-diffusion-v1-4",
                "CompVis/stable-diffusion-v1-3",
                "CompVis/stable-diffusion-v1-2",
                "CompVis/stable-diffusion-v1-1",
            ],
            deprecated_repo_ids=[
                "stabilityai/stable-diffusion-2-1",
                "stabilityai/stable-diffusion-2",
            ],
            list_all_models=list_all_models,
        )

    def add_input_parameters(self) -> None:
        self._huggingface_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._huggingface_repo_parameter.remove_input_parameters()

    def get_config_kwargs(self) -> dict:
        return {
            "model": self._node.get_parameter_value("model"),
        }

    @property
    def pipeline_class(self) -> type:
        import diffusers  # type: ignore[reportMissingImports]

        return diffusers.StableDiffusionPipeline

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = self._huggingface_repo_parameter.validate_before_node_run()
        return errors or None

    def build_pipeline(self) -> diffusers.StableDiffusionPipeline:
        import diffusers  # type: ignore[reportMissingImports]
        import torch  # type: ignore[reportMissingImports]

        repo_id, revision = self._huggingface_repo_parameter.get_repo_revision()
        return diffusers.StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=repo_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
