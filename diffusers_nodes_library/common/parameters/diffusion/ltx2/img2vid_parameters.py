import logging

import diffusers  # type: ignore[reportMissingImports]
import torch  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import (
    HuggingFaceRepoParameter,
)

from diffusers_nodes_library.common.parameters.diffusion.pipeline_type_parameters import (
    DiffusionPipelineTypePipelineParameters,
)

logger = logging.getLogger("diffusers_nodes_library")

# LTX-2.0 models (variant-based with :: separator)
LTX_2_0_I2V_VARIANTS = [
    "Lightricks/LTX-2::ltx-2-19b-dev",
    "Lightricks/LTX-2::ltx-2-19b-dev-fp8",
    "Lightricks/LTX-2::ltx-2-19b-dev-fp4",
]

# LTX-2.3 models (separate repositories)
LTX_2_3_I2V_MODELS = [
    "Lightricks/LTX-2.3",
    "Lightricks/LTX-2.3-fp8",
    "Lightricks/LTX-2.3-nvfp4",
    "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
    "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
]

# Combined model list
ALL_LTX_I2V_MODELS = LTX_2_0_I2V_VARIANTS + LTX_2_3_I2V_MODELS

# Quantized models (both versions)
QUANTIZED_LTX_I2V_MODELS = [
    "Lightricks/LTX-2::ltx-2-19b-dev-fp8",
    "Lightricks/LTX-2::ltx-2-19b-dev-fp4",
    "Lightricks/LTX-2.3-fp8",
    "Lightricks/LTX-2.3-nvfp4",
]


class LTX2ImageToVideoPipelineParameters(DiffusionPipelineTypePipelineParameters):
    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=ALL_LTX_I2V_MODELS,
            parameter_name="model",
            list_all_models=list_all_models,
        )

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()

    def get_config_kwargs(self) -> dict:
        return {
            "model": self._node.get_parameter_value("model"),
        }

    @property
    def pipeline_class(self) -> type:
        return diffusers.LTX2ImageToVideoPipeline

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        model_errors = self._model_repo_parameter.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)

        return errors or None

    def build_pipeline(self) -> diffusers.LTX2ImageToVideoPipeline:
        repo_id, revision = self._model_repo_parameter.get_repo_revision()

        # LTX-2.0 variant-based loading (repo_id contains :: separator)
        if "::" in repo_id:
            base_repo, variant = repo_id.split("::")
            return diffusers.LTX2ImageToVideoPipeline.from_pretrained(
                pretrained_model_name_or_path=base_repo,
                transformer_id=variant,
                revision=revision,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )

        # LTX-2.3 direct repo loading
        return diffusers.LTX2ImageToVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=repo_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

    def is_prequantized(self) -> bool:
        repo_id, _ = self._model_repo_parameter.get_repo_revision()
        return repo_id in QUANTIZED_LTX_I2V_MODELS
