import logging

import diffusers  # type: ignore[reportMissingImports]
import torch  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.exe_types.param_components.huggingface.huggingface_utils import (
    list_repo_revisions_in_cache,
)

from diffusers_nodes_library.common.parameters.diffusion.pipeline_type_parameters import (
    DiffusionPipelineTypePipelineParameters,
)

logger = logging.getLogger("diffusers_nodes_library")

QUANTIZED_FLUX_2_REPO_IDS = [
    "diffusers/FLUX.2-dev-bnb-4bit",
    "black-forest-labs/FLUX.2-dev-NVFP4",
]

FLUX_2_REPO_IDS = [*QUANTIZED_FLUX_2_REPO_IDS, "black-forest-labs/FLUX.2-dev", "fal/FLUX.2-dev-Turbo"]


class Flux2PipelineParameters(DiffusionPipelineTypePipelineParameters):
    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=FLUX_2_REPO_IDS,
            parameter_name="model",
            list_all_models=list_all_models,
        )

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()
        self._node.add_parameter(
            Parameter(
                name="use_small_decoder",
                default_value=False,
                type="bool",
                tooltip="Use FLUX.2-small-decoder for faster VAE decoding (requires model to be downloaded)",
            )
        )

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()
        self._node.remove_parameter_element_by_name("use_small_decoder")

    def get_config_kwargs(self) -> dict:
        return {
            "model": self._node.get_parameter_value("model"),
            "use_small_decoder": self._node.get_parameter_value("use_small_decoder"),
        }

    @property
    def pipeline_class(self) -> type:
        return diffusers.Flux2Pipeline

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        model_errors = self._model_repo_parameter.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)

        return errors or None

    def build_pipeline(self) -> diffusers.Flux2Pipeline:
        base_repo_id, base_revision = self._model_repo_parameter.get_repo_revision()
        use_small_decoder = self._node.get_parameter_value("use_small_decoder")

        # Build pipeline kwargs
        pipeline_kwargs = {
            "pretrained_model_name_or_path": base_repo_id,
            "revision": base_revision,
            "torch_dtype": torch.bfloat16,
            "local_files_only": True,
        }

        # Optionally load the small decoder VAE if requested and available
        if use_small_decoder:
            small_decoder_repo = "black-forest-labs/FLUX.2-small-decoder"
            # Check if the small decoder is downloaded
            cached_revisions = list_repo_revisions_in_cache(small_decoder_repo)
            if cached_revisions:
                logger.info("Loading FLUX.2-small-decoder VAE")
                pipeline_kwargs["vae"] = diffusers.AutoencoderKLFlux2.from_pretrained(
                    small_decoder_repo,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                )
            else:
                logger.warning(
                    "use_small_decoder is enabled but %s is not downloaded. ",
                    small_decoder_repo,
                )

        return diffusers.Flux2Pipeline.from_pretrained(**pipeline_kwargs)

    def is_prequantized(self) -> bool:
        repo_id, _ = self._model_repo_parameter.get_repo_revision()
        return repo_id in QUANTIZED_FLUX_2_REPO_IDS
