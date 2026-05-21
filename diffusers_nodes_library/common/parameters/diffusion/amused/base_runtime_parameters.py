import inspect
import logging
from typing import Any

import torch  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from diffusers_nodes_library.common.parameters.diffusion.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)

logger = logging.getLogger("diffusers_nodes_library")


class AmusedPipelineRuntimeParametersBase(DiffusionPipelineRuntimeParameters):
    """Shared runtime behavior for Amused* pipelines.

    AmusedPipeline / AmusedImg2ImgPipeline / AmusedInpaintPipeline expose the
    legacy ``callback(step, timestep, latents)`` / ``callback_steps`` API and do
    not accept ``callback_on_step_end``. The img2img and inpaint variants also
    do not accept ``width`` / ``height``. This override adapts the legacy
    callback to the base class's ``callback_on_step_end`` signature and drops
    any kwargs the specific Amused pipeline doesn't accept.
    """

    def _process_pipeline_output(self, pipe: DiffusionPipeline, callback_on_step_end: Any) -> None:
        def legacy_callback(i: int, t: int, latents: torch.Tensor) -> None:
            callback_on_step_end(pipe, i, t, {"latents": latents})

        accepted = set(inspect.signature(pipe.__call__).parameters)
        kwargs = {k: v for k, v in self.get_pipe_kwargs().items() if k in accepted}

        output_image_pil = pipe(  # type: ignore[reportCallIssue]
            **kwargs,
            output_type="pil",
            callback=legacy_callback,
            callback_steps=1,
        ).images[0]
        self.publish_output_image(output_image_pil)
