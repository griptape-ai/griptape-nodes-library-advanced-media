from griptape_nodes.exe_types.node_types import BaseNode

from diffusers_nodes_library.common.parameters.diffusion.common.upscale_runtime_parameters import (
    UpscalePipelineRuntimeParameters,
)


class FluxUpscalePipelineRuntimeParameters(UpscalePipelineRuntimeParameters):
    def __init__(self, node: BaseNode):
        super().__init__(node)
