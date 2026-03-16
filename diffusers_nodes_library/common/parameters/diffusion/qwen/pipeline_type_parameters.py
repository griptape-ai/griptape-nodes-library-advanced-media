import logging

from diffusers_nodes_library.common.parameters.diffusion.diffusion_pipeline_type_parameters import (
    DiffusionPipelineTypeParameters,
)
from diffusers_nodes_library.common.parameters.diffusion.pipeline_type_parameters import (
    DiffusionPipelineTypePipelineParameters,
)
from diffusers_nodes_library.common.parameters.diffusion.qwen.controlnet_parameters import (
    QwenImageControlNetPipelineParameters,
)
from diffusers_nodes_library.common.parameters.diffusion.qwen.edit_inpaint_parameters import (
    QwenImageEditInpaintPipelineParameters,
)
from diffusers_nodes_library.common.parameters.diffusion.qwen.edit_parameters import (
    QwenEditPipelineParameters,
)
from diffusers_nodes_library.common.parameters.diffusion.qwen.edit_plus_parameters import (
    QwenImageEditPlusPipelineParameters,
)
from diffusers_nodes_library.common.parameters.diffusion.qwen.img2img_parameters import (
    QwenImg2ImgPipelineParameters,
)
from diffusers_nodes_library.common.parameters.diffusion.qwen.inpaint_parameters import (
    QwenInpaintPipelineParameters,
)
from diffusers_nodes_library.common.parameters.diffusion.qwen.layered_parameters import (
    QwenLayeredPipelineParameters,
)
from diffusers_nodes_library.common.parameters.diffusion.qwen.qwen_parameters import (
    QwenPipelineParameters,
)
from diffusers_nodes_library.common.parameters.diffusion.qwen.upscale_parameters import (
    QwenUpscalePipelineParameters,
)

logger = logging.getLogger("diffusers_nodes_library")


QwenPipelineTypeDict: dict[str, type[DiffusionPipelineTypePipelineParameters]] = {
    "QwenImagePipeline": QwenPipelineParameters,
    "QwenImageImg2ImgPipeline": QwenImg2ImgPipelineParameters,
    "QwenImageEditPipeline": QwenEditPipelineParameters,
    "QwenImageEditInpaintPipeline": QwenImageEditInpaintPipelineParameters,
    "QwenImageEditPlusPipeline": QwenImageEditPlusPipelineParameters,
    "QwenImageInpaintPipeline": QwenInpaintPipelineParameters,
    "QwenImageUpscalePipeline": QwenUpscalePipelineParameters,
    "QwenImageControlNetPipeline": QwenImageControlNetPipelineParameters,
    "QwenImageLayeredPipeline": QwenLayeredPipelineParameters,
}


class QwenPipelineTypeParameters(DiffusionPipelineTypeParameters):
    @property
    def pipeline_type_dict(self) -> dict[str, type[DiffusionPipelineTypePipelineParameters]]:
        return QwenPipelineTypeDict
