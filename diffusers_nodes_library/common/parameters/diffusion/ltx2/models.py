"""Shared model definitions for LTX2 pipelines."""

# LTX-2.0 models (variant-based with :: separator)
LTX_2_0_VARIANTS = [
    "Lightricks/LTX-2::ltx-2-19b-dev",
    "Lightricks/LTX-2::ltx-2-19b-dev-fp8",
    "Lightricks/LTX-2::ltx-2-19b-dev-fp4",
]

# LTX-2.3 models (separate repositories)
LTX_2_3_MODELS = [
    "Lightricks/LTX-2.3",
    "Lightricks/LTX-2.3-fp8",
    "Lightricks/LTX-2.3-nvfp4",
    "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
    "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
]

# Combined model list
ALL_LTX_MODELS = LTX_2_0_VARIANTS + LTX_2_3_MODELS

# Quantized models (both versions)
QUANTIZED_LTX_MODELS = [
    "Lightricks/LTX-2::ltx-2-19b-dev-fp8",
    "Lightricks/LTX-2::ltx-2-19b-dev-fp4",
    "Lightricks/LTX-2.3-fp8",
    "Lightricks/LTX-2.3-nvfp4",
]
