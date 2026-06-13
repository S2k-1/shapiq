"""Vision-based explanation methods for image models."""

from .architecture import (
    CLIPArchitecture,
    CNNArchitecture,
    ConvNeXtArchitecture,
    CustomViTArchitecture,
    DINOv2Architecture,
    HuggingFacePixelArchitecture,
    LayerMaskedCNNArchitecture,
    ModelArchitectureStrategy,
    TransformerArchitecture,
)
from .imputer import ImageImputer
from .masking import (
    BoolMaskedPosStrategy,
    BlurMasking,
    CNNMaskingStrategy,
    LayerMasking,
    ManifoldMaskingStrategy,
    MaskTokenStrategy,
    MeanColorMasking,
    TransformerMaskingStrategy,
    ZeroMasking,
)
from .players import (
    CNNPlayerStrategy,
    PatchStrategy,
    PlayerStrategy,
    SuperpixelStrategy,
    TransformerPlayerStrategy,
)

__all__ = [
    # Architecture
    "ModelArchitectureStrategy",
    "CNNArchitecture",
    "TransformerArchitecture",
    "HuggingFacePixelArchitecture",
    "ConvNeXtArchitecture",
    "DINOv2Architecture",
    "CLIPArchitecture",
    "LayerMaskedCNNArchitecture",
    "CustomViTArchitecture",
    # Imputer
    "ImageImputer",
    # Masking
    "CNNMaskingStrategy",
    "TransformerMaskingStrategy",
    "MeanColorMasking",
    "ZeroMasking",
    "BlurMasking",
    "BoolMaskedPosStrategy",
    "MaskTokenStrategy",
    "ManifoldMaskingStrategy",
    "LayerMasking",
    # Players
    "PlayerStrategy",
    "CNNPlayerStrategy",
    "TransformerPlayerStrategy",
    "SuperpixelStrategy",
    "PatchStrategy",
]


def __getattr__(name: str) -> object:
    if name == "ImageExplainer":
        from .explainer import ImageExplainer

        return ImageExplainer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
