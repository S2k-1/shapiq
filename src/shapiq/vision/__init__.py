"""Vision-based explanation methods for image models."""

from .architecture import (
    CLIPArchitecture,
    ConvNeXtArchitecture,
    CustomViTArchitecture,
    DINOv2Architecture,
    HuggingFacePixelArchitecture,
    LayerMaskedCNNArchitecture,
    ModelArchitectureStrategy,
    ResNetArchitecture,
    ViTArchitecture,
)
from .imputer import ImageImputer
from .masking import (
    BlurMasking,
    BoolMaskedPosStrategy,
    InpaintingMasking,
    LatentMaskingStrategy,
    LayerMasking,
    ManifoldMaskingStrategy,
    MarginalSampling,
    MaskTokenStrategy,
    MeanColorMasking,
    PixelMaskingStrategy,
    ZeroMasking,
)
from .players import (
    CustomMasksStrategy,
    GridStrategy,
    LatentPlayerStrategy,
    PatchStrategy,
    PixelPlayerStrategy,
    PlayerStrategy,
    SuperpixelStrategy,
)

__all__ = [
    # Architecture
    "ModelArchitectureStrategy",
    "ResNetArchitecture",
    "ViTArchitecture",
    "HuggingFacePixelArchitecture",
    "ConvNeXtArchitecture",
    "DINOv2Architecture",
    "CLIPArchitecture",
    "CustomViTArchitecture",
    "LayerMaskedCNNArchitecture",
    # Explainer
    "ImageExplainer",
    # Imputer
    "ImageImputer",
    # Masking
    "PixelMaskingStrategy",
    "LatentMaskingStrategy",
    "ManifoldMaskingStrategy",
    "MeanColorMasking",
    "ZeroMasking",
    "BlurMasking",
    "MarginalSampling",
    "InpaintingMasking",
    "BoolMaskedPosStrategy",
    "MaskTokenStrategy",
    "LayerMasking",
    # Players
    "PlayerStrategy",
    "PixelPlayerStrategy",
    "LatentPlayerStrategy",
    "SuperpixelStrategy",
    "PatchStrategy",
    "GridStrategy",
    "CustomMasksStrategy",
]


def __getattr__(name: str) -> object:
    if name == "ImageExplainer":
        from .explainer import ImageExplainer

        return ImageExplainer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
