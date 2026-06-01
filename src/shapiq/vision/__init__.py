"""Vision-based explanation methods for image models."""

from .architecture import (
    CLIPArchitecture,
    ConvNeXtArchitecture,
    CustomViTArchitecture,
    DINOv2Architecture,
    HuggingFacePixelArchitecture,
    ModelArchitectureStrategy,
    ResNetArchitecture,
    ViTArchitecture,
)
from .imputer import ImageImputer
from .masking import (
    BoolMaskedPosStrategy,
    LatentMaskingStrategy,
    MaskTokenStrategy,
    MeanColorMasking,
    PixelMaskingStrategy,
    ZeroMasking,
)
from .players import (
    GridStrategy,
    LatentPlayerStrategy,
    PatchStrategy,
    PixelPlayerStrategy,
    PlayerStrategy,
    SuperpixelStrategy,
)

__all__ = [
    # Explainer (lazy-imported via __getattr__ to avoid circular imports)
    "ImageExplainer",
    # Architecture
    "ModelArchitectureStrategy",
    "ResNetArchitecture",
    "ViTArchitecture",
    "HuggingFacePixelArchitecture",
    "ConvNeXtArchitecture",
    "DINOv2Architecture",
    "CLIPArchitecture",
    "CustomViTArchitecture",
    # Imputer
    "ImageImputer",
    # Masking
    "PixelMaskingStrategy",
    "LatentMaskingStrategy",
    "MeanColorMasking",
    "ZeroMasking",
    "BoolMaskedPosStrategy",
    "MaskTokenStrategy",
    # Players
    "PlayerStrategy",
    "PixelPlayerStrategy",
    "LatentPlayerStrategy",
    "GridStrategy",
    "SuperpixelStrategy",
    "PatchStrategy",
]


def __getattr__(name: str) -> object:
    if name == "ImageExplainer":
        from .explainer import ImageExplainer

        return ImageExplainer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
