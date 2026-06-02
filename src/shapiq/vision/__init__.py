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
    BlurMasking,
    BoolMaskedPosStrategy,
    DatasetMeanMasking,
    GaussianNoiseMasking,
    LatentMaskingStrategy,
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
from .utils import ImageLike, as_hwc_array

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
    # Explainer
    "ImageExplainer",
    # Imputer
    "ImageImputer",
    # Masking
    "PixelMaskingStrategy",
    "LatentMaskingStrategy",
    "MeanColorMasking",
    "ZeroMasking",
    "BlurMasking",
    "BoolMaskedPosStrategy",
    "DatasetMeanMasking",
    "GaussianNoiseMasking",
    "MaskTokenStrategy",
    "ImageLike",
    "as_hwc_array",
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
