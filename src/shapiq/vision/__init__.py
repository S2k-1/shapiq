"""Vision-based explanation methods for image models."""

# TODO: Adjust this when package is more mature and we have a better sense of the structure of the vision module. For now, we can just import everything in the submodules here for ease of use.
from .players import PlayerStrategy, SuperpixelStrategy
from .masking import MaskingStrategy, ZeroMasking, MeanColorMasking
from .imputer import ImageImputer

__all__ = [
    "ImageImputer",
    "PlayerStrategy",
    "SuperpixelStrategy",
    "MaskingStrategy",
    "ZeroMasking",
    "MeanColorMasking",
]

# This function is used to lazily import the ImageExplainer class when it is accessed as an attribute of the module.
def __getattr__(name: str) -> object:
    if name == "ImageExplainer":
        from .explainer import ImageExplainer

        return ImageExplainer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)