"""This module contains all custom types used in the shapiq vision subpackage."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


class CoalitionDomain(Enum):
    """Enumeration of coalition domains used by players and masking strategies."""

    PIXEL = "pixel"
    TOKEN = "token"  # noqa: S105


@runtime_checkable
class VisionModel(Protocol):
    """Protocol for vision models called directly on image tensors."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor | ClassificationOutput:
        """Run inference on a batch of image tensors and return predictions."""
        ...


@runtime_checkable
class ClassificationOutput(Protocol):
    """Protocol for model outputs exposing classification logits."""

    logits: torch.Tensor


@runtime_checkable
class ViTLikeModel(Protocol):
    """Protocol for ViT-like models supporting token masking via bool_masked_pos.

    The protocol is intentionally wide: at runtime ``isinstance`` only verifies
    that the object is callable and exposes a ``config`` attribute. Whether a
    model *actually* honors ``bool_masked_pos`` cannot be read off its
    signature (many ``transformers`` classification heads accept it via
    ``**kwargs`` and silently drop it), so
    :func:`shapiq.vision.dispatch.probe_token_masking` verifies it
    functionally.
    """

    config: Any

    def __call__(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        **kwargs: object,
    ) -> ClassificationOutput:
        """Run inference on preprocessed pixel values, optionally masking tokens."""
        ...


@runtime_checkable
class ImageProcessorLike(Protocol):
    """Protocol for Hugging Face-style image processors.

    Matches any callable that turns raw images into a mapping containing
    ``pixel_values`` (e.g. :class:`transformers.ViTImageProcessor` or any
    ``AutoImageProcessor`` result).
    """

    def __call__(self, images: object, return_tensors: str = ..., **kwargs: object) -> object:
        """Convert raw image(s) into a mapping containing ``pixel_values``."""
        ...
