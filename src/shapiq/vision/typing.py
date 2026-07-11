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

    def __call__(self, x: torch.Tensor) -> torch.Tensor | ClassificationOutput: ...


@runtime_checkable
class ClassificationOutput(Protocol):
    """Protocol for model outputs exposing classification logits."""

    logits: torch.Tensor


@runtime_checkable
class ViTLikeModel(Protocol):
    """Protocol for ViT-like models supporting token masking via bool_masked_pos."""

    def __call__(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        **kwargs: Any,
    ) -> ClassificationOutput: ...
