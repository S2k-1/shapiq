"""This module contains all custom types used in the shapiq vision subpackage."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
        """Return raw logits, or an object exposing them as ``.logits``."""
        ...


@runtime_checkable
class ClassificationOutput(Protocol):
    """Protocol for model outputs exposing classification logits."""

    logits: torch.Tensor
