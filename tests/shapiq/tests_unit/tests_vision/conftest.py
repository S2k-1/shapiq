"""Shared fixtures and helpers for the shapiq.vision test suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shapiq.vision.players import CNNPlayerStrategy


class FixedMasksStrategy(CNNPlayerStrategy):
    """A deterministic player strategy that returns user-provided spatial masks.

    Useful for correctness tests where SLIC's non-determinism would interfere.
    """

    def __init__(self, masks: np.ndarray) -> None:
        self._masks = masks.astype(bool)

    def get_masks(self, image: np.ndarray) -> np.ndarray:
        return self._masks

    @property
    def n_players(self) -> int:
        return int(self._masks.shape[0])


class ChannelSumModel(torch.nn.Module):
    """A deterministic two-class CNN-like model.

    The class-0 logit equals the sum of all pixel intensities of the (masked)
    image; class-1 is its negation.  This makes the model output an exact linear
    function of the pixels that survive masking, which is what the correctness
    tests rely on.

    The model takes a ``(B, C, H, W)`` float tensor (as produced by
    :class:`~shapiq.vision.architecture.CNNArchitecture`) and returns a
    ``(B, 2)`` tensor.  The sum is accumulated in float64 so that comparisons
    against numpy float64 references are exact for integer-valued images.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x.double().sum(dim=(1, 2, 3))
        return torch.stack([total, -total], dim=1)


@pytest.fixture
def tiny_image() -> np.ndarray:
    """A small RGB image (4x4x3) of integer-valued pixels."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(4, 4, 3)).astype(np.float64)


@pytest.fixture
def two_player_masks() -> np.ndarray:
    """Two non-overlapping spatial masks partitioning a 4x4 image into halves."""
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, :, :2] = True  # left half
    masks[1, :, 2:] = True  # right half
    return masks


@pytest.fixture
def three_player_masks() -> np.ndarray:
    """Three non-overlapping spatial masks partitioning a 6x6 image into vertical thirds."""
    masks = np.zeros((3, 6, 6), dtype=bool)
    masks[0, :, 0:2] = True
    masks[1, :, 2:4] = True
    masks[2, :, 4:6] = True
    return masks
