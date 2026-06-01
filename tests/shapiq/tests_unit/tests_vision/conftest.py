"""Shared fixtures and helpers for the shapiq.vision test suite."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.vision.players import PixelPlayerStrategy


class FixedMasksStrategy(PixelPlayerStrategy):
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


def make_linear_pixel_model(weights: np.ndarray, bias: float = 0.0):
    """Make a deterministic callable that sums pixel intensities under given per-pixel weights.

    The model takes a batch of images of shape (B, H, W, C) and returns shape (B,).
    """

    def model(batch: np.ndarray) -> np.ndarray:
        arr = np.asarray(batch, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[np.newaxis]
        # Sum over H, W, C weighted by `weights` broadcast across channels.
        return (arr * weights[np.newaxis, :, :, np.newaxis]).sum(axis=(1, 2, 3)) + bias

    return model
