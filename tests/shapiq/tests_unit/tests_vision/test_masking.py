"""Tests for masking strategies in ``shapiq.vision.masking``."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.vision.masking import (
    LatentMaskingStrategy,
    MeanColorMasking,
    PixelMaskingStrategy,
    ZeroMasking,
)


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(4, 4, 3)).astype(np.float64)


@pytest.fixture
def half_masks() -> np.ndarray:
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, :, :2] = True
    masks[1, :, 2:] = True
    return masks


class TestMeanColorMasking:
    def test_is_pixel_masking_strategy(self) -> None:
        assert isinstance(MeanColorMasking(), PixelMaskingStrategy)

    def test_full_coalition_preserves_image(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalition = np.array([[True, True]])
        out = strategy.apply(image, half_masks, coalition)
        assert out.shape == (1, 4, 4, 3)
        np.testing.assert_array_equal(out[0], image)

    def test_empty_coalition_uses_mean_color_everywhere(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalition = np.array([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        mean_color = image.mean(axis=(0, 1))
        expected = np.broadcast_to(mean_color, image.shape)
        np.testing.assert_allclose(out[0], expected)

    def test_partial_coalition_masks_only_absent_player(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalition = np.array([[True, False]])
        out = strategy.apply(image, half_masks, coalition)
        # Left half (player 0 present) should be preserved.
        np.testing.assert_array_equal(out[0, :, :2], image[:, :2])
        # Right half (player 1 absent) should be replaced with mean color.
        mean_color = image.mean(axis=(0, 1))
        np.testing.assert_allclose(out[0, :, 2:], np.broadcast_to(mean_color, (4, 2, 3)))

    def test_multiple_coalitions_handled_independently(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalitions = np.array(
            [
                [True, True],
                [False, False],
                [True, False],
                [False, True],
            ],
        )
        out = strategy.apply(image, half_masks, coalitions)
        assert out.shape == (4, 4, 4, 3)
        np.testing.assert_array_equal(out[0], image)
        mean_color = image.mean(axis=(0, 1))
        np.testing.assert_allclose(out[1], np.broadcast_to(mean_color, image.shape))
        np.testing.assert_array_equal(out[2, :, :2], image[:, :2])
        np.testing.assert_array_equal(out[3, :, 2:], image[:, 2:])


class TestZeroMasking:
    def test_is_pixel_masking_strategy(self) -> None:
        assert isinstance(ZeroMasking(), PixelMaskingStrategy)

    def test_default_value_is_zero(self, image, half_masks) -> None:
        strategy = ZeroMasking()
        coalition = np.array([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        assert (out[0] == 0).all()

    def test_custom_value(self, image, half_masks) -> None:
        strategy = ZeroMasking(value=7.0)
        coalition = np.array([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        assert (out[0] == 7.0).all()

    def test_partial_coalition_zeros_only_absent(self, image, half_masks) -> None:
        strategy = ZeroMasking()
        coalition = np.array([[False, True]])
        out = strategy.apply(image, half_masks, coalition)
        # Left half (absent) zeroed; right half (present) preserved.
        assert (out[0, :, :2] == 0).all()
        np.testing.assert_array_equal(out[0, :, 2:], image[:, 2:])

    def test_full_coalition_preserves_image(self, image, half_masks) -> None:
        strategy = ZeroMasking()
        coalition = np.array([[True, True]])
        out = strategy.apply(image, half_masks, coalition)
        np.testing.assert_array_equal(out[0], image)


def test_latent_masking_strategy_is_abstract() -> None:
    """LatentMaskingStrategy is an ABC; cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LatentMaskingStrategy()  # type: ignore[abstract]
