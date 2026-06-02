"""Tests for masking strategies in ``shapiq.vision.masking``."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shapiq.vision.masking import (
    BlurMasking,
    BoolMaskedPosStrategy,
    LatentMaskingStrategy,
    MaskTokenStrategy,
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


class TestBlurMasking:
    def test_is_pixel_masking_strategy(self) -> None:
        assert isinstance(BlurMasking(), PixelMaskingStrategy)

    def test_default_sigma(self) -> None:
        assert BlurMasking().sigma == pytest.approx(10.0)

    def test_custom_sigma(self) -> None:
        assert BlurMasking(sigma=5.0).sigma == pytest.approx(5.0)

    def test_output_shape(self, image, half_masks) -> None:
        strategy = BlurMasking(sigma=2.0)
        coalition = np.array([[True, True], [False, False], [True, False]])
        out = strategy.apply(image, half_masks, coalition)
        assert out.shape == (3, 4, 4, 3)

    def test_full_coalition_preserves_image(self, image, half_masks) -> None:
        """When all players are present, pixels should equal the original image."""
        strategy = BlurMasking(sigma=2.0)
        coalition = np.array([[True, True]])
        out = strategy.apply(image, half_masks, coalition)
        np.testing.assert_array_equal(out[0], image)

    def test_empty_coalition_differs_from_original(self, image, half_masks) -> None:
        """When all players are absent, every pixel is replaced with a blurred value."""
        strategy = BlurMasking(sigma=2.0)
        coalition = np.array([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        # The blurred image won't equal the original unless sigma=0.
        # At least some pixels must differ (for any non-constant image).
        assert not np.allclose(out[0], image)

    def test_partial_coalition_keeps_present_region(self, image, half_masks) -> None:
        """Present players' pixels must be identical to the original image."""
        strategy = BlurMasking(sigma=2.0)
        coalition = np.array([[True, False]])  # player 0 present, player 1 absent
        out = strategy.apply(image, half_masks, coalition)
        # Left half (player 0 present) must be unchanged.
        np.testing.assert_array_equal(out[0, :, :2], image[:, :2])

    def test_partial_coalition_blurs_absent_region(self, image, half_masks) -> None:
        """Absent players' pixels must differ from the original (unless image is constant)."""
        from scipy.ndimage import gaussian_filter

        strategy = BlurMasking(sigma=2.0)
        coalition = np.array([[False, True]])  # player 0 absent, player 1 present
        out = strategy.apply(image, half_masks, coalition)
        blurred = gaussian_filter(image, sigma=[2.0, 2.0, 0])
        # Left half (player 0 absent) should match the blurred image.
        np.testing.assert_allclose(out[0, :, :2], blurred[:, :2], atol=1e-10)

    def test_multiple_coalitions_independent(self, image, half_masks) -> None:
        strategy = BlurMasking(sigma=2.0)
        coalitions = np.array([[True, True], [False, False]])
        out = strategy.apply(image, half_masks, coalitions)
        assert out.shape == (2, 4, 4, 3)
        np.testing.assert_array_equal(out[0], image)
        assert not np.allclose(out[1], image)


def test_latent_masking_strategy_is_abstract() -> None:
    """LatentMaskingStrategy is an ABC; cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LatentMaskingStrategy()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Helpers for latent masking tests
# ---------------------------------------------------------------------------


class _MockViTWithMaskToken:
    """Minimal ViT mock that satisfies MaskTokenStrategy's requirements.

    MaskTokenStrategy reads ``model.config.hidden_size`` to create the zero
    tensor and writes to ``model.vit.embeddings.mask_token``.  The forward
    returns logits whose class-0 score equals the number of *visible* tokens.
    """

    class _Config:
        hidden_size = 4

    config = _Config()

    def __init__(self) -> None:
        self.vit = SimpleNamespace(
            embeddings=SimpleNamespace(mask_token=torch.nn.Parameter(torch.ones(1, 1, 4)))
        )

    def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
        visible = (~bool_masked_pos).sum(dim=1).float()
        return SimpleNamespace(logits=torch.stack([visible, -visible], dim=1))


class TestBoolMaskedPosStrategy:
    def test_is_latent_masking_strategy(self) -> None:
        assert isinstance(BoolMaskedPosStrategy(), LatentMaskingStrategy)

    def test_predict_logits_output_shape(self) -> None:
        class _MinimalModel:
            def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
                visible = (~bool_masked_pos).sum(dim=1).float()
                return SimpleNamespace(logits=torch.stack([visible, -visible], dim=1))

        strategy = BoolMaskedPosStrategy()
        pixel_values = torch.zeros(1, 3, 4, 4)
        bool_masks = torch.zeros(5, 4, dtype=torch.bool)
        logits = strategy.predict_logits(_MinimalModel(), pixel_values, bool_masks)
        assert logits.shape == (5, 2)

    def test_predict_logits_repeats_pixel_values(self) -> None:
        """pixel_values is repeated B times for a batch of B bool_masks."""
        seen: list[tuple] = []

        class _ShapeCapture:
            def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
                seen.append(tuple(pixel_values.shape))
                b = bool_masked_pos.shape[0]
                return SimpleNamespace(logits=torch.zeros(b, 2))

        strategy = BoolMaskedPosStrategy()
        pixel_values = torch.zeros(1, 3, 4, 4)
        bool_masks = torch.zeros(7, 4, dtype=torch.bool)
        strategy.predict_logits(_ShapeCapture(), pixel_values, bool_masks)
        assert seen[-1] == (7, 3, 4, 4)


class TestMaskTokenStrategy:
    def test_is_latent_masking_strategy(self) -> None:
        assert isinstance(MaskTokenStrategy(), LatentMaskingStrategy)

    def test_predict_logits_output_shape(self) -> None:
        """predict_logits returns ``(B, n_classes)`` logits."""
        strategy = MaskTokenStrategy()
        model = _MockViTWithMaskToken()
        pixel_values = torch.zeros(1, 3, 4, 4)
        bool_masks = torch.zeros(3, 4, dtype=torch.bool)
        bool_masks[1, :2] = True
        logits = strategy.predict_logits(model, pixel_values, bool_masks)
        assert logits.shape == (3, 2)

    def test_predict_logits_zeros_mask_token(self) -> None:
        """After predict_logits the model's mask_token parameter is all zeros."""
        strategy = MaskTokenStrategy()
        model = _MockViTWithMaskToken()
        # Confirm it starts non-zero.
        assert not torch.allclose(model.vit.embeddings.mask_token.data, torch.zeros(1, 1, 4))
        pixel_values = torch.zeros(1, 3, 4, 4)
        bool_masks = torch.zeros(1, 4, dtype=torch.bool)
        strategy.predict_logits(model, pixel_values, bool_masks)
        assert torch.allclose(model.vit.embeddings.mask_token.data, torch.zeros(1, 1, 4))

    def test_predict_logits_repeats_pixel_values(self) -> None:
        """pixel_values (shape 1xCxHxW) is expanded to BxCxHxW inside the model call."""
        seen_shapes: list[tuple] = []

        class _ShapeCapture:
            class _Config:
                hidden_size = 4

            config = _Config()

            def __init__(self) -> None:
                self.vit = SimpleNamespace(
                    embeddings=SimpleNamespace(mask_token=torch.nn.Parameter(torch.zeros(1, 1, 4)))
                )

            def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
                seen_shapes.append(tuple(pixel_values.shape))
                b = bool_masked_pos.shape[0]
                return SimpleNamespace(logits=torch.zeros(b, 2))

        strategy = MaskTokenStrategy()
        pixel_values = torch.zeros(1, 3, 4, 4)
        bool_masks = torch.zeros(5, 4, dtype=torch.bool)
        strategy.predict_logits(_ShapeCapture(), pixel_values, bool_masks)
        assert seen_shapes[-1] == (5, 3, 4, 4)

    def test_logits_reflect_mask_token_zeroing(self) -> None:
        """Visible tokens (bool_mask False) drive the class-0 score; masked tokens do not."""
        strategy = MaskTokenStrategy()
        model = _MockViTWithMaskToken()
        pixel_values = torch.zeros(1, 3, 4, 4)
        # Coalition 1: all 4 tokens visible.  Coalition 2: no tokens visible.
        bool_masks = torch.stack(
            [
                torch.zeros(4, dtype=torch.bool),  # all visible
                torch.ones(4, dtype=torch.bool),  # all masked
            ]
        )
        logits = strategy.predict_logits(model, pixel_values, bool_masks)
        # Class-0 logit = #visible, so coalition 0 (4 visible) > coalition 1 (0 visible).
        assert logits[0, 0] > logits[1, 0]
