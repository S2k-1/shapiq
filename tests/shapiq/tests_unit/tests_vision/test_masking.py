"""Tests for masking strategies in ``shapiq.vision.masking``.

The pixel-space maskers (:class:`MeanColorMasking`, :class:`ZeroMasking`)
operate on ``(C, H, W)`` float tensors and a ``(n_coalitions, n_players)``
boolean coalition tensor, returning a ``(n_coalitions, C, H, W)`` batch.

The token-space maskers (:class:`BoolMaskedPosStrategy`,
:class:`MaskTokenStrategy`) turn a coalition tensor into a flat token-level
boolean mask where ``True`` marks an absent (masked) token.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from shapiq.vision.custom_types import CoalitionDomain
from shapiq.vision.masking import (
    BoolMaskedPosStrategy,
    LatentBasedMaskingStrategy,
    MaskTokenStrategy,
    MeanColorMasking,
    PixelBasedMaskingStrategy,
    ZeroMasking,
)

from .conftest import MockViT, make_vit_config


@pytest.fixture
def image() -> torch.Tensor:
    """A (C, H, W) float image with deterministic content."""
    rng = torch.Generator().manual_seed(0)
    return torch.randint(0, 255, size=(3, 4, 4), generator=rng).float()


@pytest.fixture
def half_masks() -> torch.Tensor:
    """Two non-overlapping (n_players, H, W) masks splitting a 4x4 image into halves."""
    masks = torch.zeros((2, 4, 4), dtype=torch.bool)
    masks[0, :, :2] = True  # left half
    masks[1, :, 2:] = True  # right half
    return masks


class TestMeanColorMasking:
    def test_is_cnn_masking_strategy(self) -> None:
        assert isinstance(MeanColorMasking(), PixelBasedMaskingStrategy)

    def test_full_coalition_preserves_image(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalition = torch.tensor([[True, True]])
        out = strategy.apply(image, half_masks, coalition)
        assert out.shape == (1, 3, 4, 4)
        torch.testing.assert_close(out[0], image)

    def test_empty_coalition_uses_mean_color_everywhere(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalition = torch.tensor([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        mean_color = image.mean(dim=(1, 2))  # (C,)
        expected = mean_color[:, None, None].expand(3, 4, 4)
        torch.testing.assert_close(out[0], expected)

    def test_partial_coalition_masks_only_absent_player(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalition = torch.tensor([[True, False]])
        out = strategy.apply(image, half_masks, coalition)
        # Left half (player 0 present) preserved.
        torch.testing.assert_close(out[0, :, :, :2], image[:, :, :2])
        # Right half (player 1 absent) replaced with the per-channel mean color.
        mean_color = image.mean(dim=(1, 2))
        expected_right = mean_color[:, None, None].expand(3, 4, 2)
        torch.testing.assert_close(out[0, :, :, 2:], expected_right)

    def test_multiple_coalitions_handled_independently(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalitions = torch.tensor(
            [
                [True, True],
                [False, False],
                [True, False],
                [False, True],
            ]
        )
        out = strategy.apply(image, half_masks, coalitions)
        assert out.shape == (4, 3, 4, 4)
        torch.testing.assert_close(out[0], image)
        torch.testing.assert_close(out[2, :, :, :2], image[:, :, :2])
        torch.testing.assert_close(out[3, :, :, 2:], image[:, :, 2:])


class TestZeroMasking:
    def test_is_cnn_masking_strategy(self) -> None:
        assert isinstance(ZeroMasking(), PixelBasedMaskingStrategy)

    def test_default_value_is_zero(self, image, half_masks) -> None:
        strategy = ZeroMasking()
        coalition = torch.tensor([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        assert (out[0] == 0).all()

    def test_custom_value(self, image, half_masks) -> None:
        strategy = ZeroMasking(value=7.0)
        coalition = torch.tensor([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        assert (out[0] == 7.0).all()

    def test_partial_coalition_zeros_only_absent(self, image, half_masks) -> None:
        strategy = ZeroMasking()
        coalition = torch.tensor([[False, True]])
        out = strategy.apply(image, half_masks, coalition)
        # Left half (absent) zeroed; right half (present) preserved.
        assert (out[0, :, :, :2] == 0).all()
        torch.testing.assert_close(out[0, :, :, 2:], image[:, :, 2:])

    def test_full_coalition_preserves_image(self, image, half_masks) -> None:
        strategy = ZeroMasking()
        coalition = torch.tensor([[True, True]])
        out = strategy.apply(image, half_masks, coalition)
        torch.testing.assert_close(out[0], image)


def test_cnn_masking_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        PixelBasedMaskingStrategy()  # type: ignore[abstract]


def test_transformer_masking_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        LatentBasedMaskingStrategy()  # type: ignore[abstract]


@pytest.fixture
def token_masks() -> torch.Tensor:
    """Four players owning one token each (flat token indices 0..3)."""
    return torch.tensor([[0], [1], [2], [3]])


class TestBoolMaskedPosStrategy:
    def test_is_transformer_masking_strategy(self) -> None:
        assert isinstance(BoolMaskedPosStrategy(), LatentBasedMaskingStrategy)

    def test_all_present_coalition_masks_nothing(self, token_masks) -> None:
        strategy = BoolMaskedPosStrategy()
        coalitions = torch.tensor([[True, True, True, True]])
        out = strategy.apply(coalitions, token_masks)
        assert out.shape == (1, 4)
        assert out.dtype == torch.bool
        # True == masked; all present means nothing masked.
        assert not out.any()

    def test_empty_coalition_masks_everything(self, token_masks) -> None:
        strategy = BoolMaskedPosStrategy()
        coalitions = torch.tensor([[False, False, False, False]])
        out = strategy.apply(coalitions, token_masks)
        assert out.all()

    def test_single_player_present_unmasks_only_its_token(self, token_masks) -> None:
        strategy = BoolMaskedPosStrategy()
        coalitions = torch.tensor([[True, False, False, False]])
        out = strategy.apply(coalitions, token_masks)
        # Token 0 visible (False), the rest masked (True).
        assert not out[0, 0]
        assert out[0, 1:].all()


class TestMaskTokenStrategy:
    def test_is_transformer_masking_strategy(self) -> None:
        assert isinstance(MaskTokenStrategy(MockViT()), LatentBasedMaskingStrategy)

    def test_apply_returns_token_mask(self, token_masks) -> None:
        strategy = MaskTokenStrategy(MockViT())
        coalitions = torch.tensor([[True, False, False, False]])
        out = strategy.apply(coalitions, token_masks)
        assert out.shape == (1, 4)
        assert not out[0, 0]
        assert out[0, 1:].all()

    def test_apply_zeros_mask_token(self, token_masks) -> None:
        model = MockViT()
        strategy = MaskTokenStrategy(model)
        # Starts non-zero.
        assert not torch.allclose(model.vit.embeddings.mask_token.data, torch.zeros(1, 1, 4))
        coalitions = torch.tensor([[True, True, True, True]])
        strategy.apply(coalitions, token_masks)
        assert torch.allclose(model.vit.embeddings.mask_token.data, torch.zeros(1, 1, 4))

    def test_mask_token_sized_from_config_hidden_size(self, token_masks) -> None:
        """The replacement mask token is shaped from ``config.hidden_size``, not the old token."""
        model = MockViT(hidden_size=16)
        MaskTokenStrategy(model).apply(torch.tensor([[True, True, True, True]]), token_masks)
        assert model.vit.embeddings.mask_token.shape == (1, 1, 16)


class TestMaskingStrategyModelValidation:
    """``validate_model`` guards the model attributes each token masker depends on."""

    def test_mask_token_strategy_rejects_non_callable_model(self) -> None:
        model = SimpleNamespace(config=make_vit_config(), vit=SimpleNamespace(embeddings=None))
        with pytest.raises(TypeError, match="VisionModel"):
            MaskTokenStrategy(model)

    def test_mask_token_strategy_rejects_model_without_mask_token(self) -> None:
        model = MockViT()
        del model.vit
        with pytest.raises(TypeError, match=re.escape("vit.embeddings.mask_token")):
            MaskTokenStrategy(model)

    def test_mask_token_strategy_rejects_model_without_hidden_size(self) -> None:
        model = MockViT()
        model.config.hidden_size = None
        with pytest.raises(TypeError, match="hidden_size"):
            MaskTokenStrategy(model)

    def test_mask_token_strategy_error_points_to_bool_masked_pos_strategy(self) -> None:
        """The hidden_size error names the fallback so users know what to switch to."""
        model = MockViT()
        model.config.hidden_size = None
        with pytest.raises(TypeError, match="BoolMaskedPosStrategy"):
            MaskTokenStrategy(model)

    def test_bool_masked_pos_accepts_model_with_mask_token(self) -> None:
        BoolMaskedPosStrategy.validate_model(MockViT())  # does not raise

    def test_bool_masked_pos_rejects_model_without_mask_token(self) -> None:
        model = MockViT()
        del model.vit
        with pytest.raises(TypeError, match=re.escape("vit.embeddings.mask_token")):
            BoolMaskedPosStrategy.validate_model(model)

    def test_bool_masked_pos_rejects_unset_mask_token(self) -> None:
        """``use_mask_token=False`` models leave ``mask_token`` as None and must be rejected."""
        model = MockViT()
        model.vit.embeddings.mask_token = None
        with pytest.raises(TypeError, match="use_mask_token=True"):
            BoolMaskedPosStrategy.validate_model(model)

    def test_bool_masked_pos_unset_mask_token_error_suggests_mask_token_strategy(self) -> None:
        model = MockViT()
        model.vit.embeddings.mask_token = None
        with pytest.raises(TypeError, match="MaskTokenStrategy"):
            BoolMaskedPosStrategy.validate_model(model)

    def test_bool_masked_pos_rejects_non_callable_model(self) -> None:
        model = SimpleNamespace(
            vit=SimpleNamespace(embeddings=SimpleNamespace(mask_token=torch.zeros(1, 1, 4)))
        )
        with pytest.raises(TypeError, match="VisionModel"):
            BoolMaskedPosStrategy.validate_model(model)


class TestCoalitionDomains:
    """Each masker declares the coalition domain it accepts, which the architecture cross-checks."""

    @pytest.mark.parametrize("strategy", [MeanColorMasking(), ZeroMasking()])
    def test_pixel_maskers_accept_pixel_domain(self, strategy) -> None:
        assert strategy.accepted_coalition_domain is CoalitionDomain.PIXEL

    def test_token_maskers_accept_token_domain(self) -> None:
        assert BoolMaskedPosStrategy().accepted_coalition_domain is CoalitionDomain.TOKEN
        assert MaskTokenStrategy(MockViT()).accepted_coalition_domain is CoalitionDomain.TOKEN
