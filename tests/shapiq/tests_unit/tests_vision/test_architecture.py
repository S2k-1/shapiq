"""Tests for ``shapiq.vision.architecture``."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.vision.architecture import (
    ModelArchitectureStrategy,
    ResNetArchitecture,
    ViTArchitecture,
    _build_patch_pixel_masks,
)
from shapiq.vision.masking import MeanColorMasking, ZeroMasking
from shapiq.vision.players import PatchStrategy, SuperpixelStrategy

from .conftest import FixedMasksStrategy, make_linear_pixel_model


class TestResNetArchitecture:
    def test_is_architecture_strategy(self) -> None:
        arch = ResNetArchitecture(model=lambda x: np.zeros(x.shape[0]))
        assert isinstance(arch, ModelArchitectureStrategy)

    def test_default_player_strategy(self) -> None:
        arch = ResNetArchitecture(model=lambda x: np.zeros(x.shape[0]))
        strategy = arch.default_player_strategy()
        assert isinstance(strategy, SuperpixelStrategy)
        assert strategy.n_segments == 10

    def test_default_masking_strategy(self) -> None:
        arch = ResNetArchitecture(model=lambda x: np.zeros(x.shape[0]))
        assert isinstance(arch.default_masking_strategy(), MeanColorMasking)

    def test_explicit_masking_strategy_used(self) -> None:
        zero = ZeroMasking()
        arch = ResNetArchitecture(model=lambda x: np.zeros(x.shape[0]), masking_strategy=zero)
        assert arch._masking_strategy is zero

    def test_prepare_caches_player_masks(self, tiny_image, two_player_masks) -> None:
        arch = ResNetArchitecture(model=lambda x: np.zeros(x.shape[0]))
        strategy = FixedMasksStrategy(two_player_masks)
        assert arch._player_masks is None
        arch.prepare(tiny_image, strategy)
        assert arch._player_masks is not None
        np.testing.assert_array_equal(arch._player_masks, two_player_masks)

    def test_value_function_returns_array_for_multiple_coalitions(
        self, tiny_image, two_player_masks
    ) -> None:
        weights = np.ones((4, 4))
        model = make_linear_pixel_model(weights)
        arch = ResNetArchitecture(model=model)
        arch.prepare(tiny_image, FixedMasksStrategy(two_player_masks))
        coalitions = np.array(
            [
                [False, False],
                [True, False],
                [False, True],
                [True, True],
            ],
        )
        out = arch.value_function(tiny_image, coalitions)
        out = np.atleast_1d(out)
        assert out.shape == (4,)
        # Full-coalition prediction equals model on original image.
        np.testing.assert_allclose(out[3], model(tiny_image[np.newaxis])[0])

    def test_calc_empty_prediction_uses_full_masking(self, tiny_image, two_player_masks) -> None:
        weights = np.ones((4, 4))
        model = make_linear_pixel_model(weights)
        arch = ResNetArchitecture(model=model, masking_strategy=ZeroMasking())
        arch.prepare(tiny_image, FixedMasksStrategy(two_player_masks))
        empty = arch.calc_empty_prediction(tiny_image)
        # ZeroMasking of the entire image -> linear model returns 0.
        assert empty == pytest.approx(0.0)

    def test_calc_empty_prediction_with_mean_masking(self, tiny_image, two_player_masks) -> None:
        weights = np.ones((4, 4))
        model = make_linear_pixel_model(weights)
        arch = ResNetArchitecture(model=model, masking_strategy=MeanColorMasking())
        arch.prepare(tiny_image, FixedMasksStrategy(two_player_masks))
        empty = arch.calc_empty_prediction(tiny_image)
        # All pixels become the mean color -> expected = sum(weights) * sum(mean_color).
        mean_color = tiny_image.mean(axis=(0, 1))
        expected = weights.sum() * mean_color.sum()
        assert empty == pytest.approx(expected)


class TestViTArchitectureDefaults:
    """Light-weight tests that don't require a real ViT model."""

    def test_is_architecture_strategy(self) -> None:
        # Construct with stubs since we only test the type and defaults that don't touch the model.
        arch = ViTArchitecture(model=object(), processor=object())
        assert isinstance(arch, ModelArchitectureStrategy)

    def test_default_player_strategy_uses_model_config(self) -> None:
        class _Config:
            image_size = 24
            patch_size = 8

        class _Model:
            config = _Config()

        arch = ViTArchitecture(model=_Model(), processor=object())
        strategy = arch.default_player_strategy()
        assert isinstance(strategy, PatchStrategy)
        assert strategy.grid_size == 3  # 24 // 8
        # n_players defaults to 9.
        assert strategy.n_players == 9

    def test_build_patch_pixel_masks_for_visualization(self) -> None:
        image = np.zeros((224, 224, 3))
        strategy = PatchStrategy(grid_size=14, n_players=9)
        masks = _build_patch_pixel_masks(image, strategy)
        assert masks.shape == (9, 224, 224)
        assert (masks.sum(axis=0) >= 1).all()
