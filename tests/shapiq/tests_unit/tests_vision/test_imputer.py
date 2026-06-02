"""Tests for ``shapiq.vision.imputer.ImageImputer``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from shapiq.game_theory.exact import ExactComputer
from shapiq.imputer.base import Imputer
from shapiq.vision.architecture import ResNetArchitecture
from shapiq.vision.imputer import ImageImputer
from shapiq.vision.masking import MeanColorMasking, ZeroMasking

from .conftest import FixedMasksStrategy, make_linear_pixel_model


def _build_imputer(image, masks, weights, masking_strategy, *, normalize=True):
    arch = ResNetArchitecture(
        model=make_linear_pixel_model(weights),
        masking_strategy=masking_strategy,
    )
    return ImageImputer(
        architecture=arch,
        image=image,
        player_strategy=FixedMasksStrategy(masks),
        normalize=normalize,
    )


class TestImageImputerBasics:
    def test_is_imputer_subclass(self, tiny_image, two_player_masks) -> None:
        imputer = _build_imputer(tiny_image, two_player_masks, np.ones((4, 4)), ZeroMasking())
        assert isinstance(imputer, Imputer)

    def test_n_players_matches_player_strategy(self, tiny_image, three_player_masks) -> None:
        image = np.random.default_rng(0).integers(0, 255, size=(6, 6, 3)).astype(np.float64)
        imputer = _build_imputer(image, three_player_masks, np.ones((6, 6)), ZeroMasking())
        assert imputer.n_players == 3
        assert imputer.n_features == 3

    def test_player_masks_property_exposes_spatial_masks(
        self, tiny_image, two_player_masks
    ) -> None:
        imputer = _build_imputer(tiny_image, two_player_masks, np.ones((4, 4)), ZeroMasking())
        assert imputer.player_masks is not None
        np.testing.assert_array_equal(imputer.player_masks, two_player_masks)

    def test_empty_prediction_with_zero_masking_is_zero(self, tiny_image, two_player_masks) -> None:
        imputer = _build_imputer(
            tiny_image, two_player_masks, np.ones((4, 4)), ZeroMasking(), normalize=False
        )
        assert imputer.empty_prediction == pytest.approx(0.0)

    def test_normalize_sets_normalization_value(self, tiny_image, two_player_masks) -> None:
        imputer = _build_imputer(
            tiny_image, two_player_masks, np.ones((4, 4)), MeanColorMasking(), normalize=True
        )
        assert imputer.normalization_value == pytest.approx(imputer.empty_prediction)

    def test_value_function_accepts_1d_coalition(self, tiny_image, two_player_masks) -> None:
        imputer = _build_imputer(
            tiny_image, two_player_masks, np.ones((4, 4)), ZeroMasking(), normalize=False
        )
        out = imputer.value_function(np.array([True, True]))
        # Should not raise; should return at least one value.
        out_arr = np.atleast_1d(out)
        assert out_arr.shape[0] == 1


class TestImageImputerValues:
    def test_value_function_recovers_linear_model_output(
        self, tiny_image, two_player_masks
    ) -> None:
        """With ZeroMasking + linear model, the value for a coalition equals the sum
        of weighted pixel intensities restricted to present players."""
        weights = np.ones((4, 4))
        imputer = _build_imputer(
            tiny_image, two_player_masks, weights, ZeroMasking(), normalize=False
        )
        coalitions = np.array(
            [
                [False, False],
                [True, False],
                [False, True],
                [True, True],
            ],
        )
        values = imputer.value_function(coalitions)
        values = np.atleast_1d(values)

        # Expected: sum over the kept regions (weights all 1, sum over channels).
        per_pixel = tiny_image.sum(axis=2)  # (H, W)
        v_empty = 0.0
        v_left = per_pixel[:, :2].sum()
        v_right = per_pixel[:, 2:].sum()
        v_full = per_pixel.sum()
        np.testing.assert_allclose(values, [v_empty, v_left, v_right, v_full])

    def test_correctness_against_exact_computer(self, tiny_image, two_player_masks) -> None:
        """Shapley values from the imputer-induced game match ExactComputer on a tiny image."""
        weights = np.ones((4, 4))
        imputer = _build_imputer(
            tiny_image, two_player_masks, weights, ZeroMasking(), normalize=False
        )
        ec = ExactComputer(n_players=imputer.n_players, game=imputer)
        sv = ec.probabilistic_value(index="SV")

        # Analytic Shapley value of a linear additive game with two disjoint regions:
        # the value for each player is simply the contribution of its own region.
        per_pixel = tiny_image.sum(axis=2)
        sv_left = per_pixel[:, :2].sum()
        sv_right = per_pixel[:, 2:].sum()

        assert sv[(0,)] == pytest.approx(sv_left)
        assert sv[(1,)] == pytest.approx(sv_right)

    def test_correctness_three_players(self, three_player_masks) -> None:
        rng = np.random.default_rng(7)
        image = rng.integers(0, 255, size=(6, 6, 3)).astype(np.float64)
        weights = np.ones((6, 6))
        imputer = _build_imputer(image, three_player_masks, weights, ZeroMasking(), normalize=False)
        ec = ExactComputer(n_players=imputer.n_players, game=imputer)
        sv = ec.probabilistic_value(index="SV")

        per_pixel = image.sum(axis=2)
        regions = [
            per_pixel[:, 0:2].sum(),
            per_pixel[:, 2:4].sum(),
            per_pixel[:, 4:6].sum(),
        ]
        for i, expected in enumerate(regions):
            assert sv[(i,)] == pytest.approx(expected)

    def test_call_subtracts_normalization_value(self, tiny_image, two_player_masks) -> None:
        """Calling the imputer as a Game subtracts the normalization value."""
        weights = np.ones((4, 4))
        imputer = _build_imputer(
            tiny_image, two_player_masks, weights, MeanColorMasking(), normalize=True
        )
        # Batch with the empty coalition and a non-empty one. After normalization,
        # the empty coalition's value must be zero.
        coalitions = np.array([[False, False], [True, True]])
        out = imputer(coalitions)
        assert out[0] == pytest.approx(0.0, abs=1e-8)
        # The full-coalition (no masking) prediction equals the linear model output on
        # the original image minus the normalization value.
        expected_full = tiny_image.sum() - imputer.empty_prediction
        assert out[1] == pytest.approx(expected_full)


class TestImageImputerInputFormats:
    @pytest.mark.parametrize(
        "image_input",
        [
            pytest.param(lambda img: img, id="numpy"),
            pytest.param(lambda img: Image.fromarray(img.astype(np.uint8)), id="pil"),
            pytest.param(
                lambda img: torch.from_numpy(img).permute(2, 0, 1),
                id="torch_chw",
            ),
        ],
    )
    def test_accepts_common_image_formats(self, tiny_image, two_player_masks, image_input) -> None:
        arch = ResNetArchitecture(
            model=lambda batch: np.mean(batch, axis=(1, 2, 3)),
            masking_strategy=ZeroMasking(),
        )
        imputer = ImageImputer(
            architecture=arch,
            image=image_input(tiny_image),
            player_strategy=FixedMasksStrategy(two_player_masks),
            normalize=False,
        )
        values = imputer.value_function(np.array([[True, True]]))
        assert values.shape == (1,)
        np.testing.assert_allclose(values[0], tiny_image.mean())
