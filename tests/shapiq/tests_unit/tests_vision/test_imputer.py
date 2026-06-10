"""Tests for ``shapiq.vision.imputer.ImageImputer``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from shapiq.game_theory.exact import ExactComputer
from shapiq.imputer.base import Imputer
from shapiq.vision.architecture import CNNArchitecture
from shapiq.vision.imputer import ImageImputer
from shapiq.vision.masking import MeanColorMasking, ZeroMasking

from .conftest import ChannelSumModel, FixedMasksStrategy


def _build_imputer(image, masks, masking_strategy, *, normalize=True, batch_size=32):
    arch = CNNArchitecture(
        model=ChannelSumModel(),
        masking_strategy=masking_strategy,
        player_strategy=FixedMasksStrategy(masks),
    )
    return ImageImputer(
        model_architecture=arch,
        image=image,
        normalize=normalize,
        batch_size=batch_size,
    )


class TestImageImputerBasics:
    def test_is_imputer_subclass(self, tiny_image, two_player_masks) -> None:
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking())
        assert isinstance(imputer, Imputer)

    def test_n_players_matches_player_strategy(self, three_player_masks) -> None:
        image = np.random.default_rng(0).integers(0, 255, size=(6, 6, 3)).astype(np.float64)
        imputer = _build_imputer(image, three_player_masks, ZeroMasking())
        assert imputer.n_players == 3
        assert imputer.n_features == 3

    def test_player_masks_property_exposes_spatial_masks(
        self, tiny_image, two_player_masks
    ) -> None:
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking())
        assert imputer.player_masks is not None
        np.testing.assert_array_equal(imputer.player_masks, two_player_masks)

    def test_empty_prediction_with_zero_masking_is_zero(self, tiny_image, two_player_masks) -> None:
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking(), normalize=False)
        assert imputer.empty_prediction == pytest.approx(0.0)

    def test_normalize_sets_normalization_value(self, tiny_image, two_player_masks) -> None:
        imputer = _build_imputer(tiny_image, two_player_masks, MeanColorMasking(), normalize=True)
        assert imputer.normalization_value == pytest.approx(imputer.empty_prediction)

    def test_value_function_accepts_1d_coalition(self, tiny_image, two_player_masks) -> None:
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking(), normalize=False)
        out = np.atleast_1d(imputer.value_function(np.array([True, True])))
        assert out.shape[0] == 1


class TestImageImputerValues:
    def test_value_function_recovers_linear_model_output(
        self, tiny_image, two_player_masks
    ) -> None:
        """With ZeroMasking + the channel-sum model, a coalition's value equals the
        sum of pixel intensities restricted to the present players."""
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking(), normalize=False)
        coalitions = np.array(
            [
                [False, False],
                [True, False],
                [False, True],
                [True, True],
            ],
        )
        values = np.atleast_1d(imputer.value_function(coalitions))

        v_empty = 0.0
        v_left = tiny_image[:, :2].sum()
        v_right = tiny_image[:, 2:].sum()
        v_full = tiny_image.sum()
        np.testing.assert_allclose(values, [v_empty, v_left, v_right, v_full])

    def test_correctness_against_exact_computer(self, tiny_image, two_player_masks) -> None:
        """Shapley values of the imputer-induced game match the analytic linear values."""
        imputer = _build_imputer(tiny_image, two_player_masks, ZeroMasking(), normalize=False)
        ec = ExactComputer(n_players=imputer.n_players, game=imputer)
        sv = ec.probabilistic_value(index="SV")

        sv_left = tiny_image[:, :2].sum()
        sv_right = tiny_image[:, 2:].sum()
        assert sv[(0,)] == pytest.approx(sv_left)
        assert sv[(1,)] == pytest.approx(sv_right)

    def test_correctness_three_players(self, three_player_masks) -> None:
        rng = np.random.default_rng(7)
        image = rng.integers(0, 255, size=(6, 6, 3)).astype(np.float64)
        imputer = _build_imputer(image, three_player_masks, ZeroMasking(), normalize=False)
        ec = ExactComputer(n_players=imputer.n_players, game=imputer)
        sv = ec.probabilistic_value(index="SV")

        regions = [
            image[:, 0:2].sum(),
            image[:, 2:4].sum(),
            image[:, 4:6].sum(),
        ]
        for i, expected in enumerate(regions):
            assert sv[(i,)] == pytest.approx(expected)

    def test_call_subtracts_normalization_value(self, tiny_image, two_player_masks) -> None:
        """Calling the imputer as a Game subtracts the normalization value."""
        imputer = _build_imputer(tiny_image, two_player_masks, MeanColorMasking(), normalize=True)
        coalitions = np.array([[False, False], [True, True]])
        out = imputer(coalitions)
        assert out[0] == pytest.approx(0.0, abs=1e-8)
        expected_full = tiny_image.sum() - imputer.empty_prediction
        assert out[1] == pytest.approx(expected_full)


class TestImageImputerInputFormats:
    @pytest.mark.parametrize(
        "image_input",
        [
            pytest.param(lambda img: img, id="numpy"),
            pytest.param(lambda img: Image.fromarray(img.astype(np.uint8)), id="pil"),
            pytest.param(lambda img: torch.from_numpy(img).permute(2, 0, 1), id="torch_chw"),
        ],
    )
    def test_accepts_common_image_formats(self, tiny_image, two_player_masks, image_input) -> None:
        arch = CNNArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        imputer = ImageImputer(
            model_architecture=arch,
            image=image_input(tiny_image),
            normalize=False,
        )
        values = imputer.value_function(np.array([[True, True]]))
        assert values.shape == (1,)
        assert np.isfinite(values).all()
