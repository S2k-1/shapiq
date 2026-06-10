"""End-to-end tests for ``shapiq.vision.explainer.ImageExplainer``."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.vision import ImageExplainer
from shapiq.vision.architecture import CNNArchitecture
from shapiq.vision.masking import ZeroMasking

from .conftest import ChannelSumModel, FixedMasksStrategy


def _build_arch(masks):
    return CNNArchitecture(
        model=ChannelSumModel(),
        masking_strategy=ZeroMasking(),
        player_strategy=FixedMasksStrategy(masks),
    )


class TestImageExplainer:
    def test_explainer_returns_interaction_values(self, tiny_image, two_player_masks) -> None:
        explainer = ImageExplainer(
            model_architecture=_build_arch(two_player_masks),
            data=tiny_image,
            index="k-SII",
            max_order=2,
            random_state=0,
        )
        result = explainer.explain_function(tiny_image, budget=16)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 2

    def test_explainer_baseline_matches_empty_prediction(
        self, tiny_image, two_player_masks
    ) -> None:
        explainer = ImageExplainer(
            model_architecture=_build_arch(two_player_masks),
            data=tiny_image,
            random_state=0,
        )
        assert explainer.baseline_value == pytest.approx(explainer._imputer.empty_prediction)

    def test_explainer_three_players_e2e(self, three_player_masks) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(6, 6, 3)).astype(np.float64)
        explainer = ImageExplainer(
            model_architecture=_build_arch(three_player_masks),
            data=image,
            index="k-SII",
            max_order=2,
            random_state=42,
        )
        result = explainer.explain_function(image, budget=32)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 3
        assert np.isfinite(result.values).all()
