"""End-to-end tests for ``shapiq.vision.explainer.ImageExplainer``."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import Explainer
from shapiq.explainer.utils import get_explainers
from shapiq.interaction_values import InteractionValues
from shapiq.vision import ImageExplainer
from shapiq.vision.architecture import ResNetArchitecture
from shapiq.vision.masking import ZeroMasking

from .conftest import FixedMasksStrategy, make_linear_pixel_model


def _arch_and_strategy(image, masks, weights):
    arch = ResNetArchitecture(
        model=make_linear_pixel_model(weights),
        masking_strategy=ZeroMasking(),
    )
    return arch, FixedMasksStrategy(masks)


class TestImageExplainer:
    def test_explainer_returns_interaction_values(self, tiny_image, two_player_masks) -> None:
        arch, strategy = _arch_and_strategy(tiny_image, two_player_masks, np.ones((4, 4)))
        explainer = ImageExplainer(
            architecture=arch,
            data=tiny_image,
            player_strategy=strategy,
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
        arch, strategy = _arch_and_strategy(tiny_image, two_player_masks, np.ones((4, 4)))
        explainer = ImageExplainer(
            architecture=arch,
            data=tiny_image,
            player_strategy=strategy,
            random_state=0,
        )
        assert explainer.baseline_value == pytest.approx(explainer.imputer.empty_prediction)

    def test_explainer_three_players_e2e(self, three_player_masks) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(6, 6, 3)).astype(np.float64)
        arch, strategy = _arch_and_strategy(image, three_player_masks, np.ones((6, 6)))
        explainer = ImageExplainer(
            architecture=arch,
            data=image,
            player_strategy=strategy,
            index="k-SII",
            max_order=2,
            random_state=42,
        )
        result = explainer.explain_function(image, budget=32)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 3
        # Every value should be a finite number.
        assert np.isfinite(result.values).all()


class TestExplainerAutoDispatch:
    def test_get_explainers_includes_vision(self) -> None:
        assert "vision" in get_explainers()
        assert get_explainers()["vision"] is ImageExplainer

    def test_explainer_dispatches_to_image_explainer(self, tiny_image, two_player_masks) -> None:
        arch, strategy = _arch_and_strategy(tiny_image, two_player_masks, np.ones((4, 4)))
        explainer = Explainer(
            model=arch.model,
            data=tiny_image,
            player_strategy=strategy,
            index="k-SII",
            max_order=2,
            random_state=0,
        )
        assert isinstance(explainer, ImageExplainer)
        result = explainer.explain_function(tiny_image, budget=16)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 2

    def test_explainer_accepts_architecture_strategy_as_model(
        self, tiny_image, two_player_masks
    ) -> None:
        arch, strategy = _arch_and_strategy(tiny_image, two_player_masks, np.ones((4, 4)))
        explainer = Explainer(
            model=arch,
            data=tiny_image,
            player_strategy=strategy,
            random_state=0,
        )
        assert isinstance(explainer, ImageExplainer)
        assert explainer.imputer.architecture is arch
