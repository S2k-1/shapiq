from __future__ import annotations

from typing import Any

import numpy as np

from shapiq.explainer.base import Explainer
from shapiq.explainer.configuration import setup_approximator
from shapiq.explainer.custom_types import ExplainerIndices
from shapiq.game_theory.indices import is_empty_value_the_baseline
from shapiq.interaction_values import InteractionValues

from .architecture import ModelArchitectureStrategy
from .imputer import ImageImputer
from .players import PlayerStrategy

ImageExplainerIndices = ExplainerIndices


class ImageExplainer(Explainer):
    """Explainer for vision models. Delegates all model-specific logic to a ModelArchitectureStrategy."""

    def __init__(
        self,
        architecture: ModelArchitectureStrategy,
        data: np.ndarray | None = None,
        *,
        player_strategy: PlayerStrategy | None = None,
        masking_strategy=None,
        index: ImageExplainerIndices = "k-SII",
        max_order: int = 2,
        random_state: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        if data is None:
            msg = (
                "ImageExplainer requires the image to explain at construction time. "
                "Pass it as `data=your_image`."
            )
            raise ValueError(msg)

        super().__init__(model=architecture.model, index=index, max_order=max_order)

        self._imputer = ImageImputer(
            architecture=architecture,
            image=data,
            player_strategy=player_strategy,
            masking_strategy=masking_strategy,
            batch_size=batch_size,
        )

        self._approximator = setup_approximator(
            approximator="auto",
            index=index,
            max_order=self.max_order,
            n_players=self._imputer.n_players,
            random_state=random_state,
        )

    def explain_function(self, x: np.ndarray | None, *, budget: int = 64) -> InteractionValues:
        """Compute interaction values for the image fixed at construction time.

        Args:
            x: Accepted for API compatibility with the base ``Explainer`` interface but
                **not used**. The image to explain is the one passed as ``data`` to
                ``ImageExplainer.__init__``. Passing a different array here has no effect.
            budget: Number of model evaluations available to the approximator.

        Returns:
            ``InteractionValues`` for the image supplied at construction.
        """
        interaction_values = self.approximator.approximate(budget=budget, game=self.imputer)
        interaction_values.baseline_value = self.baseline_value
        if is_empty_value_the_baseline(interaction_values.index):
            interaction_values[()] = interaction_values.baseline_value
        return interaction_values

    @property
    def baseline_value(self) -> float:
        return self.imputer.empty_prediction
