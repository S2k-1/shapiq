"""Explainer for vision models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.explainer.base import Explainer
from shapiq.explainer.configuration import setup_approximator
from shapiq.explainer.custom_types import ExplainerIndices
from shapiq.game_theory.indices import is_empty_value_the_baseline

from .imputer import ImageImputer

if TYPE_CHECKING:
    import numpy as np

    from shapiq.interaction_values import InteractionValues

    from .architecture import ModelArchitectureStrategy
    from .masking import LatentMaskingStrategy, PixelMaskingStrategy
    from .players import PlayerStrategy

ImageExplainerIndices = ExplainerIndices


class ImageExplainer(Explainer):
    """Explainer for vision models.

    Delegates all model-specific logic to a :class:`ModelArchitectureStrategy`.

    Args:
        architecture: The model architecture strategy.
        data: The ``(H, W, C)`` image array to explain.
        player_strategy: Player partitioning strategy. Defaults to the architecture's default.
        masking_strategy: Masking strategy for absent players. Defaults to the architecture's
            default.
        index: Interaction index to compute. Defaults to ``"k-SII"``.
        max_order: Maximum interaction order. Defaults to ``2``.
        random_state: Optional random seed for reproducibility.
        batch_size: Maximum coalitions per forward pass. Evaluates all at once if ``None``.
    """

    def __init__(
        self,
        architecture: ModelArchitectureStrategy,
        data: np.ndarray | None = None,
        *,
        player_strategy: PlayerStrategy | None = None,
        masking_strategy: PixelMaskingStrategy | LatentMaskingStrategy | None = None,
        index: ImageExplainerIndices = "k-SII",
        max_order: int = 2,
        random_state: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the ImageExplainer.

        Args:
            architecture: The model architecture strategy.
            data: The ``(H, W, C)`` image array to explain.
            player_strategy: Player partitioning strategy. Defaults to the architecture's default.
            masking_strategy: Masking strategy for absent players. Defaults to the architecture's
                default.
            index: Interaction index to compute. Defaults to ``"k-SII"``.
            max_order: Maximum interaction order. Defaults to ``2``.
            random_state: Optional random seed for reproducibility.
            batch_size: Maximum coalitions per forward pass. Evaluates all at once if ``None``.
        """
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

    def explain_function(self, _x: np.ndarray | None, *, budget: int = 64) -> InteractionValues:
        """Compute interaction values for the image.

        Args:
            _x: Unused; the image was provided at construction time.
            budget: Number of model evaluations (coalitions) to use.

        Returns:
            The computed interaction values.
        """
        interaction_values = self.approximator.approximate(budget=budget, game=self.imputer)
        interaction_values.baseline_value = self.baseline_value
        if is_empty_value_the_baseline(interaction_values.index):
            interaction_values[()] = interaction_values.baseline_value
        return interaction_values

    @property
    def baseline_value(self) -> float:
        """Return the empty-coalition prediction as the baseline value."""
        return self.imputer.empty_prediction
