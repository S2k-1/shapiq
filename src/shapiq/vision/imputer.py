"""Image imputer for vision-based Shapley value explanations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.imputer.base import Imputer

from .utils import ImageLike, as_hwc_array

if TYPE_CHECKING:
    from .architecture import ModelArchitectureStrategy
    from .masking import LatentMaskingStrategy, PixelMaskingStrategy
    from .players import PlayerStrategy


class ImageImputer(Imputer):
    """Imputer for images.

    Creates masked versions of the input image based on player coalitions and
    returns model predictions.

    Args:
        architecture: The model architecture strategy that handles model-specific inference.
        image: Image to explain as a ``(H, W, C)`` numpy array, PIL image, or tensor.
        player_strategy: Strategy for splitting the image into players. Defaults to the
            architecture's default.
        masking_strategy: Strategy for masking absent players. Defaults to the architecture's
            default.
        normalize: Whether to normalize predictions by subtracting the empty-coalition prediction.
        batch_size: Maximum number of coalitions per forward pass. Evaluates all at once if
            ``None``.
    """

    def __init__(
        self,
        architecture: ModelArchitectureStrategy,
        image: ImageLike,
        player_strategy: PlayerStrategy | None = None,
        masking_strategy: PixelMaskingStrategy | LatentMaskingStrategy | None = None,
        *,
        normalize: bool = True,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the ImageImputer.

        Args:
            architecture: The model architecture strategy.
            image: Image to explain as a ``(H, W, C)`` numpy array, PIL image, or tensor.
            player_strategy: Player partitioning strategy. Defaults to the architecture's default.
            masking_strategy: Masking strategy for absent players. Defaults to the architecture's
                default.
            normalize: Normalize predictions by subtracting the empty-coalition baseline.
            batch_size: Maximum coalitions per forward pass. Evaluates all at once if ``None``.
        """
        self.image = as_hwc_array(image)
        self.architecture = architecture
        self.batch_size = batch_size

        player_strategy = player_strategy or architecture.default_player_strategy()
        if masking_strategy is not None:
            architecture.masking_strategy = masking_strategy

        architecture.prepare(self.image, player_strategy)
        self._player_strategy = player_strategy

        dummy_data = np.zeros((1, player_strategy.n_players))
        super().__init__(model=architecture.model, data=dummy_data)

        self.empty_prediction = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Calculate the value function for a batch of coalitions.

        Args:
            coalitions: ``(n_coalitions, n_players)`` boolean array.

        Returns:
            ``(n_coalitions,)`` float array with model predictions.
        """
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        n = coalitions.shape[0]
        if self.batch_size is None or n <= self.batch_size:
            return np.atleast_1d(self.architecture.value_function(self.image, coalitions))

        chunks = [
            np.atleast_1d(
                self.architecture.value_function(
                    self.image, coalitions[start : start + self.batch_size]
                )
            )
            for start in range(0, n, self.batch_size)
        ]
        return np.concatenate(chunks, axis=0)

    def calc_empty_prediction(self) -> float:
        """Run the model on the empty coalition to get the empty prediction.

        Returns:
            The model prediction when all features are missing.
        """
        return self.architecture.calc_empty_prediction(self.image)

    @property
    def player_masks(self) -> np.ndarray | None:
        """Spatial masks per player, shape ``(n_players, H, W)``.

        Returns ``None`` for latent-space architectures.
        """
        return getattr(self.architecture, "_player_masks", None)
