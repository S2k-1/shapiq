"""Image imputer for vision-based Shapley value explanations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.imputer.base import Imputer

from .utils import AutoBatchSize, ImageLike, as_hwc_array, resolve_batch_size

if TYPE_CHECKING:
    from .architecture import ModelArchitectureStrategy
    from .masking import LatentMaskingStrategy, PixelMaskingStrategy
    from .players import PlayerStrategy


def _validate_strategies(
    architecture: ModelArchitectureStrategy,
    player_strategy: PlayerStrategy,
    masking_strategy: PixelMaskingStrategy | LatentMaskingStrategy | None,
) -> None:
    """Ensure player and masking strategies match the architecture family."""
    from .architecture import (
        CLIPArchitecture,
        CustomViTArchitecture,
        DINOv2Architecture,
        HuggingFacePixelArchitecture,
        LayerMaskedCNNArchitecture,
        ResNetArchitecture,
        ViTArchitecture,
    )
    from .masking import LatentMaskingStrategy, ManifoldMaskingStrategy, PixelMaskingStrategy
    from .players import LatentPlayerStrategy, PixelPlayerStrategy

    pixel_architectures = (
        ResNetArchitecture,
        HuggingFacePixelArchitecture,
        DINOv2Architecture,
        CLIPArchitecture,
        LayerMaskedCNNArchitecture,
    )
    latent_architectures = (ViTArchitecture, CustomViTArchitecture)
    active_mask = (
        masking_strategy if masking_strategy is not None else architecture.masking_strategy
    )

    if isinstance(architecture, pixel_architectures) and not isinstance(
        player_strategy, PixelPlayerStrategy
    ):
        msg = (
            f"{type(architecture).__name__} requires a PixelPlayerStrategy "
            f"(got {type(player_strategy).__name__})."
        )
        raise TypeError(msg)
    if isinstance(architecture, latent_architectures) and not isinstance(
        player_strategy, LatentPlayerStrategy
    ):
        msg = (
            f"{type(architecture).__name__} requires a LatentPlayerStrategy "
            f"(got {type(player_strategy).__name__})."
        )
        raise TypeError(msg)

    if isinstance(architecture, LayerMaskedCNNArchitecture):
        if not isinstance(active_mask, ManifoldMaskingStrategy):
            msg = (
                f"LayerMaskedCNNArchitecture requires a ManifoldMaskingStrategy "
                f"(got {type(active_mask).__name__})."
            )
            raise TypeError(msg)
    elif isinstance(architecture, pixel_architectures) and not isinstance(
        active_mask, PixelMaskingStrategy
    ):
        msg = (
            f"{type(architecture).__name__} requires a PixelMaskingStrategy "
            f"(got {type(active_mask).__name__})."
        )
        raise TypeError(msg)
    elif isinstance(architecture, latent_architectures) and not isinstance(
        active_mask, LatentMaskingStrategy
    ):
        msg = (
            f"{type(architecture).__name__} requires a LatentMaskingStrategy "
            f"(got {type(active_mask).__name__})."
        )
        raise TypeError(msg)


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
        batch_size: Maximum number of coalitions per forward pass. ``"auto"`` (default)
            picks a hardware-aware batch size. ``None`` evaluates all coalitions at once.
    """

    def __init__(
        self,
        architecture: ModelArchitectureStrategy,
        image: ImageLike,
        player_strategy: PlayerStrategy | None = None,
        masking_strategy: PixelMaskingStrategy | LatentMaskingStrategy | None = None,
        *,
        normalize: bool = True,
        batch_size: AutoBatchSize = "auto",
    ) -> None:
        """Initialize the ImageImputer."""
        self.image = as_hwc_array(image)
        self.architecture = architecture

        player_strategy = player_strategy or architecture.default_player_strategy()
        if masking_strategy is not None:
            architecture.masking_strategy = masking_strategy

        _validate_strategies(architecture, player_strategy, masking_strategy)

        # prepare() may update player counts (e.g. SLIC segment count).
        architecture.prepare(self.image, player_strategy)
        self._player_strategy = player_strategy

        self.batch_size = resolve_batch_size(
            batch_size, architecture, self.image, player_strategy.n_players
        )

        # Satisfy Imputer base ``n_features`` contract; coalitions are boolean masks.
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
        """Run the model on the empty coalition to get the empty prediction."""
        return self.architecture.calc_empty_prediction(self.image)

    @property
    def player_masks(self) -> np.ndarray | None:
        """Spatial masks per player, shape ``(n_players, H, W)``."""
        return self.architecture.player_masks
