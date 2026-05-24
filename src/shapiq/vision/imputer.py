import numpy as np

from shapiq.imputer.base import Imputer

from .architecture import ModelArchitectureStrategy
from .players import PlayerStrategy


class ImageImputer(Imputer):
    """
    Imputer for images: creates masked versions of the input image based on player coalitions and returns model predictions.    
    """
    def __init__(
        self,
        architecture: ModelArchitectureStrategy,
        image: np.ndarray,
        player_strategy: PlayerStrategy | None = None,
        masking_strategy=None,
        normalize: bool = True,
    ):
        self.image = image
        self.architecture = architecture

        player_strategy = player_strategy or architecture.default_player_strategy()
        if masking_strategy is not None:
            architecture._masking_strategy = masking_strategy

        architecture.prepare(image, player_strategy)
        self._player_strategy = player_strategy

        dummy_data = np.zeros((1, player_strategy.n_players))
        super().__init__(model=architecture.model, data=dummy_data)

        self.empty_prediction = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """
        Calculates the value function for a batch of coalitions.
        
        Args:
            coalitions: (n_coalitions, n_players) boolean array
            
        Returns:
            (n_coalitions,) float array with model-Predictions
        
        """
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        return self.architecture.value_function(self.image, coalitions)

    def calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction of the model provided only missing features.

        """
        return self.architecture.calc_empty_prediction(self.image)

    @property
    def player_masks(self) -> np.ndarray | None:
        """Spatial masks per player, shape (n_players, H, W). None for latent-space architectures."""
        return getattr(self.architecture, "_player_masks", None)
