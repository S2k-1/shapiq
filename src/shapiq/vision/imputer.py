from __future__ import annotations

import numpy as np

from shapiq.imputer.base import Imputer
from shapiq.typing import Model

from .architecture import ModelArchitectureStrategy, CNNArchitecture, TransformerArchitecture
from .players import PlayerStrategy
from .masking import CNNMaskingStrategy, TransformerMaskingStrategy

from .utils import as_hwc_array, tensor_to_numpy, ImageLike


class ImageImputer(Imputer):
    """
    Imputer for images: creates masked versions of the input image based on player coalitions and returns model predictions.    
    """
    def __init__(
        self,
        model: Model,
        image: ImageLike,
        player_strategy: PlayerStrategy | None = None,
        masking_strategy: CNNMaskingStrategy | TransformerMaskingStrategy | None = None,
        normalize: bool = True,
        model_architecture: ModelArchitectureStrategy | None = None,
        batch_size: int = 32,
        vit_processor=None,
    ):
                     
        self.image = as_hwc_array(image)
        self.architecture = model_architecture or self._predict_model_architecture(model, masking_strategy, player_strategy, vit_processor)
        self.batch_size = batch_size

        self.architecture.prepare(self.image)
        self.n_features = self.architecture._player_strategy.n_players

        dummy_data = np.zeros((1, self.n_features))
        super().__init__(model=model, data=dummy_data)

        self.empty_prediction = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """
        Evaluate the model for a batch of player coalitions.
        
        Args:
            coalitions: (n_coalitions, n_players) boolean array
            
        Returns:
            (n_coalitions,) float array with model-predictions for each coalition.
        
        """
        import torch
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
            
        n = len(coalitions)
        if n <= self.batch_size:
            print("No batching required for value function evaluation.")
            coalitions_t = torch.from_numpy(coalitions).bool()
            return tensor_to_numpy(self.architecture.value_function(coalitions_t))
            
        chunks = [
            tensor_to_numpy(
                self.architecture.value_function(
                    torch.from_numpy(coalitions[start : start + self.batch_size]).bool()
                )
            )
            for start in range(0, n, self.batch_size)
        ]
        print(f"Batched to {len(chunks)} chunks for value function evaluation.")
        return np.concatenate(chunks, axis=0)

    def calc_empty_prediction(self) -> float:
        """Evaluate the model with all players absent to obtain the baseline prediction.

        Returns:
            The scalar model output when no players are present.
        """
        return float(self.value_function(np.zeros((1, self.n_features), dtype=bool))[0])

    @property
    def player_masks(self) -> np.ndarray:
        """Spatial masks per player as a ``(n_players, H, W)`` boolean numpy array.

        Returns:
            Boolean numpy array of shape ``(n_players, H, W)``.
        """
        return tensor_to_numpy(self.architecture.player_masks)
    
    def _predict_model_architecture(self, model: Model, masking_strategy=None, player_strategy=None, vit_processor=None) -> ModelArchitectureStrategy:
        """Auto-detects the model architecture and returns the appropriate ModelArchitectureStrategy."""
        
        import torchvision.models as models
        if isinstance(model, models.ResNet):
            return CNNArchitecture(model, masking_strategy, player_strategy)
        
        from transformers import ViTForImageClassification
        if isinstance(model, ViTForImageClassification):
            if vit_processor is None:
                raise ValueError("Please provide a processor for ViT models.")
            return TransformerArchitecture(model, vit_processor, masking_strategy, player_strategy)
        
        raise ValueError(f"Could not auto-detect architecture for model type '{type(model)}'.")