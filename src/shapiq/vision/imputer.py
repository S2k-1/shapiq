import torch
from transformers import ViTImageProcessor

from shapiq.imputer.base import Imputer
from .masking import MaskingStrategy
from .players import PlayerStrategy
from .explainer import ModelType

import numpy as np

class ImageImputer(Imputer):
    """
    Imputer for images: creates masked versions of the input image based on player coalitions and returns model predictions.    
    
    """
    def __init__(
        self,
        model,           
        image: np.ndarray, 
        player_strategy: PlayerStrategy,
        masking_strategy: MaskingStrategy,
        model_type: ModelType,
        normalize: bool = True,
    ):
        dummy_data = np.zeros((1, player_strategy.n_players)) # mock data input to match Imputers expected input shape
        super().__init__(model=model, data=dummy_data)
        
        self.image = image
        self.player_strategy = player_strategy
        self.masking_strategy = masking_strategy
        self.model_type = model_type
        
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
        if self.model_type == "resnet":
            player_masks = self.player_strategy.get_masks(self.image)
            masked_images = self.masking_strategy.apply(self.image, player_masks, coalitions) # shape (n_coalitions, H, W, C)
            predictions = self.model(masked_images)
            
            return np.array(predictions).squeeze()
        
        elif self.model_type == "vit":
            # TODO: exclude preprocessing
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch32-384")
            inputs = processor(images=self.image, return_tensors="pt")
            pixel_values = inputs["pixel_values"]
            
            if coalitions.ndim == 1:
                coalitions = coalitions.reshape(1, -1)

            bool_masks = torch.stack(
                [self.player_strategy.get_masks(c) for c in coalitions]
            )

            batch = pixel_values.repeat(bool_masks.shape[0], 1, 1, 1)
            with torch.no_grad():
                logits = self.model(pixel_values=batch, bool_masked_pos=bool_masks).logits
                probs = torch.softmax(logits, dim=-1)

            return probs[:, self.class_id].cpu().numpy()
        
        else: 
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    
    def calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction of the model provided only missing features.

        """
        if self.model_type == "resnet":
            empty_image = self.masking_strategy.apply(self.image, self.player_masks, np.array([[False] * self.player_strategy.n_players]))[0]
            prediction = self.model(empty_image[np.newaxis]) # batch: shape (1, H, W, C)

            return float(prediction[0])
        
        elif self.model_type == "vit":
            return float(self.value_function(np.zeros((1, self.n_players), dtype=bool))[0])
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")