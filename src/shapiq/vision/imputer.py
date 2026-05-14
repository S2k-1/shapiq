from shapiq.imputer.base import Imputer
from .masking import MaskingStrategy
from .players import PlayerStrategy

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
        normalize: bool = True
    ):
        dummy_data = np.zeros((1, player_strategy.n_players)) # mock data input to match Imputers expected input shape
        super().__init__(model=model, data=dummy_data)
        
        self.image = image
        self.player_strategy = player_strategy
        self.player_masks = player_strategy.get_masks(image)
        self.masking_strategy = masking_strategy
    
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
        masked_images = self.masking_strategy.apply(self.image, self.player_masks, coalitions) # shape (n_coalitions, H, W, C)
        predictions = self.model(masked_images)
        
        return np.array(predictions).squeeze()
    
    
    def calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction of the model provided only missing features.

        """
        empty_image = self.masking_strategy.apply(self.image, self.player_masks, np.array([[False] * self.player_strategy.n_players]))[0]
        prediction = self.model(empty_image[np.newaxis]) # batch: shape (1, H, W, C)

        return float(prediction[0])