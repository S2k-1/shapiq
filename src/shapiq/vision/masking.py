from abc import ABC, abstractmethod
import numpy as np

class MaskingStrategy(ABC):
    """Defines how the masked pixels are imputed/replaced."""
    
    @abstractmethod
    def apply(self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        """
        Args:
            image:        (H, W, C) original image
            player_masks: (n_players, H, W) boolean masks per player
            coalition:    (n_coalitions, n_players) boolean array
        
        Returns:
            masked_images: (n_coalitions, H, W, C)
        """
        ...


class MeanColorMasking(MaskingStrategy):
    """Imputes the masked pixels with the mean color of the entire image."""
    
    def apply(self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape
        
        masked_images = np.stack([image] * n_coalitions, axis=0) # shape (n_coalitions, H, W, C)
        
        mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    mask[i] |= player_masks[j]
                    
        masked_images[mask] = image.mean(axis=(0, 1))
        
        return masked_images


class ZeroMasking(MaskingStrategy):
    """Imputes the masked pixels with zeros (or a configurable value)."""
    
    def __init__(self, value: float = 0.0):
        self.value = value
    
    def apply(self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape
        
        masked_images = np.stack([image] * n_coalitions, axis=0) # shape (n_coalitions, H, W, C)
        
        mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    mask[i] |= player_masks[j]
                    
        masked_images[mask] = self.value
        
        return masked_images