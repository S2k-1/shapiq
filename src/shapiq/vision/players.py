import torch
import math
import numpy as np
from abc import ABC, abstractmethod

class PlayerStrategy(ABC):
    """Defines how the image is split into n_players regions/players."""
    
    @abstractmethod
    def get_masks(self, image: np.ndarray) -> np.ndarray:
        """
        Returns binary masks for each player/region.
        
        Returns:
            masks: shape (n_players, H, W) — masks[i] == 1 where player i can see the image, 0 otherwise.
        """
        ...
    
    @property
    @abstractmethod
    def n_players(self) -> int:
        """Returns the number of players/regions."""
        ...


class PatchStrategy(PlayerStrategy):
    """Splits the image into patches."""
    
    def __init__(self, grid_size: int, n_players: int):
        side = int(math.sqrt(n_players))
        if side * side != n_players:
            raise ValueError("n_players must be a perfect square.")
        if grid_size % side != 0:
            raise ValueError("grid_size must be divisible by sqrt(n_players).")

        self.grid_size = grid_size
        self.patch_size = grid_size // side
        self.side = side
        self._n_players = n_players

    
    def get_masks(self, coalition: np.ndarray) -> torch.Tensor:
        """Transforms a coalition of players (i.e. super-patches) into a boolean mask.

        The Vision Transformer model uses a 1D boolean mask to mask out patches that are not part of
        for the prediction. The underlying model operates on 12x12 (i.e. 144) patches. To reduce the
        number of players (i.e. patches), we group them together to form super-patches. The
        super-patches are then masked out jointly.

        Args:
            coalition: The coalition of players.
            n_patches: The number of patches used in the model.

        Returns:
            The boolean mask for the model in 1D with shape (144,).

        """
        # True = masked, False = visible
        mask_2d = torch.ones((self.grid_size, self.grid_size), dtype=torch.bool)
        for player, is_present in enumerate(coalition):
            if is_present:
                y = (player // self.side) * self.patch_size
                x = (player % self.side) * self.patch_size
                mask_2d[y : y + self.patch_size, x : x + self.patch_size] = False
        return mask_2d.flatten()
    
    @property
    def n_players(self) -> int:
        return self._n_players


class SuperpixelStrategy(PlayerStrategy):
    """Splits the image into superpixels using SLIC."""
    
    def __init__(self, n_segments: int = 10):
        self.n_segments = n_segments
    
    def get_masks(self, image: np.ndarray) -> np.ndarray: # used code form shapiq_games._setup._resnet_setup
        """Run SLIC and return the superpixel mask.

        Runs SLIC and retrying with randomized values if the number of superpixels does not match
        the desired number.

        Args:
            image: The image

        Returns:
            The superpixel mask

        """
        from skimage.segmentation import slic
        
        # run slic for first time
        superpixels = slic(image, n_segments=self.n_segments, start_label=1, slic_zero=True)
        n_superpixels = len(np.unique(superpixels))

        # retry with increasing segments
        if n_superpixels < self.n_segments:
            iteration, n_segments_iter = 0, self.n_segments
            while iteration < 20 and n_superpixels < self.n_segments:
                n_segments_iter += 1
                superpixels = slic(image, n_segments=n_segments_iter, start_label=1, slic_zero=True)
                n_superpixels = len(np.unique(superpixels))
                iteration += 1

        # fallback to clipping the last superpixels
        if n_superpixels >= self.n_segments:
            # clip the superpixels to the desired number of segments
            superpixels = np.clip(superpixels, a_min=1, a_max=self.n_segments)
            n_superpixels = self.n_segments
            
        players = np.arange(1, self.n_segments + 1).reshape(-1, 1, 1) # shape (n_players, 1, 1), reshape for broadcasting
        masks = (superpixels == players) # shape (n_players, H, W)
        
        return masks
    
    @property
    def n_players(self) -> int:
        return self.n_segments