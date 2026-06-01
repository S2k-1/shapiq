from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import torch


class PlayerStrategy(ABC):
    """Defines how the image is split into n_players regions."""

    @property
    @abstractmethod
    def n_players(self) -> int: ...


class PixelPlayerStrategy(PlayerStrategy, ABC):
    """Player strategy that returns spatial masks in pixel space."""

    @abstractmethod
    def get_masks(self, image: np.ndarray) -> np.ndarray:
        # returns (n_players, H, W)
        ...


class LatentPlayerStrategy(PlayerStrategy, ABC):
    """Player strategy that returns a 1D boolean mask in latent/token space."""

    @abstractmethod
    def get_latent_mask(self, coalition: np.ndarray) -> torch.Tensor:
        # returns (n_tokens,) bool
        ...


class PatchStrategy(LatentPlayerStrategy):
    """Splits the image into patches for ViT models.

    Groups the ``grid_size × grid_size`` token grid into ``n_players`` macro-regions
    arranged in a ``sqrt(n_players) × sqrt(n_players)`` layout. Region boundaries are
    computed via integer division so the macro-regions tile the full grid even when
    ``grid_size`` is not evenly divisible by ``sqrt(n_players)`` (e.g. ViT-B/16 with
    ``grid_size=14`` and ``n_players=9``).
    """

    def __init__(self, grid_size: int, n_players: int):
        side = int(math.sqrt(n_players))
        if side * side != n_players:
            raise ValueError("n_players must be a perfect square.")
        self.grid_size = grid_size
        self.patch_size = grid_size // side  # kept for backward compatibility
        self.side = side
        self._n_players = n_players

    def get_latent_mask(self, coalition: np.ndarray) -> torch.Tensor:
        # True = masked (absent), False = visible (present); shape (grid_size^2,)
        mask_2d = torch.ones((self.grid_size, self.grid_size), dtype=torch.bool)
        for player, is_present in enumerate(coalition):
            if is_present:
                row = player // self.side
                col = player % self.side
                y_start = row * self.grid_size // self.side
                y_end = (row + 1) * self.grid_size // self.side
                x_start = col * self.grid_size // self.side
                x_end = (col + 1) * self.grid_size // self.side
                mask_2d[y_start:y_end, x_start:x_end] = False
        return mask_2d.flatten()

    @property
    def n_players(self) -> int:
        return self._n_players


class SuperpixelStrategy(PixelPlayerStrategy):
    """Splits the image into superpixels using SLIC."""

    def __init__(self, n_segments: int = 10):
        self.n_segments = n_segments

    def get_masks(self, image: np.ndarray) -> np.ndarray:
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

        if n_superpixels < self.n_segments:
            iteration, n_segments_iter = 0, self.n_segments
            while iteration < 20 and n_superpixels < self.n_segments:
                n_segments_iter += 1
                superpixels = slic(image, n_segments=n_segments_iter, start_label=1, slic_zero=True)
                n_superpixels = len(np.unique(superpixels))
                iteration += 1

        if n_superpixels >= self.n_segments:
            superpixels = np.clip(superpixels, a_min=1, a_max=self.n_segments)
            n_superpixels = self.n_segments

        players = np.arange(1, self.n_segments + 1).reshape(
            -1, 1, 1
        )  # shape (n_players, 1, 1), reshape for broadcasting
        masks = superpixels == players  # shape (n_players, H, W)

        return masks

    @property
    def n_players(self) -> int:
        return self.n_segments
