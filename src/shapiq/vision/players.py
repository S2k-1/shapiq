from __future__ import annotations

import math
import warnings
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
        """Return per-player boolean masks of shape ``(n_players, H, W)``."""
        ...


class LatentPlayerStrategy(PlayerStrategy, ABC):
    """Player strategy that returns a 1D boolean mask in latent/token space."""

    @abstractmethod
    def get_latent_mask(self, coalition: np.ndarray) -> torch.Tensor:
        """Return a ``(n_tokens,)`` bool tensor; True = token masked (absent)."""
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
        #: Number of token-grid cells along each side of a macro-region.
        self.patch_size = grid_size // side
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


class GridStrategy(PixelPlayerStrategy):
    """Splits the image into a regular rectangular grid without any external dependencies.

    Divides the image into ``rows × cols`` non-overlapping tiles.  Tiles are sized
    via integer division, so the rightmost column and bottom row absorb any remainder
    pixels when the image dimensions are not evenly divisible.

    This is the fastest option for quick experiments.  For content-aware segmentation
    use :class:`SuperpixelStrategy` instead.

    Args:
        rows: Number of tile rows.
        cols: Number of tile columns. Defaults to ``rows`` (square grid).

    Example::

        strategy = GridStrategy(rows=3, cols=3)  # 9 players
    """

    def __init__(self, rows: int, cols: int | None = None):
        self.rows = rows
        self.cols = cols if cols is not None else rows

    def get_masks(self, image: np.ndarray) -> np.ndarray:
        """Return per-tile boolean masks of shape ``(n_players, H, W)``."""
        H, W = image.shape[:2]
        masks = []
        for r in range(self.rows):
            for c in range(self.cols):
                y0 = r * H // self.rows
                y1 = (r + 1) * H // self.rows
                x0 = c * W // self.cols
                x1 = (c + 1) * W // self.cols
                m = np.zeros((H, W), dtype=bool)
                m[y0:y1, x0:x1] = True
                masks.append(m)
        return np.stack(masks, axis=0)

    @property
    def n_players(self) -> int:
        return self.rows * self.cols


class SuperpixelStrategy(PixelPlayerStrategy):
    """Splits the image into superpixels using SLIC."""

    def __init__(self, n_segments: int = 10):
        self.n_segments = n_segments

    def get_masks(self, image: np.ndarray) -> np.ndarray:
        """Run SLIC and return per-segment boolean masks of shape ``(n_players, H, W)``.

        If SLIC produces fewer segments than requested, the segment count is nudged
        upward and SLIC is retried up to 20 times.  If the target still cannot be
        reached, a warning is issued and ``n_segments`` is updated to the actual
        count so that ``n_players`` stays consistent with the returned masks.

        Args:
            image: ``(H, W, C)`` image array.

        Returns:
            Boolean mask array of shape ``(n_players, H, W)``.
        """
        from skimage.segmentation import slic

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
        else:
            warnings.warn(
                f"SLIC could only produce {n_superpixels} superpixels for the requested "
                f"{self.n_segments}. Using {n_superpixels} players instead.",
                stacklevel=2,
            )
            self.n_segments = n_superpixels

        players = np.arange(1, self.n_segments + 1).reshape(-1, 1, 1)
        masks = superpixels == players  # (n_players, H, W)
        return masks

    @property
    def n_players(self) -> int:
        return self.n_segments
