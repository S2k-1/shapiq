"""Player strategies for vision-based explanations."""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


class PlayerStrategy(ABC):
    """Defines how the image is split into n_players regions."""

    @property
    @abstractmethod
    def n_players(self) -> int:
        """Return the number of players (regions) in the strategy."""
        ...


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

    Groups the ``grid_size x grid_size`` token grid into ``n_players`` macro-regions
    arranged in a ``sqrt(n_players) x sqrt(n_players)`` layout. Region boundaries are
    computed via integer division so the macro-regions tile the full grid even when
    ``grid_size`` is not evenly divisible by ``sqrt(n_players)`` (e.g. ViT-B/16 with
    ``grid_size=14`` and ``n_players=9``).
    """

    def __init__(self, grid_size: int, n_players: int) -> None:
        """Initialize the PatchStrategy.

        Args:
            grid_size: Number of tokens along each side of the full token grid.
            n_players: Number of macro-regions. Must be a perfect square.

        Raises:
            ValueError: If ``n_players`` is not a perfect square.
        """
        side = int(math.sqrt(n_players))
        if side * side != n_players:
            msg = "n_players must be a perfect square."
            raise ValueError(msg)
        self.grid_size = grid_size
        #: Number of token-grid cells along each side of a macro-region.
        self.patch_size = grid_size // side
        self.side = side
        self._n_players = n_players

    def get_latent_mask(self, coalition: np.ndarray) -> torch.Tensor:
        """Return a ``(grid_size^2,)`` bool mask; True = token masked (absent)."""
        import torch  # lazy: only ViT/latent users pay this cost

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
        """Return the number of macro-region players."""
        return self._n_players


class CustomMasksStrategy(PixelPlayerStrategy):
    """Uses a set of pre-computed binary masks as players.

    Lets users define completely arbitrary regions rather than relying on SLIC or
    a fixed grid.  Each mask is a boolean ``(H, W)`` array where ``True`` marks
    the pixels belonging to that player.

    Masks may overlap; pixels not covered by any mask are treated as always absent
    regardless of coalition membership.

    Args:
        masks: Array of shape ``(n_players, H, W)``.  Any dtype is accepted and
            will be cast to ``bool``.

    Example::

        masks = np.zeros((3, 224, 224), dtype=bool)
        masks[0, :112, :] = True    # top half
        masks[1, 112:, :] = True    # bottom half
        masks[2, :, 100:124] = True  # centre column (overlaps both)
        strategy = CustomMasksStrategy(masks)
    """

    def __init__(self, masks: np.ndarray) -> None:
        """Initialize the CustomMasksStrategy.

        Args:
            masks: Array of shape ``(n_players, H, W)`` defining player regions.

        Raises:
            ValueError: If ``masks`` is not a 3-D array.
        """
        if np.asarray(masks).ndim != 3:
            msg = (
                f"masks must be a 3-D array of shape (n_players, H, W), "
                f"got shape {np.asarray(masks).shape}."
            )
            raise ValueError(msg)
        self._masks = np.asarray(masks, dtype=bool)

    def get_masks(self, _image: np.ndarray) -> np.ndarray:
        """Return the pre-computed masks (image argument is ignored)."""
        return self._masks

    @property
    def n_players(self) -> int:
        """Return the number of pre-computed player masks."""
        return self._masks.shape[0]


class GridStrategy(PixelPlayerStrategy):
    """Splits the image into a regular rectangular grid without any external dependencies.

    Divides the image into ``rows x cols`` non-overlapping tiles.  Tiles are sized
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

    def __init__(self, rows: int, cols: int | None = None) -> None:
        """Initialize the GridStrategy.

        Args:
            rows: Number of tile rows.
            cols: Number of tile columns. Defaults to ``rows``.
        """
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
        """Return the number of grid tiles."""
        return self.rows * self.cols


class SuperpixelStrategy(PixelPlayerStrategy):
    """Splits the image into superpixels using SLIC."""

    def __init__(self, n_segments: int = 10) -> None:
        """Initialize the SuperpixelStrategy.

        Args:
            n_segments: Target number of superpixel segments (players).
        """
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
        return superpixels == players

    @property
    def n_players(self) -> int:
        """Return the number of superpixel segments."""
        return self.n_segments
