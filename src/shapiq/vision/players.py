"""Player strategies for vision-based explanations."""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import torch


def _require_positive_int(name: str, value: int) -> None:
    """Raise ``ValueError`` if ``value`` is not a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer, got {value!r}."
        raise ValueError(msg)


def _validate_coalition(coalition: np.ndarray, n_players: int) -> np.ndarray:
    """Return ``coalition`` as a 1D bool array after checking its length."""
    coalition = np.asarray(coalition, dtype=bool).ravel()
    if coalition.shape[0] != n_players:
        msg = f"coalition length {coalition.shape[0]} does not match n_players ({n_players})."
        raise ValueError(msg)
    return coalition


class PlayerStrategy(ABC):
    """Defines how the image is split into n_players regions."""

    @property
    @abstractmethod
    def n_players(self) -> int:
        """Return the number of players (regions) in the strategy."""
        ...


class PixelPlayerStrategy(PlayerStrategy, ABC):
    """Player strategy that returns spatial masks in pixel space.

    Per-player masks use ``True`` to mark pixels *belonging* to that player.
    Coalition-to-absence conversion (``True`` in a coalition = player present) is
    handled downstream in :func:`~shapiq.vision.masking.coalition_absence_mask`.
    """

    @abstractmethod
    def get_masks(self, image: np.ndarray) -> np.ndarray:
        """Return per-player boolean masks of shape ``(n_players, H, W)``.

        ``True`` marks pixels that belong to the player.
        """
        ...


class LatentPlayerStrategy(PlayerStrategy, ABC):
    """Player strategy that returns a 1D boolean mask in latent/token space.

    Latent masks use the opposite convention from pixel masks: ``True`` means the
    token is *masked away* (player absent). ``False`` means the token is visible.
    """

    @abstractmethod
    def get_latent_mask(self, coalition: np.ndarray) -> torch.Tensor:
        """Return a ``(n_tokens,)`` bool tensor; ``True`` = token masked (absent)."""
        ...

    def get_latent_mask_array(self, coalition: np.ndarray) -> np.ndarray:
        """Return a ``(n_tokens,)`` numpy bool mask; ``True`` = token masked (absent)."""
        coalition = _validate_coalition(coalition, self.n_players)
        return np.asarray(self.get_latent_mask(coalition), dtype=bool)


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
            ValueError: If ``grid_size`` or ``n_players`` is invalid.
        """
        _require_positive_int("grid_size", grid_size)
        _require_positive_int("n_players", n_players)
        side = int(math.sqrt(n_players))
        if side * side != n_players:
            msg = "n_players must be a perfect square."
            raise ValueError(msg)
        self.grid_size = grid_size
        self.side = side
        self._n_players = n_players

    def get_latent_mask_array(self, coalition: np.ndarray) -> np.ndarray:
        """Return a ``(grid_size^2,)`` bool mask; ``True`` = token masked (absent)."""
        coalition = _validate_coalition(coalition, self.n_players)
        mask_2d = np.ones((self.grid_size, self.grid_size), dtype=bool)
        for player, is_present in enumerate(coalition):
            if is_present:
                row = player // self.side
                col = player % self.side
                y_start = row * self.grid_size // self.side
                y_end = (row + 1) * self.grid_size // self.side
                x_start = col * self.grid_size // self.side
                x_end = (col + 1) * self.grid_size // self.side
                mask_2d[y_start:y_end, x_start:x_end] = False
        return mask_2d.reshape(-1)

    def get_latent_mask(self, coalition: np.ndarray) -> torch.Tensor:
        """Return a ``(grid_size^2,)`` bool mask; ``True`` = token masked (absent)."""
        import torch  # lazy: only ViT/latent users pay this cost

        return torch.from_numpy(self.get_latent_mask_array(coalition))

    @property
    def n_players(self) -> int:
        """Return the number of macro-region players."""
        return self._n_players


class CustomMasksStrategy(PixelPlayerStrategy):
    """Uses a set of pre-computed binary masks as players.

    Lets users define completely arbitrary regions rather than relying on SLIC or
    a fixed grid.  Each mask is a boolean ``(H, W)`` array where ``True`` marks
    the pixels belonging to that player.

    Masks may overlap.  Pixels not covered by any player mask are outside the
    game: they stay visible in every coalition because no player owns them, so
    they cannot be attributed or masked away.

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
        _require_positive_int("n_players", self._masks.shape[0])

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

        Raises:
            ValueError: If ``rows`` or ``cols`` is not a positive integer.
        """
        _require_positive_int("rows", rows)
        self.rows = rows
        self.cols = cols if cols is not None else rows
        _require_positive_int("cols", self.cols)

    def get_masks(self, image: np.ndarray) -> np.ndarray:
        """Return per-tile boolean masks of shape ``(n_players, H, W)``."""
        height, width = image.shape[:2]
        row_edges = [row * height // self.rows for row in range(self.rows + 1)]
        col_edges = [col * width // self.cols for col in range(self.cols + 1)]

        row_assign = np.empty(height, dtype=int)
        for row in range(self.rows):
            row_assign[row_edges[row] : row_edges[row + 1]] = row

        col_assign = np.empty(width, dtype=int)
        for col in range(self.cols):
            col_assign[col_edges[col] : col_edges[col + 1]] = col

        player_grid = row_assign[:, None] * self.cols + col_assign[None, :]
        return player_grid == np.arange(self.n_players)[:, None, None]

    @property
    def n_players(self) -> int:
        """Return the number of grid tiles."""
        return self.rows * self.cols


class SuperpixelStrategy(PixelPlayerStrategy):
    """Splits the image into superpixels using SLIC.

    Args:
        n_segments: Target number of superpixel segments. Defaults to ``10`` when
            ``mask`` is not provided. Pass ``None`` only together with ``mask``.
        algorithm: ``"slic"`` (default) or ``"slico"`` (regular-sized superpixels).
        mask: Optional precomputed segmentation — either a 2D integer label array
            ``(H, W)`` or a 3D boolean array ``(n_players, H, W)`` with non-overlapping
            regions. For overlapping regions use :class:`CustomMasksStrategy` instead.
    """

    def __init__(
        self,
        n_segments: int | None = 10,
        *,
        algorithm: Literal["slic", "slico"] = "slic",
        mask: np.ndarray | None = None,
    ) -> None:
        """Initialize the SuperpixelStrategy.

        Raises:
            ValueError: If neither ``n_segments`` nor ``mask`` is provided, or if
                ``n_segments`` is not a positive integer.
        """
        if mask is None and n_segments is None:
            msg = "n_segments must be provided when mask is not given."
            raise ValueError(msg)
        if n_segments is not None:
            _require_positive_int("n_segments", n_segments)
        self.n_segments = n_segments
        self._algorithm = algorithm
        self._custom_mask: np.ndarray | None = None
        self._n_players: int = n_segments if mask is None else 0
        if mask is not None:
            self.set_mask(mask)

    @staticmethod
    def _labels_to_masks(labels: np.ndarray) -> np.ndarray:
        """Convert a 2D integer label array to ``(n_players, H, W)`` boolean masks."""
        unique_labels = np.unique(labels)
        return labels == unique_labels.reshape(-1, 1, 1)

    def set_mask(self, mask: np.ndarray) -> None:
        """Validate and store a user-provided segmentation mask."""
        mask = np.asarray(mask)

        if mask.ndim == 2:
            if not np.issubdtype(mask.dtype, np.integer):
                msg = "2D mask must contain integer labels."
                raise ValueError(msg)
            if mask.size == 0:
                msg = "Provided 2D mask is empty."
                raise ValueError(msg)
            mask = self._labels_to_masks(mask)

        if mask.ndim == 3:
            mask = mask.astype(bool)
            if (mask.sum(axis=0) > 1).any():
                msg = (
                    "Masks are overlapping — each pixel must belong to exactly one player. "
                    "Use CustomMasksStrategy for overlapping regions."
                )
                raise ValueError(msg)
            if not mask.any(axis=0).all():
                msg = "Not all pixels are covered by at least one player."
                raise ValueError(msg)
        else:
            msg = (
                "mask must be either a 2D label array (H, W) or a "
                "3D boolean array (n_players, H, W)."
            )
            raise ValueError(msg)

        self._custom_mask = mask
        self.n_segments = mask.shape[0]
        self._n_players = mask.shape[0]

    def get_masks(self, image: np.ndarray) -> np.ndarray:
        """Run SLIC and return per-segment boolean masks of shape ``(n_players, H, W)``.

        SLIC may return more or fewer segments than ``n_segments``. The actual segment
        count is stored in ``_n_players`` and exposed via :attr:`n_players` — labels
        are never clipped or merged.
        """
        if self._custom_mask is not None:
            if self._custom_mask.shape[1:] != image.shape[:2]:
                msg = (
                    f"Custom mask shape {self._custom_mask.shape[1:]} does not match "
                    f"image shape {image.shape[:2]}."
                )
                raise ValueError(msg)
            return self._custom_mask

        from skimage.segmentation import slic

        slic_zero = self._algorithm == "slico"
        superpixels = slic(image, n_segments=self.n_segments, start_label=1, slic_zero=slic_zero)
        n_superpixels = len(np.unique(superpixels))

        if n_superpixels < self.n_segments:
            iteration, n_segments_iter = 0, self.n_segments
            while iteration < 20 and n_superpixels < self.n_segments:
                n_segments_iter += 1
                superpixels = slic(
                    image,
                    n_segments=n_segments_iter,
                    start_label=1,
                    slic_zero=slic_zero,
                )
                n_superpixels = len(np.unique(superpixels))
                iteration += 1

        if n_superpixels < self.n_segments:
            warnings.warn(
                f"SLIC could only produce {n_superpixels} superpixels for the requested "
                f"{self.n_segments}. Using {n_superpixels} players instead.",
                stacklevel=2,
            )
        elif n_superpixels > self.n_segments:
            warnings.warn(
                f"SLIC produced {n_superpixels} superpixels (requested {self.n_segments}). "
                f"Using all {n_superpixels} segments as players.",
                stacklevel=2,
            )

        self._n_players = n_superpixels
        return self._labels_to_masks(superpixels)

    @property
    def n_players(self) -> int:
        """Return the number of superpixel segments (actual count after ``get_masks``)."""
        return self._n_players
