"""Tests for player strategies in ``shapiq.vision.players``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from shapiq.vision.players import (
    LatentPlayerStrategy,
    PatchStrategy,
    PixelPlayerStrategy,
    PlayerStrategy,
    SuperpixelStrategy,
)


class TestPatchStrategy:
    def test_is_latent_player_strategy(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        assert isinstance(strategy, LatentPlayerStrategy)
        assert isinstance(strategy, PlayerStrategy)

    def test_n_players_property(self) -> None:
        strategy = PatchStrategy(grid_size=6, n_players=9)
        assert strategy.n_players == 9

    def test_init_computes_patch_size_and_side(self) -> None:
        strategy = PatchStrategy(grid_size=8, n_players=4)
        assert strategy.side == 2
        assert strategy.patch_size == 4

    def test_init_rejects_non_perfect_square(self) -> None:
        with pytest.raises(ValueError, match="perfect square"):
            PatchStrategy(grid_size=8, n_players=5)

    def test_non_divisible_grid_works_and_covers_all_tokens(self) -> None:
        """ViT-B/16 grid_size=14 with a 3×3 macro-grid (n_players=9) must work."""
        strategy = PatchStrategy(grid_size=14, n_players=9)
        assert strategy.n_players == 9
        # All-present coalition must cover every token (no gaps in the mask).
        coalition = np.ones(9, dtype=bool)
        mask = strategy.get_latent_mask(coalition).reshape(14, 14)
        assert not mask.any(), "all-present coalition should leave no masked tokens"

    def test_mask_shape_is_flattened_grid(self) -> None:
        strategy = PatchStrategy(grid_size=6, n_players=9)
        coalition = np.array([True, False, True, False, True, False, True, False, True])
        mask = strategy.get_latent_mask(coalition)
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert mask.shape == (36,)

    def test_all_present_coalition_visible_everywhere(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        coalition = np.array([True, True, True, True])
        mask = strategy.get_latent_mask(coalition)
        # True == masked, so all-present means all-False (nothing masked).
        assert not mask.any()

    def test_empty_coalition_masks_everything(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        coalition = np.array([False, False, False, False])
        mask = strategy.get_latent_mask(coalition)
        assert mask.all()

    def test_single_player_visible_unmasks_correct_block(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        # Only player 0 (top-left 2x2 block) present.
        coalition = np.array([True, False, False, False])
        mask = strategy.get_latent_mask(coalition).reshape(4, 4)
        # Top-left 2x2 block is visible (False); rest masked (True).
        assert not mask[:2, :2].any()
        assert mask[:2, 2:].all()
        assert mask[2:, :].all()


class TestSuperpixelStrategy:
    def test_is_pixel_player_strategy(self) -> None:
        strategy = SuperpixelStrategy(n_segments=5)
        assert isinstance(strategy, PixelPlayerStrategy)
        assert isinstance(strategy, PlayerStrategy)

    def test_n_players_matches_n_segments(self) -> None:
        strategy = SuperpixelStrategy(n_segments=7)
        assert strategy.n_players == 7

    def test_get_masks_shape_and_dtype(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(32, 32, 3)).astype(np.float64)
        strategy = SuperpixelStrategy(n_segments=4)
        masks = strategy.get_masks(image)
        assert masks.shape[0] == 4
        assert masks.shape[1:] == (32, 32)
        assert masks.dtype == bool

    def test_get_masks_partition_property(self) -> None:
        """Each pixel should be covered by exactly one superpixel (after clipping)."""
        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, size=(24, 24, 3)).astype(np.float64)
        strategy = SuperpixelStrategy(n_segments=6)
        masks = strategy.get_masks(image)
        coverage = masks.sum(axis=0)
        # Every pixel is covered by exactly one segment.
        assert (coverage == 1).all()
