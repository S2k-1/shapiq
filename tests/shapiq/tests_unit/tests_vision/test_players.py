"""Tests for player strategies in ``shapiq.vision.players``."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.vision.players import (
    CNNPlayerStrategy,
    PatchStrategy,
    PlayerStrategy,
    SuperpixelStrategy,
    TransformerPlayerStrategy,
)


class TestPatchStrategy:
    def test_is_transformer_player_strategy(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        assert isinstance(strategy, TransformerPlayerStrategy)
        assert isinstance(strategy, PlayerStrategy)

    def test_n_players_property(self) -> None:
        strategy = PatchStrategy(grid_size=6, n_players=9)
        assert strategy.n_players == 9

    def test_init_computes_side_and_patch_size(self) -> None:
        strategy = PatchStrategy(grid_size=8, n_players=4)
        assert strategy.side == 2
        assert strategy.patch_size == 4

    def test_init_rejects_non_perfect_square(self) -> None:
        with pytest.raises(ValueError, match="perfect square"):
            PatchStrategy(grid_size=8, n_players=5)

    def test_init_rejects_non_divisible_grid_size(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            PatchStrategy(grid_size=14, n_players=9)

    def test_get_token_masks_shape(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        token_masks = strategy.get_token_masks()
        # 4 players, each owns a 2x2 patch -> 4 tokens per player.
        assert token_masks.shape == (4, 4)
        assert np.issubdtype(token_masks.dtype, np.integer)

    def test_get_token_masks_partition_covers_all_tokens(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        token_masks = strategy.get_token_masks()
        # The union of all token indices must cover the full flattened grid exactly once.
        all_tokens = np.sort(token_masks.reshape(-1))
        np.testing.assert_array_equal(all_tokens, np.arange(16))

    def test_get_token_masks_top_left_player_indices(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        token_masks = strategy.get_token_masks()
        # Player 0 owns the top-left 2x2 patch: flat indices 0, 1, 4, 5.
        np.testing.assert_array_equal(np.sort(token_masks[0]), np.array([0, 1, 4, 5]))

    def test_get_pixel_masks_shape_and_partition(self) -> None:
        strategy = PatchStrategy(grid_size=4, n_players=4)
        image = np.zeros((8, 8, 3))
        masks = strategy.get_pixel_masks(image)
        assert masks.shape == (4, 8, 8)
        assert masks.dtype == bool
        # Each pixel belongs to exactly one player.
        assert (masks.sum(axis=0) == 1).all()


class TestSuperpixelStrategy:
    def test_is_cnn_player_strategy(self) -> None:
        strategy = SuperpixelStrategy(n_segments=5)
        assert isinstance(strategy, CNNPlayerStrategy)
        assert isinstance(strategy, PlayerStrategy)

    def test_n_players_matches_n_segments(self) -> None:
        strategy = SuperpixelStrategy(n_segments=7)
        assert strategy.n_players == 7

    def test_get_masks_shape_and_dtype(self) -> None:
        pytest.importorskip("skimage")
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(32, 32, 3)).astype(np.float64)
        strategy = SuperpixelStrategy(n_segments=4)
        masks = strategy.get_masks(image)
        assert masks.shape[0] == strategy.n_players
        assert masks.shape[1:] == (32, 32)
        assert masks.dtype == bool

    def test_get_masks_partition_property(self) -> None:
        """Each pixel should belong to exactly one superpixel."""
        pytest.importorskip("skimage")
        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, size=(24, 24, 3)).astype(np.float64)
        strategy = SuperpixelStrategy(n_segments=6)
        masks = strategy.get_masks(image)
        coverage = masks.sum(axis=0)
        assert (coverage == 1).all()

    def test_slic_updates_n_players_after_segmentation(self) -> None:
        """n_players reflects the actual SLIC output, not just the request."""
        pytest.importorskip("skimage")
        image = np.random.default_rng(2).integers(0, 255, size=(16, 16, 3)).astype(np.float64)
        strategy = SuperpixelStrategy(n_segments=20)
        masks = strategy.get_masks(image)
        assert strategy.n_players == masks.shape[0]
        assert strategy.n_players >= 1
        assert (masks.sum(axis=0) == 1).all()

    def test_custom_label_mask(self) -> None:
        labels = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        strategy = SuperpixelStrategy(mask=labels)
        image = np.zeros((4, 4, 3))
        masks = strategy.get_masks(image)
        assert masks.shape == (4, 4, 4)
        assert (masks.sum(axis=0) == 1).all()
        assert strategy.n_players == 4

    def test_custom_bool_mask_sets_n_players(self) -> None:
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[0, :, :2] = True
        masks[1, :, 2:] = True
        strategy = SuperpixelStrategy(mask=masks)
        assert strategy.n_players == 2

    def test_rejects_overlapping_custom_mask(self) -> None:
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[0, :, :2] = True
        masks[1, :, 1:3] = True
        with pytest.raises(ValueError, match="overlapping"):
            SuperpixelStrategy(mask=masks)

    def test_rejects_incomplete_custom_mask(self) -> None:
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[0, :, :1] = True
        masks[1, :, 1:2] = True  # columns 2, 3 uncovered
        with pytest.raises(ValueError, match="covered"):
            SuperpixelStrategy(mask=masks)

    def test_rejects_missing_n_segments_without_mask(self) -> None:
        with pytest.raises(ValueError, match="Either n_segments or mask"):
            SuperpixelStrategy(n_segments=None)

    def test_custom_mask_shape_mismatch_raises(self) -> None:
        labels = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
        strategy = SuperpixelStrategy(mask=labels)
        with pytest.raises(ValueError, match="does not match"):
            strategy.get_masks(np.zeros((8, 8, 3)))
