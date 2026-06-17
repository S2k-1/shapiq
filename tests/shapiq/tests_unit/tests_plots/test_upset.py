"""This module contains all tests for the upset plot."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from shapiq.interaction_values import InteractionValues
from shapiq.plot import upset_plot


def test_upset_plot():
    """Test the force plot function."""
    lookup = {
        (0,): 0,
        (1,): 1,
        (2,): 2,
        (0, 1): 3,
        (0, 2): 4,
        (0, 1, 2): 5,
        (1, 4): 6,
        (2, 3): 7,
        (0, 1, 3): 8,
        (0, 2, 3): 9,
        (0, 1, 2, 3): 10,
        (0, 1, 2, 4): 11,
        (0, 1, 2, 3, 4): 12,
    }
    iv = InteractionValues(
        values=np.array([1, 2, 1.5, -0.9, 0.1, 0.3, -0.2, 0.1, 0.11, -0.1, 0.2, 0.8, 0.05]),
        interaction_lookup=lookup,
        index="k-SII",
        min_order=1,
        max_order=5,
        baseline_value=0.0,
        n_players=5,
    )
    n_players = iv.n_players
    feature_names = [f"feature-{i}" for i in range(n_players)]
    feature_names = np.array(feature_names)

    fig = upset_plot(iv, feature_names=feature_names, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")

    fig = upset_plot(iv, feature_names=feature_names, color_matrix=True, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")

    # in the following feature 3 is not shown
    fig = upset_plot(iv, n_interactions=5, all_features=False, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")

    # in the following feature 3 is shown
    fig = upset_plot(iv, n_interactions=5, all_features=True, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")

    # test once directly from the interaction values
    fig = iv.plot_upset(feature_names=feature_names, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_upset_plot_image_patches():
    """Test the upset plot with image patches instead of textual feature names."""
    lookup = {
        (0,): 0,
        (1,): 1,
        (2,): 2,
        (3,): 3,
        (0, 1): 4,
        (1, 2): 5,
        (0, 2, 3): 6,
    }
    iv = InteractionValues(
        values=np.array([1.0, -2.0, 0.5, -0.3, 0.8, -0.4, 0.3]),
        interaction_lookup=lookup,
        index="k-SII",
        min_order=1,
        max_order=3,
        baseline_value=0.0,
        n_players=4,
    )

    def _patch(channel: int) -> Image.Image:
        array = np.zeros((16, 16, 3), dtype=np.uint8)
        array[..., channel % 3] = 200
        return Image.fromarray(array)

    patches = [_patch(i) for i in range(iv.n_players)]

    # patches as a list (one per player)
    fig = upset_plot(iv, feature_image_patches=patches, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")

    # patches as a partial dict -> missing players fall back to textual names
    fig = upset_plot(
        iv,
        feature_image_patches={0: patches[0], 3: patches[3]},
        feature_names=[f"f-{i}" for i in range(iv.n_players)],
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close("all")

    # custom patch size and via the interaction values method
    fig = iv.plot_upset(
        feature_image_patches=patches,
        feature_image_patches_size=0.6,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close("all")
