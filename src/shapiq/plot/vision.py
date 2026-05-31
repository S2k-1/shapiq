"""Image attribution heatmap plot for vision explainers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from shapiq.interaction_values import InteractionValues

__all__ = ["image_attribution_plot"]


def image_attribution_plot(
    interaction_values: InteractionValues,
    image: np.ndarray,
    player_masks: np.ndarray,
    *,
    region_label: str = "Region",
    alpha: float = 0.5,
    cmap: Colormap | str | None = None,
    show: bool = True,
) -> tuple[Figure, tuple[Axes, Axes]] | None:
    """Visualize first-order attributions as a heatmap overlaid on the original image.

    Args:
        interaction_values: The interaction values to visualize.
        image: The original image as a ``(H, W, C)`` numpy array.
        player_masks: Boolean array of shape ``(n_players, H, W)`` mapping each player
            to its pixel region. Use ``explainer.imputer.player_masks``.
        region_label: x-axis label for the bar chart. Defaults to ``"Region"``.
        alpha: Transparency of the heatmap overlay. Defaults to ``0.5``.
        cmap: Matplotlib colormap or name. ``None`` uses shapiq's BLUE→white→RED
            diverging palette. Defaults to ``None``.
        show: Whether to display the plot. Defaults to ``True``.

    Returns:
        If ``show`` is ``False``, returns ``(figure, (ax_heatmap, ax_bar))``.

    """
    import matplotlib.cm as cm

    if cmap is None:
        from shapiq.plot._config import BLUE, NEUTRAL, RED

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "shapiq_diverging", [BLUE.hex, NEUTRAL.hex, RED.hex]
        )
    elif isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    first_order = interaction_values.get_n_order_values(1)
    n_players = len(first_order)

    heatmap = np.zeros(image.shape[:2], dtype=float)
    for i, mask in enumerate(player_masks):
        heatmap[mask] = first_order[i]

    vmin = min(float(first_order.min()), -1e-9)
    vmax = max(float(first_order.max()), 1e-9)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(image)
    axes[0].imshow(heatmap, alpha=alpha, cmap=cmap, norm=norm)
    try:
        from skimage.segmentation import mark_boundaries

        axes[0].imshow(mark_boundaries(image, player_masks.argmax(axis=0) + 1), alpha=0.3)
    except ImportError:
        pass
    axes[0].set_title("First-order attributions")
    axes[0].axis("off")

    bar_colors = [cmap(norm(v)) for v in first_order]
    axes[1].bar(range(n_players), first_order, color=bar_colors)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel(region_label)
    axes[1].set_ylabel("Attribution")
    axes[1].set_title(f"First-order values per {region_label.lower()}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=list(axes), fraction=0.02, pad=0.04)
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])

    if show:
        plt.show()
        return None
    return fig, (axes[0], axes[1])
