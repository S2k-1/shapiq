"""This module contains the upset plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from ._config import BLUE, RED

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from PIL.Image import Image

    from shapiq.interaction_values import InteractionValues


def upset_plot(
    interaction_values: InteractionValues,
    *,
    n_interactions: int = 20,
    feature_names: Sequence[str] | None = None,
    feature_image_patches: dict[int, Image] | list[Image] | None = None,
    feature_image_patches_size: float = 0.4,
    feature_image_patches_gap: float = 1.5,
    color_matrix: bool = False,
    all_features: bool = True,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
) -> Figure | None:
    """Plots the upset plot.

    UpSet plots [Lex14]_ can be used to visualize the interactions between features. The plot
    consists of two parts: the upper part shows the interaction values as bars, and the lower part
    shows the interactions as a matrix. Originally, the UpSet plot was introduced by Lex et al.
    (2014) [Lex14]_.
    For a more detailed explanation about the plots, see the references or the original
    [documentation](https://upset.app/).

    An example of this plot is shown below.

    .. image:: /_static/images/upset_plot.png
        :width: 600
        :align: center

    Args:
        interaction_values: The interaction values as an ``InteractionValues`` object.
        feature_names: The names of the features. Defaults to ``None``. If ``None``, the features
            will be named with their index. Ignored for the players that have an image patch in
            ``feature_image_patches``.
        feature_image_patches: Image patches to display instead of textual feature names along the
            y-axis (one image per player), analogous to the ``feature_image_patches`` argument of
            :func:`~shapiq.plot.si_graph.si_graph_plot`. Either a ``dict`` mapping player index to
            image or a ``list`` of images indexed by player. Players not contained in the
            ``dict``/``list`` fall back to their textual ``feature_names``. Defaults to ``None``.
        feature_image_patches_size: The size of the image patches relative to the height of a single
            matrix row. A value of ``1.0`` makes a patch as tall as one row. Defaults to ``0.4``.
        feature_image_patches_gap: Horizontal gap between the image patches and the first matrix
            column, expressed in units of the matrix column spacing (the distance between two dots).
            A value of ``1.0`` makes the gap equal to the dot-to-dot spacing; values ``> 1.0`` make
            it clearly larger, so the patches read as a separate row-index column. Defaults to
            ``1.5``.
        n_interactions: The number of top interactions to plot. Defaults to ``20``. Note this number
            is completely arbitrary and can be adjusted to the user's needs.
        color_matrix: Whether to color the matrix (red for positive values, blue for negative) or
            not (black). Defaults to ``False``.
        all_features: Whether to plot all ``n_players`` features or only the features that are
            present in the top interactions. Defaults to ``True``.
        figsize: The size of the figure. Defaults to ``None``. If ``None``, the size will be set
            automatically depending on the number of features.
        show: Whether to show the plot. Defaults to ``False``.

    Returns:
        If ``show`` is ``True``, the function returns ``None``. Otherwise, it returns a tuple with
        the figure and the axis of the plot.

    References:
        .. [Lex14] Alexander Lex, Nils Gehlenborg, Hendrik Strobelt, Romain Vuillemot, Hanspeter Pfister. UpSet: Visualization of Intersecting Sets IEEE Transactions on Visualization and Computer Graphics (InfoVis), 20(12): 1983--1992, doi:10.1109/TVCG.2014.2346248, 2014.

    """
    # prepare data ---------------------------------------------------------------------------------
    values = interaction_values.values
    values_ids: dict[int, tuple[int, ...]] = {
        v: k for k, v in interaction_values.interaction_lookup.items()
    }
    values_abs = abs(values)
    idx = values_abs.argsort()[::-1]
    idx = idx[:n_interactions] if n_interactions > 0 else idx
    values = values[idx]
    interactions: list[tuple[int, ...]] = [values_ids[i] for i in idx]

    # prepare feature names ------------------------------------------------------------------------
    if all_features:
        features = set(range(interaction_values.n_players))
    else:
        features = {feature for interaction in interactions for feature in interaction}
    n_features = len(features)
    feature_pos = {feature: n_features - 1 - i for i, feature in enumerate(features)}
    if feature_names is None:
        feature_name_map = {feature: f"Feature {feature}" for feature in features}
    else:
        feature_name_map = {feature: feature_names[feature] for feature in features}

    # create figure --------------------------------------------------------------------------------
    height_upper, height_lower = 5, n_features * 0.75
    height = height_upper + height_lower
    ratio = [height_upper, height_lower]
    if figsize is None:
        figsize = (10, height)
    else:
        if figsize[1] is None:
            figsize = (figsize[0], height)
        if figsize[0] is None:
            figsize = (10, figsize[1])

    fig, ax = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": ratio}, sharex=True)

    # plot lower part of the upset plot
    for x_pos, interaction in enumerate(interactions):
        color = RED.hex if values[x_pos] >= 0 else BLUE.hex

        # plot upper part
        bar = ax[0].bar(x_pos, values[x_pos], color=color)
        label = [f"{values[x_pos]:.2f}"]
        ax[0].bar_label(bar, label, label_type="edge", color="black", fontsize=12, padding=3)

        # plot lower part
        # plot the matrix in the background
        ax[1].plot(
            [x_pos for _ in range(n_features)],
            list(range(n_features)),
            color="lightgray",
            marker="o",
            markersize=15,
            linewidth=0,
        )
        # add the interaction to the matrix
        y_pos = [feature_pos[feature] for feature in interaction]
        ax[1].plot(
            [x_pos for _ in range(len(interaction))],
            y_pos,
            color="black" if not color_matrix else color,
            marker="o",
            markersize=15,
            linewidth=1.5,
        )

    # beautify upper plot --------------------------------------------------------------------------
    min_max = (min(values), max(values))
    delta = (min_max[1] - min_max[0]) * 0.1
    ax[0].set_ylim(min_max[0] - delta, min_max[1] + delta)
    ax[0].set_ylabel("Interaction Value")
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["bottom"].set_visible(False)
    ax[0].axhline(0, color="black", linewidth=0.5)  # add line at 0

    # beautify lower plot --------------------------------------------------------------------------
    ax[1].set_ylim(-1, n_features)
    ax[1].yaxis.set_ticks(range(n_features))
    # build the y-tick labels by position; players with an image patch get a blank label
    patches = {feature: _get_feature_patch(feature_image_patches, feature) for feature in features}
    pos_to_feature = {pos: feature for feature, pos in feature_pos.items()}
    ax[1].set_yticklabels(
        [
            ""
            if patches[pos_to_feature[pos]] is not None
            else feature_name_map[pos_to_feature[pos]]
            for pos in range(n_features)
        ],
    )
    ax[1].tick_params(axis="y", length=0)  # remove y-ticks
    ax[1].set_xticks([])  # remove x-axis
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    # background shading
    for i in range(n_features):
        if i % 2 == 0:
            ax[1].axhspan(i - 0.5, i + 0.5, color="lightgray", alpha=0.25, zorder=0, lw=0)

    # adjust whitespace
    plt.subplots_adjust(hspace=0.0)

    # draw image patches in place of the textual feature names -------------------------------------
    if any(patch is not None for patch in patches.values()):
        _draw_upset_feature_patches(
            ax[1],
            patches=patches,
            feature_pos=feature_pos,
            patch_size=feature_image_patches_size,
            row_height=height_lower / n_features,
            gap=feature_image_patches_gap,
        )

    if not show:
        return fig
    plt.show()
    return None


def _get_feature_patch(
    feature_image_patches: dict[int, Image] | list[Image] | None,
    feature: int,
) -> Image | None:
    """Returns the image patch for a player or ``None`` if none is available."""
    if feature_image_patches is None:
        return None
    if isinstance(feature_image_patches, dict):
        return feature_image_patches.get(feature)
    if 0 <= feature < len(feature_image_patches):
        return feature_image_patches[feature]
    return None


def _draw_upset_feature_patches(
    ax: Axes,
    *,
    patches: Mapping[int, Image | None],
    feature_pos: Mapping[int, int],
    patch_size: float,
    row_height: float,
    gap: float,
) -> None:
    """Draws the player image patches along the y-axis of the upset matrix.

    The patches replace the textual feature labels and are vertically centered on their respective
    row. They are anchored in *data* coordinates so the gap to the first matrix column is expressed
    in the same units as the dot-to-dot spacing: the right edge of each patch sits ``gap`` data
    units to the left of the first column (which lives at ``x = 0``). Because the matrix dots are
    spaced one data unit apart, a ``gap`` of ``1.5`` makes the patch-to-dot distance 1.5x the
    dot-to-dot distance, independent of the figure size or the number of interactions. Patches keep
    their native aspect ratio and are sized relative to a single matrix row, mirroring the behavior
    of the image patches in :func:`~shapiq.plot.si_graph.si_graph_plot`.

    Args:
        ax: The matrix axis to draw the patches on.
        patches: Mapping from player index to its image patch (or ``None``).
        feature_pos: Mapping from player index to its row position on the y-axis.
        patch_size: Patch height relative to a single matrix row (``1.0`` fills one row).
        row_height: Height of a single matrix row in inches.
        gap: Distance between the right edge of the patches and the first matrix column, in units
            of the matrix column spacing (the distance between two dots).

    """
    # target patch height in inches (a row spans ``row_height`` inches and one data unit)
    target_height_in = row_height * patch_size

    annotations: list[AnnotationBbox] = []
    for feature, image in patches.items():
        if image is None:
            continue
        array = np.asarray(image)
        img_h = array.shape[0]
        # OffsetImage displays ``img_px * zoom / 72`` inches, independent of the figure dpi
        zoom = target_height_in * 72.0 / img_h
        image_box = OffsetImage(array, zoom=zoom)
        annotation = AnnotationBbox(
            image_box,
            # right edge of the patch sits ``gap`` dot-spacings left of the first matrix column
            (-gap, feature_pos[feature]),
            xycoords="data",
            box_alignment=(1.0, 0.5),  # right-center of the patch sits at the anchor
            frameon=False,
            pad=0.0,
            annotation_clip=False,
        )
        ax.add_artist(annotation)
        annotations.append(annotation)

    _fit_left_margin(ax, annotations)


def _fit_left_margin(ax: Axes, annotations: list[AnnotationBbox], pad: float = 0.02) -> None:
    """Expands the figure's left margin so the data-anchored patches are not clipped.

    The patches extend to the left of the axes into the figure margin. Their width is fixed in
    display units, so the required margin is found by rendering once and measuring the leftmost
    patch, then nudging the left margin until every patch clears the figure edge by ``pad``.

    Args:
        ax: The matrix axis the patches belong to.
        annotations: The patch annotation boxes to keep inside the figure.
        pad: Minimum figure-relative gap to keep between the leftmost patch and the figure edge.

    """
    fig = ax.get_figure()
    if fig is None or not annotations:
        return
    fig.canvas.draw()
    # ``get_renderer`` is only available on raster canvases (e.g. Agg)
    get_renderer = getattr(fig.canvas, "get_renderer", None)
    if get_renderer is None:
        # backends without a renderer: fall back to a generous fixed margin
        fig.subplots_adjust(left=0.3)
        return
    renderer = get_renderer()

    # iterate because changing the left margin rescales the axes (and thus the patch positions)
    for _ in range(5):
        min_x = min(ann.get_window_extent(renderer).x0 for ann in annotations)
        min_frac = min_x / fig.bbox.width
        if min_frac >= pad:
            break
        new_left = min(fig.subplotpars.left + (pad - min_frac), 0.6)
        fig.subplots_adjust(left=new_left)
        fig.canvas.draw()
