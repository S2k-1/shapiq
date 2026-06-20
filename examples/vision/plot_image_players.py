"""
Defining Players for Image Explanations
=========================================

This example explores the different ways to define players for image
explanations with ``shapiq``.  Players partition the image into regions —
each region's contribution to the model prediction is then quantified by
Shapley values.

We demonstrate three player strategies for CNN models, all on the same
ResNet-18 and ImageNet sample image:

- :class:`~shapiq.vision.players.GridStrategy` — a fixed rectangular grid
- :class:`~shapiq.vision.players.SuperpixelStrategy` — automatic SLIC
  superpixels (as used in the quickstart notebook)
- :class:`~shapiq.vision.players.CustomPlayerStrategy` — user-supplied masks,
  here derived from a fine-grained SLIC segmentation to show how any external
  segmenter output can be plugged in directly

All three strategies plug into the same
:class:`~shapiq.vision.architecture.CNNArchitecture` /
:class:`~shapiq.vision.explainer.ImageExplainer` pipeline without any other
changes.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# %%
# Load Image and Model
# ---------------------
# We reuse the same ImageNet sample and ResNet-18 model as in the quickstart
# notebook.  The preprocessing pipeline (resize → center-crop → normalize) is
# applied once and shared across all three player strategies.

from pathlib import Path

import numpy as np
import torchvision.models as models
from PIL import Image
from torchvision import transforms

from shapiq.vision import ImageExplainer
from shapiq.vision.architecture import CNNArchitecture
from shapiq.vision.masking import MeanColorMasking

image_path = Path("imagenet_sample.png")
pil_image = Image.open(image_path).convert("RGB")

resize_and_crop = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])
tensor_and_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resized_image = resize_and_crop(pil_image)
tensor_image = tensor_and_norm(resized_image)
image_np = np.array(resized_image)

resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.eval()

resized_image


# %%
# Strategy 1: Grid Players
# =========================
#
# :class:`~shapiq.vision.players.GridStrategy` divides the image into a
# regular rectangular grid.  You can specify either:
#
# - ``grid_shape`` — fix the number of tiles; patch sizes are derived from
#   the image dimensions.
# - ``patchsize`` — fix the pixel size of each patch; the grid dimensions
#   are inferred.
#
# Edge patches absorb any remainder pixels when the image size is not an
# exact multiple of the grid, so every pixel is always owned by exactly one
# player.

from shapiq.vision.players import GridStrategy

grid_strategy = GridStrategy(grid_shape=5)  # 5×5 = 25 players
grid_masks = grid_strategy.get_masks(image_np)

print(f"Grid players: {grid_strategy.n_players}")

# %%
# We can also fix the patch size in pixels and let the grid shape be inferred.
# Here each patch is 56×56 px, which gives a 4×4 grid for a 224×224 image —
# equivalent to the ``gridshape=4`` call above.
#
# .. code-block:: python
#
#     grid_strategy = GridStrategy(patchsize=56)

# %%
# Explain with Grid Players
# ---------------------------
# We pass the ``GridStrategy`` instance to ``CNNArchitecture`` via the
# ``player_strategy`` argument.  Everything else stays at its default.

cnn_arch_grid = CNNArchitecture(
    model=resnet,
    player_strategy=grid_strategy,
)
explainer_grid = ImageExplainer(
    model_architecture=cnn_arch_grid,
    data=tensor_image,
    index="k-SII",
    max_order=2,
    batch_size=32,
)

iv_grid = explainer_grid.explain(budget=512)

iv_grid.plot_image_attributions(image=image_np, explainer=explainer_grid, heatmap_only=False)


# %%
# Strategy 2: Automatic SLIC Superpixels
# ========================================
#
# :class:`~shapiq.vision.players.SuperpixelStrategy` runs SLIC internally
# and is the default player strategy for ``CNNArchitecture``.  It is shown
# here for comparison — see the quickstart notebook for a full walkthrough.
# The number of superpixels (players) might increase or decrease from the 
# requested ``nsegments`` depending on the SLIC output.

from shapiq.vision.players import SuperpixelStrategy

slic_strategy = SuperpixelStrategy(n_segments=16)
slic_masks = slic_strategy.get_masks(image_np)

print(f"SLIC players (requested 16, got {slic_strategy.n_players})")


# %%
# Strategy 3: Custom Players from a Fine-Grained SLIC Segmentation
# =================================================================
#
# :class:`~shapiq.vision.players.CustomPlayerStrategy` accepts any
# user-supplied partition — binary masks, a segmentation label map, or the
# output of an external segmenter.
#
# Here we demonstrate the workflow with a SLIC segmentation received from the ``skimage`` implementation.

from skimage.segmentation import slic as skimage_slic

fine_labels = skimage_slic(image_np, n_segments=64)
print(f"Unique SLIC labels (fine-grained): {len(np.unique(fine_labels))}")

# %%
# Pass the 2-D integer label map directly to ``CustomPlayerStrategy``.
# It converts the label map to a 3-D boolean mask array automatically.

from shapiq.vision.players import CustomPlayerStrategy

custom_strategy = CustomPlayerStrategy(masks=fine_labels)
print(f"Custom players from SLIC: {custom_strategy.n_players}")

custom_masks = custom_strategy.get_masks(image_np)

# %%
# The same interface accepts a pre-built 3-D boolean array, which is useful
# when you already have masks from an external source:
#
# .. code-block:: python
#
#     my_masks = np.zeros((n_players, H, W), dtype=bool)
#     # ... fill in your masks ...
#     custom_strategy = CustomPlayerStrategy(masks=my_masks)
#
# Any pixels not covered by any mask stay visible in every coalition and
# are not attributed.  ``CustomPlayerStrategy`` will raise a ``UserWarning``
# if such pixels exist, so you can catch accidental gaps.

# %%
# Explain with Custom Players
# ----------------------------
# With 64 players a larger budget improves estimate quality.

cnn_arch_custom = CNNArchitecture(
    model=resnet,
    player_strategy=custom_strategy,
    masking_strategy=MeanColorMasking(),
)
explainer_custom = ImageExplainer(
    model_architecture=cnn_arch_custom,
    data=tensor_image,
    index="k-SII",
    max_order=2,
    batch_size=32,
)

iv_custom = explainer_custom.explain(budget=512)

iv_custom.plot_image_attributions(image=image_np, explainer=explainer_custom, heatmap_only=False)

# %%
# References
# ----------
# .. footbibliography::