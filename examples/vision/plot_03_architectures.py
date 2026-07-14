"""
Architectures and Automatic Dispatch
====================================

Every :class:`~shapiq.vision.ImageExplainer` is backed by an *architecture
strategy* that bundles three things: the model, a player strategy (how the
image is split into regions), and a masking strategy (how removed regions
are replaced). Two architectures exist:

- :class:`~shapiq.vision.CNNArchitecture` operates in **pixel space**:
  masking edits the image itself. It accepts any classification model --
  CNNs, torchvision ViTs, or Hugging Face models (pass the ``processor``).
- :class:`~shapiq.vision.TransformerArchitecture` operates in **token
  space**: masking removes patch tokens before the forward pass. It
  requires a model that verifiably honors ``bool_masked_pos``.

The ``model`` argument of ``ImageExplainer`` takes either a raw model --
the architecture is then chosen by automatic dispatch -- or an architecture
you constructed yourself for full control. This example shows both paths.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from huggingface_hub import logging as hf_logging
from PIL import Image
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

hf_logging.set_verbosity_error()  # hide hub warnings (e.g. unauthenticated requests)
transformers.logging.set_verbosity_error()  # hide download noise and load reports
transformers.utils.logging.disable_progress_bar()

resize_and_crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
pil_image = Image.open(Path("imagenet_sample.png")).convert("RGB")
image = np.array(resize_and_crop(pil_image))

# %%
# Passing a Raw Model: Automatic Dispatch
# ---------------------------------------
# With a raw model, ``ImageExplainer`` calls
# :func:`~shapiq.vision.resolve_architecture` internally. The decision is
# *functional*, not name-based: the model is run once with every token
# masked, and only if the output changes does it get the token path. A ViT
# classifier passes that probe:

from shapiq.vision import resolve_architecture

vit_id = "google/vit-base-patch16-224"
vit_processor = AutoImageProcessor.from_pretrained(vit_id)
vit = AutoModelForImageClassification.from_pretrained(vit_id).eval()

architecture = resolve_architecture(vit, vit_processor)
print(type(architecture).__name__)

# %%
# BEiT accepts ``bool_masked_pos`` but its classification head silently
# ignores it -- token masking would produce constant, meaningless
# attributions. The probe catches this and dispatch falls back to pixel
# masking:

beit_id = "microsoft/beit-base-patch16-224-pt22k-ft22k"
beit_processor = AutoImageProcessor.from_pretrained(beit_id)
beit = AutoModelForImageClassification.from_pretrained(beit_id).eval()

architecture = resolve_architecture(beit, beit_processor)
print(type(architecture).__name__)

# %%
# Models without 2-D classification logits (encoder-only models such as
# ViT-MAE, segmentation or detection models) are rejected at dispatch with
# a ``TypeError``:

mae_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
mae = AutoModel.from_pretrained("facebook/vit-mae-base").eval()

try:
    resolve_architecture(mae, mae_processor)
except TypeError as err:
    print(f"TypeError: {err}")

# %%
# Constructing an Architecture Yourself
# -------------------------------------
# Construct the architecture explicitly to override any of its parts, then
# pass it as the ``model`` of ``ImageExplainer``. All arguments are
# optional except the model; defaults are
# :class:`~shapiq.vision.SuperpixelStrategy` and
# :class:`~shapiq.vision.MeanColorMasking` for ``CNNArchitecture``, and
# :class:`~shapiq.vision.PatchStrategy` (sized to the token grid) and
# :class:`~shapiq.vision.MaskTokenStrategy` for ``TransformerArchitecture``.

from shapiq.vision import CNNArchitecture, ImageExplainer, ZeroMasking
from shapiq.vision.players import GridStrategy

resnet = torch.nn.Sequential(
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
)
resnet = resnet.eval()

architecture = CNNArchitecture(
    model=resnet,
    player_strategy=GridStrategy(grid_shape=4),
    masking_strategy=ZeroMasking(),
)
explainer = ImageExplainer(model=architecture, data=image, random_state=42)
interaction_values = explainer.explain(budget=256)
fig, ax = interaction_values.plot_image_attributions(
    image, explainer.imputer.player_masks, show=False
)
ax.set_title("CNNArchitecture with explicit players and masking")
plt.show()

# %%
# :class:`~shapiq.vision.TransformerArchitecture` is constructed the same
# way (the ``processor`` is required):
# ``TransformerArchitecture(model=vit, processor=vit_processor,
# player_strategy=..., masking_strategy=...)``. The constructors run the
# same checks as dispatch, so an invalid combination -- a pixel-space
# strategy on ``TransformerArchitecture``, or token masking on a model that
# ignores it (like BEiT above) -- raises immediately instead of producing
# wrong attributions.
#
# Forcing Pixel Masking for a Transformer
# ---------------------------------------
# Any Hugging Face classifier can be explained on the pixel path by
# constructing ``CNNArchitecture`` with its processor -- useful when every
# model in a comparison should play the same pixel-space game, or when a
# model fails the token probe:

architecture = CNNArchitecture(model=vit, processor=vit_processor)
explainer = ImageExplainer(model=architecture, data=image, random_state=42)
interaction_values = explainer.explain(budget=128)
fig, ax = interaction_values.plot_image_attributions(
    image, explainer.imputer.player_masks, show=False
)
ax.set_title("ViT on the pixel path (superpixel players)")
plt.show()
