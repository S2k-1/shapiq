"""
Image Explanations with shapiq: CNN and Vision Transformer
============================================================

This example demonstrates how to explain image classifiers using ``shapiq``'s vision package. 
We cover two common architectures side-by-side:

- A **ResNet-18** (CNN) explained using SLIC superpixels and mean-color masking
- A **Vision Transformer (ViT)** explained using patch-token masking

For both models we compute first-order Shapley values and second-order
interaction indices (k-SII), and visualise how individual image regions
and region pairs impact the predicted class.

We use a sample image from the ImageNet dataset.
"""

from __future__ import annotations

import os

# Prevent OpenMP / MKL thread conflicts with PyTorch
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# %%
# Load the ImageNet Sample Image
# -------------------------------
# We load a single ImageNet validation image which will be passed to both models.

from pathlib import Path
import numpy as np
from PIL import Image

image_path = Path("imagenet_sample.png")
pil_image = Image.open(image_path).convert("RGB")
pil_image

# %%
# Part 1: Explaining a CNN (ResNet-18) with Superpixels
# ======================================================
#
# ResNets process images at the pixel level, so we define players as
# compact image regions (superpixels).  Absent players are replaced
# by the per-channel mean colour of the original image.

import torchvision.models as models
from torchvision import transforms

from shapiq.vision import ImageExplainer
from shapiq.vision.architecture import CNNArchitecture

# %%
# Load a pre-trained ResNet-18 and set it to evaluation mode.

resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.eval()

# %%
# Resize the image and normalize the pixel values as the loaded model expects.

# Resize and crop
resize_and_crop = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])

# Convert to tensor and normalize
tensor_and_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resized_image = resize_and_crop(pil_image)
tensor_image = tensor_and_norm(resized_image)

# %%
# Build the architecture strategy.  The
# :class:`~shapiq.vision.architecture.CNNArchitecture` wraps the model and
# handles the forward pass. By default it will compute SLIC superpixel masks
# aiming at around 16 players and apply mean-color masking for absent players.
# To customise these values you can pass a
# :class:`~shapiq.vision.players.CNNPlayerStrategy` and a
# :class:`~shapiq.vision.masking.CNNMaskingStrategy`.

cnn_arch = CNNArchitecture(
    model=resnet,
)

# %%
# Create the :class:`~shapiq.vision.explainer.ImageExplainer`.  We choose
# ``index="k-SII"`` and ``max_order=2`` from the start so we get
# Shapley values *and* pairwise interaction indices from a single run.
# You can pass the image as a PIL image, preprocessed tensor, or numpy array.

cnn_explainer = ImageExplainer(
    model_architecture=cnn_arch,
    data=tensor_image,
    index="k-SII",
    max_order=2,
    batch_size=32,
)

print(f"Number of superpixel players: {cnn_explainer.imputer.n_features}")

# %%
# Compute Shapley Interaction Values
# -----------------------------------
# :class:`~shapiq.KernelSHAPIQ` is selected automatically by the explainer.
# A budget of ``256`` evaluations is sufficient for 16 players.

cnn_iv = cnn_explainer.explain(budget=256)
print(cnn_iv)

# %%
# Visualize Superpixel Importance
# --------------------------------
# The heatmap overlay shows the Shapley value of each superpixel, with red
# indicating positive contributions and blue indicating negative contributions.

cnn_iv.plot_image_attributions(
    image=np.array(resized_image),
    explainer=cnn_explainer,
    heatmap_only=False,
)

# %%
# Visualize Pairwise Interaction Network
# ----------------------------------------
# The network plot encodes second-order k-SII values as edge weights.
# Positive (blue) edges indicate synergy between superpixel pairs;
# negative (red) edges indicate redundancy or suppression.

player_names_cnn = [f"SP {i}" for i in range(cnn_explainer.imputer.n_features)]
cnn_iv.plot_network(feature_names=player_names_cnn)

# %%
# Part 2: Explaining a Vision Transformer (ViT-B/16) with Patch Masking
# =======================================================================
#
# Vision Transformers split the image into fixed-size patch tokens.  Absent
# players are masked *in latent space* by zeroing the mask-token embedding
# before the forward pass — the same mechanism used during masked image
# modelling pre-training — which avoids pixel-space distribution shifts.
#
# The :class:`~shapiq.vision.architecture.TransformerArchitecture` uses the
# Hugging Face processor for preprocessing and runs a batched forward pass.
# By default it groups the ViT's token grid into 9 players (3×3 super-patches)
# and applies :class:`~shapiq.vision.masking.MaskTokenStrategy` for masking.
# To customise, pass a :class:`~shapiq.vision.players.TransformerPlayerStrategy`
# and a :class:`~shapiq.vision.masking.TransformerMaskingStrategy`.

from transformers import ViTForImageClassification, ViTImageProcessor

from shapiq.vision.architecture import TransformerArchitecture

# %%
# Load ViT-B/16 from Hugging Face.  The model is pretrained on ImageNet-21k
# and fine-tuned on ImageNet-1k — the same label space as ResNet-18 above.

vit_name = "google/vit-base-patch32-384"
vit_processor = ViTImageProcessor.from_pretrained(vit_name)
vit_model = ViTForImageClassification.from_pretrained(vit_name)
vit_model.eval()

# %%
# Build the architecture strategy.  The :class:`~shapiq.vision.architecture.TransformerArchitecture`
# wraps the model and the Hugging Face processor. Preprocessing (resizing,
# normalization, conversion to ``pixel_values``) is handled internally by
# the processor, so we pass the original PIL image directly.

vit_arch = TransformerArchitecture(
    model=vit_model,
    vit_processor=vit_processor,
)

# %%
# Create the :class:`~shapiq.vision.explainer.ImageExplainer` for the ViT.
# We again use ``index="k-SII"`` and ``max_order=2`` for a combined
# first- and second-order explanation in a single run.

vit_explainer = ImageExplainer(
    model_architecture=vit_arch,
    data=pil_image,
    index="k-SII",
    max_order=2,
    batch_size=32,
)

print(f"Number of patch players (ViT): {vit_explainer.imputer.n_features}")

# %%
# Compute Shapley Interaction Values (ViT)
# -----------------------------------------
# The same :class:`~shapiq.KernelSHAPIQ` approximator is used. With 9
# players a budget of ``256`` is more than sufficient for reliable estimates.

vit_iv = vit_explainer.explain(budget=50)
print(vit_iv)

# %%
# Visualize Patch Importance
# ---------------------------
# Player names encode the 3×3 grid position for interpretability.

vit_iv.plot_image_attributions(
    image=np.array(pil_image),
    explainer=vit_explainer,
    heatmap_only=False,
)

# %%
# Visualize Pairwise Interaction Network (ViT)
# ----------------------------------------------
# The network plot highlights which patch pairs reinforce or suppress each
# other's contribution.
player_names_vit = [f"Patch {i}" for i in range(vit_explainer.imputer.n_features)]
vit_iv.plot_network(feature_names=player_names_vit, center_image=resized_image)

# %%
# References
# ----------
# .. footbibliography::