# 🏞️ Vision for shapiq - Starter README

Explain image classifiers with Shapley values and Shapley interactions.

The idea is the same as everywhere else in `shapiq`: split the input into players, hide subsets of them, and watch what the prediction does.
For images a player is a region, either a group of pixels for CNNs or a group of patch tokens for Vision Transformers.

## Install

The vision code needs `torch` and `scikit-image`, which are not part of the base install:

```sh
pip install shapiq[vision]
```

## Quickstart Example

```python
import numpy as np
from PIL import Image
from torchvision.models import resnet18

from shapiq.vision import ClassificationArchitecture, ImageExplainer

image = np.asarray(Image.open("your_image.png").convert("RGB"))
model = resnet18(weights="IMAGENET1K_V1").eval()

architecture = ClassificationArchitecture(model=model)
explainer = ImageExplainer(model=architecture, data=image, index="k-SII", max_order=2)

values = explainer.explain(budget=256)
values.plot_image_attributions(image=image, player_masks=explainer.imputer.player_masks)
```

`data` can be a PIL image, an `(H, W, C)` numpy array, or a torch tensor.
Pass one image, not a batch.
The plotting helpers are stricter than the explainer and want a numpy array, so it is easiest to convert once up front.

By default the image is cut into roughly 10 SLIC superpixels and absent regions are filled with the mean colour.
The explained class defaults to whatever the model predicts, so pass `class_index=...` if you want a different one.

## Vision Transformers

ViTs get their own architecture, because masking happens in token space rather than on pixels:

```python
from transformers import ViTForImageClassification, ViTImageProcessor

from shapiq.vision import ImageExplainer, ViTClassificationArchitecture

name = "google/vit-base-patch32-384"
model = ViTForImageClassification.from_pretrained(name).eval()

architecture = ViTClassificationArchitecture(
    model=model,
    vit_processor=ViTImageProcessor.from_pretrained(name),
)
explainer = ImageExplainer(model=architecture, data=image)
```

Players are patch groups derived from the model's token grid, and absent players are hidden with the model's `mask_token`.
This needs a model that accepts `bool_masked_pos`, which most Hugging Face ViTs do.

## Things worth knowing

**Players and masking have to agree on a domain.**
Pixel-space players (`SuperpixelStrategy`, `GridStrategy`, `CustomPlayerStrategy`) go with pixel-space masking (`MeanColorMasking`, `ZeroMasking`), and token-space players (`PatchStrategy`) go with token-space masking (`MaskTokenStrategy`, `BoolMaskedPosStrategy`).
Mixing them raises a `TypeError` when you build the architecture rather than producing quiet nonsense later.

**Models without `bool_masked_pos` still work.**
Swin, BEiT, and friends can go through `ClassificationArchitecture` with their processor attached, which masks pixels before preprocessing:

```python
ClassificationArchitecture(model=model, processor=processor)
```

You lose token-space masking that way, but you keep the explanation.

**The two architectures return different scales.**
`ClassificationArchitecture` reports the raw logit of the explained class, `ViTClassificationArchitecture` reports the softmax probability.
Don't compare the numbers across the two without keeping that in mind.

**SLIC does not always return the number of segments you asked for.**
`n_players` reflects what you actually got, so read it off the imputer rather than assuming.

## Examples

Runnable versions of all of this live in `examples/vision`, including custom player layouts and the interaction network plots.
