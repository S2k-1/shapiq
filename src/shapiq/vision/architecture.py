"""Architecture strategies for vision model explanation."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from shapiq.explainer.utils import print_class

from .masking import (
    BoolMaskedPosStrategy,
    LatentMaskingStrategy,
    LayerMasking,
    ManifoldMaskingStrategy,
    MaskTokenStrategy,
    MeanColorMasking,
    PixelMaskingStrategy,
)
from .players import (
    LatentPlayerStrategy,
    PatchStrategy,
    PixelPlayerStrategy,
    PlayerStrategy,
    SuperpixelStrategy,
)
from .utils import get_torch_device, normalize_pixel_values, softmax_numpy, tensor_to_numpy

MaskingStrategy = PixelMaskingStrategy | LatentMaskingStrategy | ManifoldMaskingStrategy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    import torch

    from shapiq.typing import Model


def _extract_hf_features(output: torch.Tensor | object) -> torch.Tensor:
    """Return the feature tensor from a HuggingFace model output or a plain tensor.

    Transformers >= 5.x changed several ``get_*_features`` methods to return a
    ``BaseModelOutputWithPooling`` instead of a bare tensor.  This helper handles
    both shapes so callers don't have to repeat the same isinstance/hasattr chain.
    """
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    return output.last_hidden_state[:, 0]


def _build_patch_pixel_masks(image: np.ndarray, player_strategy: PatchStrategy) -> np.ndarray:
    """Map patch macro-regions to pixel-space masks for visualization."""
    n = player_strategy.n_players
    height, width = image.shape[:2]
    side = player_strategy.side
    masks = np.zeros((n, height, width), dtype=bool)
    for player_idx in range(n):
        row, col = divmod(player_idx, side)
        y0 = row * height // side
        y1 = height if row == side - 1 else (row + 1) * height // side
        x0 = col * width // side
        x1 = width if col == side - 1 else (col + 1) * width // side
        masks[player_idx, y0:y1, x0:x1] = True
    return masks


class ModelArchitectureStrategy(ABC):
    """Encapsulates model-specific inference logic, decoupling it from ImageImputer."""

    #: The underlying model callable. Must be set by every concrete subclass.
    model: Model

    @abstractmethod
    def default_player_strategy(self) -> PlayerStrategy:
        """Return the default player strategy for this architecture."""
        ...

    @abstractmethod
    def default_masking_strategy(self) -> MaskingStrategy:
        """Return the default masking strategy for this architecture."""
        ...

    @abstractmethod
    def prepare(self, image: np.ndarray, player_strategy: PlayerStrategy) -> None:
        """Cache image-dependent state. Called once by ImageImputer before value_function."""
        ...

    @abstractmethod
    def value_function(self, image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        """Return model predictions for each coalition. Returns (n_coalitions,)."""
        ...

    @abstractmethod
    def calc_empty_prediction(self, image: np.ndarray) -> float:
        """Return the model prediction for the empty coalition (all players absent)."""
        ...

    @property
    def masking_strategy(self) -> MaskingStrategy:
        """The active masking strategy. Settable to swap strategies after construction."""
        return self._masking_strategy

    @masking_strategy.setter
    def masking_strategy(self, value: MaskingStrategy) -> None:
        self._masking_strategy = value

    @property
    def player_masks(self) -> np.ndarray | None:
        """Spatial masks per player, shape (n_players, H, W).

        Returns ``None`` for latent-space architectures and before ``prepare()`` is called.
        """
        return getattr(self, "_player_masks", None)


class ResNetArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for CNN models (e.g. ResNet) using pixel-space masking."""

    def __init__(self, model: Model, masking_strategy: PixelMaskingStrategy | None = None) -> None:
        """Initialize the ResNetArchitecture.

        Args:
            model: A CNN model callable (e.g. a ResNet from torchvision).
            masking_strategy: Pixel-space masking strategy. Defaults to ``MeanColorMasking``.
        """
        self.model = model
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_masks: np.ndarray | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        """Return the default superpixel player strategy."""
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        """Return the default mean-color masking strategy."""
        return MeanColorMasking()

    def prepare(self, image: np.ndarray, player_strategy: PixelPlayerStrategy) -> None:
        """Precompute player masks for the given image."""
        self._player_masks = player_strategy.get_masks(image)

    def value_function(self, image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        """Apply pixel masking and return model predictions for each coalition."""
        masked = self._masking_strategy.apply(image, self._player_masks, coalitions)
        return np.asarray(self.model(masked)).reshape(-1)

    def calc_empty_prediction(self, image: np.ndarray) -> float:
        """Return the model prediction for the empty coalition."""
        n = self._player_masks.shape[0]
        return float(self.value_function(image, np.zeros((1, n), dtype=bool))[0])


class ViTArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for Vision Transformer models using latent-space masking."""

    def __init__(
        self,
        model: Model,
        processor: object,
        masking_strategy: LatentMaskingStrategy | None = None,
    ) -> None:
        """Initialize the ViTArchitecture.

        Args:
            model: A HuggingFace ``ViTForImageClassification`` model.
            processor: A HuggingFace image processor (e.g. ``ViTImageProcessor``).
            masking_strategy: Latent-space masking strategy. Defaults to ``MaskTokenStrategy``.
        """
        self.model = model
        self.processor = processor
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._pixel_values: torch.Tensor | None = None
        self._class_id: int | None = None
        self._player_strategy_ref: LatentPlayerStrategy | None = None
        self._player_masks: np.ndarray | None = None

    def default_player_strategy(self) -> PatchStrategy:
        """Return the default patch player strategy derived from the model config."""
        grid_size = self.model.config.image_size // self.model.config.patch_size
        return PatchStrategy(grid_size=grid_size, n_players=9)

    def default_masking_strategy(self) -> MaskTokenStrategy:
        """Return the default mask-token masking strategy."""
        # ViTForImageClassification has mask_token=None by default; MaskTokenStrategy initialises it
        return MaskTokenStrategy()

    def prepare(self, image: np.ndarray, player_strategy: LatentPlayerStrategy) -> None:
        """Pre-process the image and cache pixel values and the predicted class."""
        import torch

        self._player_strategy_ref = player_strategy
        device = get_torch_device(self.model)
        inputs = self.processor(images=image, return_tensors="pt")
        self._pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            logits = self.model(pixel_values=self._pixel_values).logits
        self._class_id = int(logits.argmax(-1).item())
        if isinstance(player_strategy, PatchStrategy):
            self._player_masks = _build_patch_pixel_masks(image, player_strategy)

    def value_function(self, _image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        """Apply latent masking and return class probabilities for each coalition."""
        import torch

        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        device = self._pixel_values.device
        bool_masks = torch.stack(
            [self._player_strategy_ref.get_latent_mask(c) for c in coalitions]
        ).to(device)
        with torch.no_grad():
            logits = self._masking_strategy.predict_logits(
                self.model, self._pixel_values, bool_masks
            )
            probs = torch.softmax(logits, dim=-1)
        return tensor_to_numpy(probs[:, self._class_id])

    def calc_empty_prediction(self, image: np.ndarray) -> float:
        """Return the model prediction for the empty coalition."""
        n = self._player_strategy_ref.n_players
        return float(self.value_function(image, np.zeros((1, n), dtype=bool))[0])


class LayerMaskedCNNArchitecture(ModelArchitectureStrategy):
    """CNN architecture that masks intermediate activations instead of input pixels.

    Pairs a raw ``torch.nn.Module`` (e.g. ``torchvision.models.resnet18``) with
    a :class:`~shapiq.vision.masking.ManifoldMaskingStrategy` such as
    :class:`~shapiq.vision.masking.LayerMasking`. A forward hook installed by
    the strategy attenuates activations at the chosen layer, so the model never
    receives a pixel-space replacement and the missingness bias of mean / zero
    / blur masking is avoided. The CNN analogue of using
    :class:`~shapiq.vision.architecture.ViTArchitecture` with
    :class:`~shapiq.vision.masking.MaskTokenStrategy`.

    Args:
        model: A raw ``torch.nn.Module`` (not a pre-wrapped callable). Layer
            hooks need direct access to submodules.
        preprocess: Callable mapping a ``(H, W, C)`` image array to a
            ``(C, H, W)`` tensor (e.g. ``ResNet18_Weights.DEFAULT.transforms()``).
        class_id: Class index whose probability is returned as the value.
        masking_strategy: Manifold masking strategy. Defaults to
            :class:`~shapiq.vision.masking.LayerMasking` hooking ``"layer2"``.

    Example::

        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights).eval()
        arch = LayerMaskedCNNArchitecture(
            model=model,
            preprocess=weights.transforms(),
            class_id=281,  # tabby cat
        )
        explainer = ImageExplainer(architecture=arch, data=image)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        preprocess: Callable[[np.ndarray], torch.Tensor],
        class_id: int,
        masking_strategy: ManifoldMaskingStrategy | None = None,
    ) -> None:
        """Initialize the LayerMaskedCNNArchitecture.

        Args:
            model: A raw ``torch.nn.Module``.
            preprocess: Callable ``f((H, W, C) ndarray) -> (C, H, W) tensor``.
            class_id: Class index whose probability is returned.
            masking_strategy: A :class:`ManifoldMaskingStrategy`. Defaults to
                :class:`LayerMasking` hooking ``"layer2"``.
        """
        self.model = model
        self.preprocess = preprocess
        self.class_id = class_id
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_masks: np.ndarray | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        """Return the default superpixel player strategy."""
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> LayerMasking:
        """Return the default layer-masking strategy (hooks ``layer2``)."""
        return LayerMasking(layer_name="layer2")

    def prepare(self, image: np.ndarray, player_strategy: PixelPlayerStrategy) -> None:
        """Precompute player masks for the given image."""
        self._player_masks = player_strategy.get_masks(image)

    def value_function(self, image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        """Delegate to the manifold masking strategy."""
        return self._masking_strategy.value_function(
            self.model,
            self.preprocess,
            image,
            self._player_masks,
            coalitions,
            self.class_id,
        )

    def calc_empty_prediction(self, image: np.ndarray) -> float:
        """Return the model prediction for the empty coalition."""
        n = self._player_masks.shape[0]
        return float(self.value_function(image, np.zeros((1, n), dtype=bool))[0])


class HuggingFacePixelArchitecture(ModelArchitectureStrategy):
    """Pixel-space masking for HuggingFace classification models.

    Works with any HuggingFace vision model whose forward pass accepts
    ``pixel_values`` and returns an object with a ``.logits`` attribute

    Args:
        model: A HuggingFace model. Must be callable and return an output with ``.logits``.
        processor: A HuggingFace image processor (e.g. ``AutoImageProcessor``).
        class_id: The class index to score. If ``None``, auto-detected once from the
            argmax prediction on the original (unmasked) image during the first call to
            ``prepare``. Subsequent calls to ``prepare`` reuse the cached value. To
            re-detect for a new image, set ``arch.class_id = None`` before calling
            ``prepare`` again.
        masking_strategy: Pixel-space masking strategy. Defaults to ``MeanColorMasking``.

    """

    def __init__(
        self,
        model: Model,
        processor: object,
        class_id: int | None = None,
        masking_strategy: PixelMaskingStrategy | None = None,
    ) -> None:
        """Initialize the HuggingFacePixelArchitecture.

        Args:
            model: A HuggingFace model returning an output with ``.logits``.
            processor: A HuggingFace image processor.
            class_id: Class index to score; auto-detected from first ``prepare`` call if ``None``.
            masking_strategy: Pixel-space masking strategy. Defaults to ``MeanColorMasking``.
        """
        self.model = model
        self.processor = processor
        self.class_id = class_id
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_masks: np.ndarray | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        """Return the default superpixel player strategy."""
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        """Return the default mean-color masking strategy."""
        return MeanColorMasking()

    def prepare(self, image: np.ndarray, player_strategy: PixelPlayerStrategy) -> None:
        """Precompute player masks and auto-detect class ID if not set."""
        import torch

        self._player_masks = player_strategy.get_masks(image)
        if self.class_id is None:
            device = get_torch_device(self.model)
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                logits = self.model(pixel_values=pixel_values).logits
            self.class_id = int(logits.argmax(-1).item())

    def _predict_batch(self, batch: np.ndarray) -> np.ndarray:
        import torch

        device = get_torch_device(self.model)
        inputs = self.processor(images=[np.asarray(img) for img in batch], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            logits = self.model(pixel_values=pixel_values).logits
            probs = torch.softmax(logits, dim=-1)
        return tensor_to_numpy(probs[:, self.class_id])

    def value_function(self, image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        """Apply pixel masking and return class probabilities for each coalition."""
        masked = self._masking_strategy.apply(image, self._player_masks, coalitions)
        return self._predict_batch(masked)

    def calc_empty_prediction(self, image: np.ndarray) -> float:
        """Return the model prediction for the empty coalition."""
        n = self._player_masks.shape[0]
        return float(self.value_function(image, np.zeros((1, n), dtype=bool))[0])


class ConvNeXtArchitecture(HuggingFacePixelArchitecture):
    """Architecture strategy for HuggingFace ConvNeXt classification models.

    Thin specialization of :class:`HuggingFacePixelArchitecture` — exposed as a
    named class so users and demos can refer to ConvNeXt directly. All behavior
    is inherited.
    """


class DINOv2Architecture(ModelArchitectureStrategy):
    """Architecture for DINOv2 (and similar SSL backbones) via feature similarity.

    DINOv2 checkpoints are typically released without a classification head, so the
    natural score for an explanation is the cosine similarity between the embedding
    of a masked image and the embedding of the original unmasked image. The
    reference embedding is computed once in ``prepare``.

    Works with any backbone whose forward pass returns one of:

    - ``.pooler_output``
    - ``.last_hidden_state`` (the CLS token at index 0 is used)
    - a tensor directly

    Args:
        model: A backbone (callable).
        processor: A HuggingFace image processor.
        masking_strategy: Pixel-space masking strategy. Defaults to ``MeanColorMasking``.

    """

    def __init__(
        self,
        model: Model,
        processor: object,
        masking_strategy: PixelMaskingStrategy | None = None,
    ) -> None:
        """Initialize the DINOv2Architecture.

        Args:
            model: A backbone model callable.
            processor: A HuggingFace image processor.
            masking_strategy: Pixel-space masking strategy. Defaults to ``MeanColorMasking``.
        """
        self.model = model
        self.processor = processor
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_masks: np.ndarray | None = None
        self._reference_embedding: torch.Tensor | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        """Return the default superpixel player strategy."""
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        """Return the default mean-color masking strategy."""
        return MeanColorMasking()

    def _embed(self, batch: np.ndarray) -> torch.Tensor:
        import torch

        device = get_torch_device(self.model)
        inputs = self.processor(images=[np.asarray(img) for img in batch], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            out = self.model(pixel_values=pixel_values)
        emb = _extract_hf_features(out)
        return emb / emb.norm(dim=-1, keepdim=True)

    def prepare(self, image: np.ndarray, player_strategy: PixelPlayerStrategy) -> None:
        """Precompute player masks and reference embedding for the image."""
        self._player_masks = player_strategy.get_masks(image)
        self._reference_embedding = self._embed(image[np.newaxis])[0]

    def value_function(self, image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        """Apply pixel masking and return cosine similarity scores for each coalition."""
        masked = self._masking_strategy.apply(image, self._player_masks, coalitions)
        emb = self._embed(masked)
        sims = emb @ self._reference_embedding
        return tensor_to_numpy(sims)

    def calc_empty_prediction(self, image: np.ndarray) -> float:
        """Return the model prediction for the empty coalition."""
        n = self._player_masks.shape[0]
        return float(self.value_function(image, np.zeros((1, n), dtype=bool))[0])


class CLIPArchitecture(ModelArchitectureStrategy):
    """Architecture for CLIP zero-shot image classification.

    The "class" being explained is one of a list of text prompts. The value
    function returns the softmax probability over text prompts for the target
    prompt index, given each (masked) image.

    Args:
        model: A CLIP model (e.g. HF ``CLIPModel``) with ``get_image_features``,
            ``get_text_features``, and a ``logit_scale`` parameter.
        processor: A HF CLIP processor (handles both text and image branches).
        text_prompts: List of text prompts to compare against.
        target_prompt_idx: Index of the prompt to score. Defaults to ``0``.
        masking_strategy: Pixel-space masking strategy. Defaults to ``MeanColorMasking``.

    """

    def __init__(
        self,
        model: Model,
        processor: object,
        text_prompts: Sequence[str],
        target_prompt_idx: int = 0,
        masking_strategy: PixelMaskingStrategy | None = None,
    ) -> None:
        """Initialize the CLIPArchitecture.

        Args:
            model: A CLIP model with ``get_image_features``, ``get_text_features``, and
                ``logit_scale``.
            processor: A HF CLIP processor.
            text_prompts: List of text prompts to compare against.
            target_prompt_idx: Index of the target prompt to score.
            masking_strategy: Pixel-space masking strategy. Defaults to ``MeanColorMasking``.
        """
        self.model = model
        self.processor = processor
        self.text_prompts = list(text_prompts)
        self.target_prompt_idx = target_prompt_idx
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_masks: np.ndarray | None = None
        self._text_features: torch.Tensor | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        """Return the default superpixel player strategy."""
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        """Return the default mean-color masking strategy."""
        return MeanColorMasking()

    def prepare(self, image: np.ndarray, player_strategy: PixelPlayerStrategy) -> None:
        """Precompute player masks and text features for the configured prompts."""
        import torch

        self._player_masks = player_strategy.get_masks(image)
        device = get_torch_device(self.model)
        text_inputs = self.processor(text=self.text_prompts, return_tensors="pt", padding=True)
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        with torch.no_grad():
            tf = _extract_hf_features(self.model.get_text_features(**text_inputs))
            self._text_features = torch.nn.functional.normalize(tf, dim=-1)

    def value_function(self, image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        """Apply pixel masking and return CLIP similarity scores for each coalition."""
        import torch

        masked = self._masking_strategy.apply(image, self._player_masks, coalitions)
        device = get_torch_device(self.model)
        image_inputs = self.processor(
            images=[np.asarray(img) for img in masked], return_tensors="pt"
        )
        pixel_values = image_inputs["pixel_values"].to(device)
        with torch.no_grad():
            f = _extract_hf_features(self.model.get_image_features(pixel_values=pixel_values))
            f = torch.nn.functional.normalize(f, dim=-1)
            logits = self.model.logit_scale.exp() * f @ self._text_features.T
            probs = torch.softmax(logits, dim=-1)
        return tensor_to_numpy(probs[:, self.target_prompt_idx])

    def calc_empty_prediction(self, image: np.ndarray) -> float:
        """Return the model prediction for the empty coalition."""
        n = self._player_masks.shape[0]
        return float(self.value_function(image, np.zeros((1, n), dtype=bool))[0])


class CustomViTArchitecture(ModelArchitectureStrategy):
    """ViT latent-space masking without HuggingFace dependencies.

    Useful for small or custom ViT models where a HuggingFace processor and
    ``model.config`` are not available. The caller pre-processes the image and
    supplies ``pixel_values`` and the explicit token grid directly.

    Accepts numpy arrays for framework-agnostic callables (including JAX/Flax
    models wrapped as ``numpy in → numpy out`` callables) or PyTorch tensors for
    torch-native models.

    Args:
        model: A ViT-like model that accepts ``pixel_values`` and (optionally)
            ``bool_masked_pos``, and returns logits or an object with ``.logits``.
        pixel_values: Pre-processed ``(1, C, H, W)`` array or tensor for the image being explained.
        class_id: The class index to score.
        n_tokens: Number of patch tokens (e.g. 196 for a 14x14 grid).
        masking_strategy: Latent-space masking strategy. Defaults to ``BoolMaskedPosStrategy``.

    """

    def __init__(
        self,
        model: Model,
        pixel_values: np.ndarray | torch.Tensor,
        class_id: int,
        n_tokens: int,
        masking_strategy: LatentMaskingStrategy | None = None,
    ) -> None:
        """Initialize the CustomViTArchitecture.

        Args:
            model: A ViT-like model callable.
            pixel_values: Pre-processed ``(1, C, H, W)`` array or tensor.
            class_id: The class index to score.
            n_tokens: Number of patch tokens.
            masking_strategy: Latent-space masking strategy. Defaults to
                ``BoolMaskedPosStrategy``.
        """
        self.model = model
        self._pixel_values, self._uses_torch = normalize_pixel_values(pixel_values)
        self.class_id = class_id
        self.n_tokens = n_tokens
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy_ref: LatentPlayerStrategy | None = None

    def default_player_strategy(self) -> PatchStrategy:
        """Return a PatchStrategy derived from the number of tokens."""
        side = int(math.sqrt(self.n_tokens))
        if side * side != self.n_tokens:
            msg = (
                f"n_tokens ({self.n_tokens}) must be a perfect square for the "
                "default PatchStrategy. Provide an explicit player_strategy instead."
            )
            raise ValueError(msg)
        return PatchStrategy(grid_size=side, n_players=side * side)

    def default_masking_strategy(self) -> BoolMaskedPosStrategy:
        """Return the default bool-masked-pos masking strategy."""
        return BoolMaskedPosStrategy()

    def prepare(self, _image: np.ndarray, player_strategy: LatentPlayerStrategy) -> None:
        """Cache the player strategy reference and align pixel values with the model device."""
        if self._uses_torch:
            device = get_torch_device(self.model)
            self._pixel_values = self._pixel_values.to(device)
        self._player_strategy_ref = player_strategy

    def value_function(self, _image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        """Apply latent masking and return class probabilities for each coalition."""
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        if self._uses_torch:
            import torch

            device = self._pixel_values.device
            bool_masks = torch.stack(
                [self._player_strategy_ref.get_latent_mask(c) for c in coalitions]
            ).to(device)
            with torch.no_grad():
                logits = self._masking_strategy.predict_logits(
                    self.model, self._pixel_values, bool_masks
                )
                probs = torch.softmax(logits, dim=-1)
            return tensor_to_numpy(probs[:, self.class_id])

        bool_masks = np.stack(
            [self._player_strategy_ref.get_latent_mask_array(c) for c in coalitions]
        )
        logits = self._masking_strategy.predict_logits(self.model, self._pixel_values, bool_masks)
        probs = softmax_numpy(np.asarray(logits))
        return probs[:, self.class_id]

    def calc_empty_prediction(self, image: np.ndarray) -> float:
        """Return the model prediction for the empty coalition."""
        n = self._player_strategy_ref.n_players
        return float(self.value_function(image, np.zeros((1, n), dtype=bool))[0])


def get_architecture_for_model(
    model: Model,
    *,
    architecture: ModelArchitectureStrategy | None = None,
    processor: object | None = None,
    **kwargs: Any,
) -> ModelArchitectureStrategy:
    """Pick a default :class:`ModelArchitectureStrategy` for ``model``.

    Used by :class:`~shapiq.explainer.base.Explainer` when ``data`` is image-like.
    Pass ``architecture=`` explicitly to override inference. HuggingFace models that
    need a processor (ViT, ConvNeXt, CLIP, DINOv2) require ``processor=`` unless the
    model is already wrapped in a strategy instance.

    Args:
        model: The model to explain, or an existing architecture strategy.
        architecture: Explicit architecture override.
        processor: HuggingFace image processor, when required by the detected model type.
        **kwargs: Extra constructor args (e.g. ``text_prompts`` for CLIP).

    Returns:
        A ready-to-use architecture strategy.

    Raises:
        TypeError: If a HuggingFace model needs extra arguments that were not provided.
    """
    if architecture is not None:
        return architecture
    if isinstance(model, ModelArchitectureStrategy):
        return model

    try:
        from transformers import CLIPModel, ViTForImageClassification
    except ImportError:
        CLIPModel = None  # type: ignore[misc, assignment]
        ViTForImageClassification = None  # type: ignore[misc, assignment]

    if ViTForImageClassification is not None and isinstance(model, ViTForImageClassification):
        if processor is None:
            msg = (
                "ViT models require `processor=` when using shapiq.Explainer auto-dispatch. "
                "Pass `architecture=ViTArchitecture(model, processor)` explicitly."
            )
            raise TypeError(msg)
        return ViTArchitecture(model=model, processor=processor)

    if CLIPModel is not None and isinstance(model, CLIPModel):
        if processor is None:
            msg = "CLIP models require `processor=` for auto-dispatch."
            raise TypeError(msg)
        text_prompts = kwargs.get("text_prompts")
        if text_prompts is None:
            msg = (
                "CLIP models require `text_prompts=` when using shapiq.Explainer "
                "auto-dispatch. Pass `architecture=CLIPArchitecture(...)` explicitly "
                "or supply `text_prompts`."
            )
            raise TypeError(msg)
        return CLIPArchitecture(
            model=model,
            processor=processor,
            text_prompts=text_prompts,
            target_prompt_idx=kwargs.get("target_prompt_idx", 0),
        )

    try:
        import torchvision.models as tv_models
    except ImportError:
        tv_models = None  # type: ignore[assignment]

    if tv_models is not None and isinstance(model, tv_models.ResNet):
        return ResNetArchitecture(model=model)

    model_class = print_class(model)

    if processor is not None:
        if "ViTForImageClassification" in model_class or (
            hasattr(model, "vit") and "ForImageClassification" in model_class
        ):
            return ViTArchitecture(model=model, processor=processor)
        if "CLIPModel" in model_class:
            text_prompts = kwargs.get("text_prompts")
            if text_prompts is None:
                msg = (
                    "CLIP models require `text_prompts=` when using shapiq.Explainer "
                    "auto-dispatch. Pass `architecture=CLIPArchitecture(...)` explicitly "
                    "or supply `text_prompts`."
                )
                raise TypeError(msg)
            return CLIPArchitecture(
                model=model,
                processor=processor,
                text_prompts=text_prompts,
                target_prompt_idx=kwargs.get("target_prompt_idx", 0),
            )
        if "Dinov2" in model_class or "DINOv2" in model_class:
            return DINOv2Architecture(model=model, processor=processor)
        if "ConvNeXt" in model_class:
            return ConvNeXtArchitecture(model=model, processor=processor)
        if "ForImageClassification" in model_class:
            return HuggingFacePixelArchitecture(model=model, processor=processor)
        return HuggingFacePixelArchitecture(model=model, processor=processor)

    if hasattr(model, "vit"):
        msg = (
            "ViT models require `processor=` when using shapiq.Explainer auto-dispatch. "
            "Pass `architecture=ViTArchitecture(model, processor)` explicitly."
        )
        raise TypeError(msg)

    return ResNetArchitecture(model=model)
