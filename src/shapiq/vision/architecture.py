from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .masking import (
    BoolMaskedPosStrategy,
    LatentMaskingStrategy,
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

if TYPE_CHECKING:
    from collections.abc import Sequence


class ModelArchitectureStrategy(ABC):
    """Encapsulates model-specific inference logic, decoupling it from ImageImputer."""

    #: The underlying model callable. Must be set by every concrete subclass.
    model: Any

    @abstractmethod
    def default_player_strategy(self) -> PlayerStrategy:
        """Return the default player strategy for this model."""
        ...

    @abstractmethod
    def default_masking_strategy(self) -> PixelMaskingStrategy:
        """Return the default masking strategy for this model."""
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
    def masking_strategy(self):
        """The active masking strategy. Settable to swap strategies after construction."""
        return self._masking_strategy

    @masking_strategy.setter
    def masking_strategy(self, value) -> None:
        self._masking_strategy = value

    @property
    def player_masks(self) -> np.ndarray | None:
        """Spatial masks per player, shape (n_players, H, W).

        Returns ``None`` for latent-space architectures and before ``prepare()`` is called.
        """
        return getattr(self, "_player_masks", None)


class ResNetArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for CNN models (e.g. ResNet) using pixel-space masking."""

    def __init__(self, model, masking_strategy: PixelMaskingStrategy | None = None):
        self.model = model
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_masks: np.ndarray | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        return MeanColorMasking()

    def prepare(self, image: np.ndarray, player_strategy: PixelPlayerStrategy) -> None:
        self._player_masks = player_strategy.get_masks(image)

    def value_function(self, image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        masked = self._masking_strategy.apply(image, self._player_masks, coalitions)
        return np.asarray(self.model(masked)).reshape(-1)

    def calc_empty_prediction(self, image: np.ndarray) -> float:
        n = self._player_masks.shape[0]
        empty_image = self._masking_strategy.apply(
            image, self._player_masks, np.zeros((1, n), dtype=bool)
        )[0]
        return float(self.model(empty_image[np.newaxis])[0])


class ViTArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for Vision Transformer models using latent-space masking."""

    def __init__(self, model, processor, masking_strategy: LatentMaskingStrategy | None = None):
        self.model = model
        self.processor = processor
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._pixel_values: torch.Tensor | None = None
        self._class_id: int | None = None
        self._player_strategy_ref: LatentPlayerStrategy | None = None

    def default_player_strategy(self) -> PatchStrategy:
        grid_size = self.model.config.image_size // self.model.config.patch_size
        return PatchStrategy(grid_size=grid_size, n_players=9)

    def default_masking_strategy(self) -> MaskTokenStrategy:
        # ViTForImageClassification has mask_token=None by default; MaskTokenStrategy initialises it
        return MaskTokenStrategy()

    def prepare(self, image: np.ndarray, player_strategy: LatentPlayerStrategy) -> None:
        self._player_strategy_ref = player_strategy
        inputs = self.processor(images=image, return_tensors="pt")
        self._pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            logits = self.model(pixel_values=self._pixel_values).logits
        self._class_id = int(logits.argmax(-1).item())

    def value_function(self, image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        bool_masks = torch.stack([self._player_strategy_ref.get_latent_mask(c) for c in coalitions])
        with torch.no_grad():
            logits = self._masking_strategy.predict_logits(
                self.model, self._pixel_values, bool_masks
            )
            probs = torch.softmax(logits, dim=-1)
        return probs[:, self._class_id].cpu().numpy()

    def calc_empty_prediction(self, image: np.ndarray) -> float:
        n = self._player_strategy_ref.n_players
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
        model,
        processor,
        class_id: int | None = None,
        masking_strategy: PixelMaskingStrategy | None = None,
    ):
        self.model = model
        self.processor = processor
        self.class_id = class_id
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_masks: np.ndarray | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        return MeanColorMasking()

    def prepare(self, image: np.ndarray, player_strategy: PixelPlayerStrategy) -> None:
        self._player_masks = player_strategy.get_masks(image)
        if self.class_id is None:
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits
            self.class_id = int(logits.argmax(-1).item())

    def _predict_batch(self, batch: np.ndarray) -> np.ndarray:
        # batch is (B, H, W, C)
        # HF processors accept lists of arrays / PILs.
        inputs = self.processor(images=[np.asarray(img) for img in batch], return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        return probs[:, self.class_id].cpu().numpy()

    def value_function(self, image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        masked = self._masking_strategy.apply(image, self._player_masks, coalitions)
        return self._predict_batch(masked)

    def calc_empty_prediction(self, image: np.ndarray) -> float:
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
        model,
        processor,
        masking_strategy: PixelMaskingStrategy | None = None,
    ):
        self.model = model
        self.processor = processor
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_masks: np.ndarray | None = None
        self._reference_embedding: torch.Tensor | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        return MeanColorMasking()

    def _embed(self, batch: np.ndarray) -> torch.Tensor:
        inputs = self.processor(images=[np.asarray(img) for img in batch], return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inputs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            emb = out.last_hidden_state[:, 0]  # CLS token
        else:
            emb = out
        return emb / emb.norm(dim=-1, keepdim=True)

    def prepare(self, image: np.ndarray, player_strategy: PixelPlayerStrategy) -> None:
        self._player_masks = player_strategy.get_masks(image)
        self._reference_embedding = self._embed(image[np.newaxis])[0]

    def value_function(self, image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        masked = self._masking_strategy.apply(image, self._player_masks, coalitions)
        emb = self._embed(masked)
        sims = emb @ self._reference_embedding
        return sims.cpu().numpy()

    def calc_empty_prediction(self, image: np.ndarray) -> float:
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
        model,
        processor,
        text_prompts: Sequence[str],
        target_prompt_idx: int = 0,
        masking_strategy: PixelMaskingStrategy | None = None,
    ):
        self.model = model
        self.processor = processor
        self.text_prompts = list(text_prompts)
        self.target_prompt_idx = target_prompt_idx
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_masks: np.ndarray | None = None
        self._text_features: torch.Tensor | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        return MeanColorMasking()

    def prepare(self, image: np.ndarray, player_strategy: PixelPlayerStrategy) -> None:
        self._player_masks = player_strategy.get_masks(image)
        text_inputs = self.processor(text=self.text_prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            tf = self.model.get_text_features(**text_inputs)
            # transformers >=5.x returns BaseModelOutputWithPooling instead of a tensor
            if not isinstance(tf, torch.Tensor):
                tf = (
                    tf.pooler_output
                    if (hasattr(tf, "pooler_output") and tf.pooler_output is not None)
                    else tf.last_hidden_state[:, 0]
                )
            self._text_features = torch.nn.functional.normalize(tf, dim=-1)

    def value_function(self, image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        masked = self._masking_strategy.apply(image, self._player_masks, coalitions)
        image_inputs = self.processor(
            images=[np.asarray(img) for img in masked], return_tensors="pt"
        )
        with torch.no_grad():
            f = self.model.get_image_features(pixel_values=image_inputs["pixel_values"])
            # transformers >=5.x may return a model output object instead of a tensor
            if not isinstance(f, torch.Tensor):
                f = (
                    f.pooler_output
                    if (hasattr(f, "pooler_output") and f.pooler_output is not None)
                    else f.last_hidden_state[:, 0]
                )
            f = torch.nn.functional.normalize(f, dim=-1)
            logits = self.model.logit_scale.exp() * f @ self._text_features.T
            probs = torch.softmax(logits, dim=-1)
        return probs[:, self.target_prompt_idx].cpu().numpy()

    def calc_empty_prediction(self, image: np.ndarray) -> float:
        n = self._player_masks.shape[0]
        return float(self.value_function(image, np.zeros((1, n), dtype=bool))[0])


class CustomViTArchitecture(ModelArchitectureStrategy):
    """ViT latent-space masking without HuggingFace dependencies.

    Useful for small or custom ViT models where a HuggingFace processor and
    ``model.config`` are not available. The caller pre-processes the image and
    supplies ``pixel_values`` and the explicit token grid directly.

    Args:
        model: A ViT-like model that accepts ``pixel_values`` and (optionally)
            ``bool_masked_pos``, and returns logits or an object with ``.logits``.
        pixel_values: Pre-processed ``(1, C, H, W)`` tensor for the image being explained.
        class_id: The class index to score.
        n_tokens: Number of patch tokens (e.g. 196 for a 14x14 grid).
        masking_strategy: Latent-space masking strategy. Defaults to ``BoolMaskedPosStrategy``.

    """

    def __init__(
        self,
        model,
        pixel_values: torch.Tensor,
        class_id: int,
        n_tokens: int,
        masking_strategy: LatentMaskingStrategy | None = None,
    ):
        self.model = model
        self._pixel_values = pixel_values
        self.class_id = class_id
        self.n_tokens = n_tokens
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy_ref: LatentPlayerStrategy | None = None

    def default_player_strategy(self) -> PatchStrategy:
        side = int(math.sqrt(self.n_tokens))
        if side * side != self.n_tokens:
            msg = (
                f"n_tokens ({self.n_tokens}) must be a perfect square for the "
                "default PatchStrategy. Provide an explicit player_strategy instead."
            )
            raise ValueError(msg)
        return PatchStrategy(grid_size=side, n_players=side * side)

    def default_masking_strategy(self) -> BoolMaskedPosStrategy:
        return BoolMaskedPosStrategy()

    def prepare(self, image: np.ndarray, player_strategy: LatentPlayerStrategy) -> None:
        self._player_strategy_ref = player_strategy

    def value_function(self, image: np.ndarray, coalitions: np.ndarray) -> np.ndarray:
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        bool_masks = torch.stack([self._player_strategy_ref.get_latent_mask(c) for c in coalitions])
        with torch.no_grad():
            out = self._masking_strategy.predict_logits(self.model, self._pixel_values, bool_masks)
            logits = out.logits if hasattr(out, "logits") else out
            probs = torch.softmax(logits, dim=-1)
        return probs[:, self.class_id].cpu().numpy()

    def calc_empty_prediction(self, image: np.ndarray) -> float:
        n = self._player_strategy_ref.n_players
        return float(self.value_function(image, np.zeros((1, n), dtype=bool))[0])
