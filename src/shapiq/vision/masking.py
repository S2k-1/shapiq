from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class PixelMaskingStrategy(ABC):
    @abstractmethod
    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        """Args:
            image:        (H, W, C) original image
            player_masks: (n_players, H, W) boolean masks per player
            coalition:    (n_coalitions, n_players) boolean array

        Returns:
            masked_images: (n_coalitions, H, W, C)
        """
        ...


def _apply_pixel_masking(
    image: np.ndarray,
    player_masks: np.ndarray,
    coalition: np.ndarray,
    fill: float | np.ndarray,
) -> np.ndarray:
    """Build masked images by replacing absent-player pixels with *fill*.

    Args:
        image:        (H, W, C) original image.
        player_masks: (n_players, H, W) boolean masks per player.
        coalition:    (n_coalitions, n_players) boolean array; True = player present.
        fill:         Scalar or (C,) array used as the replacement value.

    Returns:
        masked_images: (n_coalitions, H, W, C)
    """
    n_coalitions, n_players = coalition.shape
    H, W, _ = image.shape

    masked_images = np.stack([image] * n_coalitions, axis=0)

    # For each coalition, a pixel is absent if at least one absent player covers it.
    # This reduces to a matrix product: absent @ flat_masks > 0.
    absent = (~coalition).view(np.uint8)  # (n_coalitions, n_players)
    flat_masks = player_masks.reshape(n_players, H * W).view(np.uint8)  # (n_players, H*W)
    absent_mask = (absent @ flat_masks).reshape(n_coalitions, H, W).astype(bool)

    masked_images[absent_mask] = fill
    return masked_images


class MeanColorMasking(PixelMaskingStrategy):
    """Imputes the masked pixels with the mean color of the entire image."""

    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        return _apply_pixel_masking(image, player_masks, coalition, image.mean(axis=(0, 1)))


class ZeroMasking(PixelMaskingStrategy):
    """Imputes masked pixels with a constant value (default: black / 0.0)."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        return _apply_pixel_masking(image, player_masks, coalition, self.value)


class LatentMaskingStrategy(ABC):
    """Defines how tokens are masked in latent/embedding space."""

    @abstractmethod
    def predict_logits(
        self,
        model,
        pixel_values: torch.Tensor,  # (1, 3, H, W)
        bool_masks: torch.Tensor,  # (B, n_tokens)
    ) -> torch.Tensor:  # (B, n_classes)
        ...


def _ensure_vit_mask_token(model, device: torch.device) -> None:
    """Initialize mask_token to zeros when it is None.

    ``ViTForImageClassification`` stores ``mask_token = None`` in its
    embeddings module because it was not pretrained with masked-image
    modelling. Both latent masking strategies pass ``bool_masked_pos`` to the
    model's forward pass, which unconditionally calls
    ``self.mask_token.expand(...)`` inside ``ViTEmbeddings.forward``. Without
    this guard that line crashes with an ``AttributeError``.
    """
    if not (hasattr(model, "vit") and hasattr(model.vit, "embeddings")):
        return
    emb = model.vit.embeddings
    if getattr(emb, "mask_token", None) is None:
        emb.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, model.config.hidden_size, device=device)
        )


class BoolMaskedPosStrategy(LatentMaskingStrategy):
    """Masks tokens via the bool_masked_pos argument in the model forward pass."""

    def predict_logits(self, model, pixel_values, bool_masks):
        _ensure_vit_mask_token(model, pixel_values.device)
        batch = pixel_values.repeat(bool_masks.shape[0], 1, 1, 1)
        return model(pixel_values=batch, bool_masked_pos=bool_masks).logits


class MaskTokenStrategy(LatentMaskingStrategy):
    """Masks tokens by zeroing the mask_token embedding before the forward pass."""

    def predict_logits(self, model, pixel_values, bool_masks):
        # Ensure mask_token exists (ViTForImageClassification leaves it None by default),
        # then zero it in-place so absent patches carry no signal.  Re-creating the
        # nn.Parameter on every call would replace the model's parameter dict entry.
        _ensure_vit_mask_token(model, pixel_values.device)
        if hasattr(model, "vit") and hasattr(model.vit, "embeddings"):
            model.vit.embeddings.mask_token.data.zero_()
        batch = pixel_values.repeat(bool_masks.shape[0], 1, 1, 1)
        return model(pixel_values=batch, bool_masked_pos=bool_masks).logits
