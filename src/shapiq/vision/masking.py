"""Masking strategies for vision-based explanations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from shapiq.typing import Model


class PixelMaskingStrategy(ABC):
    """Abstract base class for pixel-space masking strategies."""

    @abstractmethod
    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        """Apply the masking strategy to a batch of coalitions.

        Args:
            image:        ``(H, W, C)`` original image.
            player_masks: ``(n_players, H, W)`` boolean masks per player.
            coalition:    ``(n_coalitions, n_players)`` boolean array; ``True`` = player present.

        Returns:
            masked_images: ``(n_coalitions, H, W, C)`` array of masked images.
        """
        ...


class MeanColorMasking(PixelMaskingStrategy):
    """Imputes absent players' regions with the mean color of the entire image."""

    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        """Apply mean-color masking to a batch of coalitions."""
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape

        masked_images = np.stack([image] * n_coalitions, axis=0)

        mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    mask[i] |= player_masks[j]

        masked_images[mask] = image.mean(axis=(0, 1))
        return masked_images


class ZeroMasking(PixelMaskingStrategy):
    """Imputes absent players' regions with a constant fill value (default: ``0.0``)."""

    def __init__(self, value: float = 0.0) -> None:
        """Initialize the ZeroMasking strategy.

        Args:
            value: Fill value for absent regions. Defaults to ``0.0``.
        """
        self.value = value

    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        """Apply constant-value masking to a batch of coalitions."""
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape

        masked_images = np.stack([image] * n_coalitions, axis=0)

        mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    mask[i] |= player_masks[j]

        masked_images[mask] = self.value
        return masked_images


class BlurMasking(PixelMaskingStrategy):
    """Imputes absent players' regions with a Gaussian-blurred version of the image.

    Absent regions are replaced with pixels drawn from a Gaussian-blurred copy
    of the original image.  This provides a smooth, natural-looking baseline
    that retains global color statistics while removing fine-grained spatial
    detail from hidden regions — useful when the model is sensitive to hard
    color discontinuities at region boundaries.

    Args:
        sigma: Standard deviation of the Gaussian kernel in pixels.
            Defaults to ``10.0``.

    Example::

        masking = BlurMasking(sigma=8)
        imputer = ImageImputer(arch, image, masking_strategy=masking)
    """

    def __init__(self, sigma: float = 10.0) -> None:
        """Initialize the BlurMasking strategy.

        Args:
            sigma: Standard deviation of the Gaussian kernel in pixels.
        """
        self.sigma = sigma

    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        """Apply Gaussian-blur masking to a batch of coalitions."""
        from scipy.ndimage import gaussian_filter

        blurred = gaussian_filter(image, sigma=[self.sigma, self.sigma, 0])
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape

        masked_images = np.stack([image] * n_coalitions, axis=0)

        absence_mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    absence_mask[i] |= player_masks[j]

        return np.where(absence_mask[..., np.newaxis], blurred[np.newaxis], masked_images)


class LatentMaskingStrategy(ABC):
    """Defines how tokens are masked in latent/embedding space."""

    @abstractmethod
    def predict_logits(
        self,
        model: Model,
        pixel_values: torch.Tensor,
        bool_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model with masked tokens and return logits.

        Args:
            model: The vision model.
            pixel_values: ``(1, C, H, W)`` pre-processed image tensor.
            bool_masks: ``(B, n_tokens)`` boolean mask; ``True`` = token masked (absent).

        Returns:
            Logit tensor of shape ``(B, n_classes)``.
        """
        ...


class BoolMaskedPosStrategy(LatentMaskingStrategy):
    """Masks tokens via the ``bool_masked_pos`` argument in the model forward pass."""

    def predict_logits(
        self,
        model: Model,
        pixel_values: torch.Tensor,
        bool_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model with ``bool_masked_pos`` masking and return logits."""
        import torch

        # ViTForImageClassification has mask_token=None by default, initialise it so the
        # embedding layer can replace masked patch tokens during the forward pass.
        embeddings = model.vit.embeddings
        if embeddings.mask_token is None:
            embeddings.mask_token = torch.nn.Parameter(torch.zeros(1, 1, model.config.hidden_size))
        batch = pixel_values.repeat(bool_masks.shape[0], 1, 1, 1)
        return model(pixel_values=batch, bool_masked_pos=bool_masks).logits


class MaskTokenStrategy(LatentMaskingStrategy):
    """Masks tokens by zeroing the ``mask_token`` embedding before the forward pass.

    Initialises ``model.vit.embeddings.mask_token`` to a zero vector so that
    masked patches contribute nothing to the CLS representation.  Works with
    HuggingFace ``ViTForImageClassification`` checkpoints where the mask token
    is not pre-initialised.
    """

    def predict_logits(
        self,
        model: Model,
        pixel_values: torch.Tensor,
        bool_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model with zero mask-token masking and return logits."""
        import torch

        model.vit.embeddings.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, model.config.hidden_size)
        )
        batch = pixel_values.repeat(bool_masks.shape[0], 1, 1, 1)
        return model(pixel_values=batch, bool_masked_pos=bool_masks).logits
