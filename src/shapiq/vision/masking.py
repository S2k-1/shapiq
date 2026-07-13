"""Masking strategies for vision models.

Defines how to replace absent players in masked images before forwarding
through the model. Masking is applied in pixel space for CNNs and token
space for ViTs. Requires PyTorch to be installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from shapiq.vision.validation import ModelCompatible

try:
    import torch
except ImportError as err:
    from ._error import _vision_import_error

    raise _vision_import_error from err

from .custom_types import CoalitionDomain, VisionModel, ViTLikeModel

if TYPE_CHECKING:
    from shapiq.typing import Model


class MaskingStrategy(ModelCompatible, ABC):
    """Base class for masking strategies with compatibility validation.

    Subclasses declare the coalition domain they accept via
    ``accepted_coalition_domain``. This is used to ensure the masking strategy
    matches the player strategy that produced the coalitions. Compatibility with
    a model protocol is enforced via ``compatible_model_protocol``, default is
    ``VisionModel``.
    """

    accepted_coalition_domain: CoalitionDomain
    compatible_model_protocol = VisionModel


class CNNMaskingStrategy(MaskingStrategy, ABC):
    """Base class for pixel-space masking strategies used with CNN models.

    Implementations receive the original image as a ``(C, H, W)`` tensor and
    a coalition matrix, and return a batch of masked images ready for a
    single forward pass through the model. ``accepted_coalition_domain`` is
    ``CoalitionDomain.PIXEL``.
    """

    accepted_coalition_domain: CoalitionDomain = CoalitionDomain.PIXEL
    compatible_model_protocol = VisionModel

    @abstractmethod
    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply masking to produce a batch of masked images.

        Args:
            image: Original image as a float32 ``(C, H, W)`` tensor.
            player_masks: Boolean tensor of shape ``(n_players, H, W)``
                mapping each player to its pixel region.
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``
                where ``True`` indicates a player is present (unmasked).

        Returns:
            Float32 tensor of shape ``(n_coalitions, C, H, W)`` with absent
            players replaced by the imputation value.
        """
        ...

    def _build_pixel_mask(
        self,
        player_masks: torch.Tensor,
        coalitions: torch.Tensor,
    ) -> torch.Tensor:
        """Build a combined pixel absence mask for all coalitions.

        Args:
            player_masks: Boolean tensor of shape ``(n_players, H, W)``.
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Boolean tensor of shape ``(n_coalitions, H, W)`` where ``True``
            means the pixel belongs to an absent player and should be imputed.
        """
        absent_players = ~coalitions  # (n_coalitions, n_players)

        n_players, H, W = player_masks.shape
        masks_flat = player_masks.view(n_players, -1).float()  # (n_players, H*W)

        # Union pixel masks of all absent players per coalition
        pixel_mask = (absent_players.float() @ masks_flat).bool()  # (n_coalitions, H*W)
        return pixel_mask.view(-1, H, W)  # (n_coalitions, H, W)


class MeanColorMasking(CNNMaskingStrategy):
    """Imputes absent player regions with the per-channel mean color of the original image.

    The mean is computed per channel across all spatial positions of the
    original image and broadcast into the masked regions.
    """

    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean color masking to absent player regions."""
        pixel_mask = self._build_pixel_mask(player_masks, coalitions)  # (n_coalitions, H, W)
        mean_color = image.mean(dim=(1, 2))  # (C,)

        return torch.where(
            pixel_mask.unsqueeze(1),  # (n_coalitions, 1, H, W)
            mean_color[None, :, None, None],  # (1, C, 1, 1)
            image.unsqueeze(0),  # (1, C, H, W)
        )


class ZeroMasking(CNNMaskingStrategy):
    """Imputes absent player regions with a constant scalar value.

    Args:
        value: The fill value used for masked pixels. Defaults to ``0.0``.
    """

    def __init__(self, value: float = 0.0) -> None:
        """Initialize the zero masking strategy with a specified fill value."""
        self.value = value

    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply zero (or constant) masking to absent player regions."""
        pixel_mask = self._build_pixel_mask(player_masks, coalitions)  # (n_coalitions, H, W)

        return torch.where(
            pixel_mask.unsqueeze(1),  # (n_coalitions, 1, H, W)
            torch.tensor(self.value, dtype=image.dtype, device=image.device),
            image.unsqueeze(0),  # (1, C, H, W)
        )


class TransformerMaskingStrategy(MaskingStrategy, ABC):
    """Base class for token-space masking strategies used with ViT models.

    Implementations convert a coalition matrix into a ``bool_masked_pos``
    tensor suitable for passing directly to a ViT forward call.
    ``accepted_coalition_domain`` is ``CoalitionDomain.TOKEN``.
    """

    accepted_coalition_domain: CoalitionDomain = CoalitionDomain.TOKEN
    compatible_model_protocol = ViTLikeModel

    @abstractmethod
    def apply(
        self,
        coalitions: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Convert coalitions to a token-level boolean mask.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``
                where ``True`` indicates a player is present.
            token_masks: Integer tensor of shape
                ``(n_players, tokens_per_player)`` mapping each player to its
                flat token indices.

        Returns:
            Boolean tensor of shape ``(n_coalitions, n_tokens)`` where
            ``True`` means the token is masked (player absent) and ``False``
            means the token is visible (player present).
        """
        ...

    def _to_token_mask(
        self,
        coalitions: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a coalition tensor to a flat token-level boolean mask.

        Tokens belonging to absent players are set to ``True`` (masked);
        tokens belonging to present players are set to ``False`` (visible).

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``
                where ``True`` indicates a player is present.
            token_masks: Integer tensor of shape
                ``(n_players, tokens_per_player)`` containing the flat token
                indices for each player.

        Returns:
            Boolean tensor of shape ``(n_coalitions, n_tokens)`` on the same
            device as ``coalitions``.
        """
        n_players = token_masks.shape[0]
        n_tokens = int(token_masks.max()) + 1

        # (n_players, n_tokens): one-hot encoding of which tokens belong to which player
        player_to_token = torch.zeros(
            (n_players, n_tokens), dtype=torch.bool, device=coalitions.device
        )
        player_to_token.scatter_(dim=1, index=token_masks, value=True)  # (n_players, n_tokens)

        # A token is visible (False) if at least one present player owns it
        visible = coalitions.float() @ player_to_token.float()  # (n_coalitions, n_tokens)
        return ~visible.bool()


class BoolMaskedPosStrategy(TransformerMaskingStrategy):
    """Masks tokens by passing ``bool_masked_pos`` directly to the model forward call.

    This strategy requires the model to support the ``bool_masked_pos``
    argument (e.g. :class:`~transformers.ViTForMaskedImageModeling`).
    """

    def apply(self, coalitions: torch.Tensor, token_masks: torch.Tensor) -> torch.Tensor:
        """Apply boolean masking by converting coalitions to a ``bool_masked_pos`` tensor."""
        return self._to_token_mask(coalitions, token_masks)


class MaskTokenStrategy(TransformerMaskingStrategy):
    """Masks tokens by zeroing the mask_token embedding before the forward pass."""

    def __init__(self, model: Model) -> None:
        """Initialise with the ViT-like model whose mask token will be zeroed.

        Args:
            model: A model whose backbone embeddings carry a ``mask_token``
                slot (e.g. ``model.vit.embeddings``, ``model.deit.embeddings``,
                ``model.beit.embeddings`` — resolved generically through
                ``model.base_model``).
        """
        self._model = model
        type(self).validate_model(model)

    @classmethod
    def validate_model(cls, model: Model) -> None:
        """Validate that ``model`` satisfies the declared protocol and exposes embeddings.

        Args:
            model: Object to validate against ``compatible_model_protocol``. Its
                backbone (``model.base_model`` for Hugging Face task heads, else
                the model itself) must expose an ``embeddings`` module with a
                ``mask_token`` slot.

        Raises:
            TypeError: If ``model`` has no backbone embeddings with a
                ``mask_token`` slot or is not compatible with the declared
                protocol.
        """
        from .dispatch import embeddings_of

        super().validate_model(model)
        embeddings = embeddings_of(model)
        if embeddings is None or not hasattr(embeddings, "mask_token"):
            msg = (
                f"{cls.__name__} requires a model whose backbone exposes "
                "``embeddings.mask_token`` (e.g. ViT, DeiT, BEiT, Swin). "
                f"Got {type(model).__name__}."
            )
            raise TypeError(msg)

    def apply(self, coalitions: torch.Tensor, token_masks: torch.Tensor) -> torch.Tensor:
        """Apply masking, ensuring the model carries an all-zero mask token."""
        from .dispatch import ensure_zero_mask_token

        if not ensure_zero_mask_token(self._model):
            msg = (
                f"Could not create a zero mask token for {type(self._model).__name__}: "
                "the embedding dimension could not be inferred. Pass a masking strategy "
                "explicitly (e.g. BoolMaskedPosStrategy) or use pixel-space masking."
            )
            raise TypeError(msg)
        return self._to_token_mask(coalitions, token_masks)
