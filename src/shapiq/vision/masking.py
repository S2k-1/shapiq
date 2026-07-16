"""Masking strategies for vision models.

Defines how to replace absent players in masked images before forwarding
through the model. Masking is applied in pixel space for CNNs and token
space for ViTs. Requires PyTorch to be installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from shapiq.vision.validation import ModelCompatible, validate_config_attributes

try:
    import torch
except ImportError as err:
    from ._error import _vision_import_error

    raise _vision_import_error from err

from .custom_types import CoalitionDomain, VisionModel

if TYPE_CHECKING:
    from shapiq.typing import Model


class MaskingStrategy(ModelCompatible, ABC):
    """Base class for masking strategies with compatibility validation.

    Subclasses declare the coalition domain they accept via
    ``accepted_coalition_domain``. This is used to ensure the masking strategy
    matches the player strategy that produced the coalitions. Compatibility with
    a model protocol is enforced via ``compatible_model_protocol``, default is
    ``VisionModel``. Some strategies require additional model attributes, which
    are validated in ``validate_model``.
    """

    accepted_coalition_domain: CoalitionDomain
    compatible_model_protocol = VisionModel


class PixelBasedMaskingStrategy(MaskingStrategy, ABC):
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
        n_players, H, W = player_masks.shape
        masks_flat = player_masks.view(n_players, -1).float()

        absent_players = (~coalitions).to(masks_flat.device).float()  

        pixel_mask = (absent_players @ masks_flat).bool()
        return pixel_mask.view(-1, H, W)  


class MeanColorMasking(PixelBasedMaskingStrategy):
    """Imputes absent player regions with the per-channel mean color of the original image.

    The mean is computed per channel across all spatial positions of the
    original image and broadcast into the masked regions.
    """

    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean color masking to absent player regions."""
        pixel_mask = self._build_pixel_mask(player_masks, coalitions)
        mean_color = image.mean(dim=(1, 2))  

        return torch.where(
            pixel_mask.unsqueeze(1),  
            mean_color[None, :, None, None], 
            image.unsqueeze(0),  
        )


class ZeroMasking(PixelBasedMaskingStrategy):
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
        pixel_mask = self._build_pixel_mask(player_masks, coalitions)

        return torch.where(
            pixel_mask.unsqueeze(1),
            torch.tensor(self.value, dtype=image.dtype, device=image.device),
            image.unsqueeze(0), 
        )


class LatentBasedMaskingStrategy(MaskingStrategy, ABC):
    """Base class for token-space masking strategies used with ViT models.

    Implementations convert a coalition matrix into a ``bool_masked_pos``
    tensor suitable for passing directly to a ViT forward call.
    ``accepted_coalition_domain`` is ``CoalitionDomain.TOKEN``.
    """

    accepted_coalition_domain: CoalitionDomain = CoalitionDomain.TOKEN

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

        player_to_token = torch.zeros(
            (n_players, n_tokens), dtype=torch.bool, device=coalitions.device
        )
        player_to_token.scatter_(dim=1, index=token_masks, value=True)

        visible = coalitions.float() @ player_to_token.float() 
        return ~visible.bool()


class BoolMaskedPosStrategy(LatentBasedMaskingStrategy):
    """Masks tokens by passing ``bool_masked_pos`` directly to the model forward call.

    This strategy requires the model to support the ``bool_masked_pos``
    argument (e.g. :class:`~transformers.ViTForMaskedImageModeling`).

    Unlike :class:`MaskTokenStrategy`, it does not initialize the model's
    ``mask_token``: the model must already own one, which Hugging Face only
    provides when the model was built with ``use_mask_token=True``.
    """

    @classmethod
    def validate_model(cls, model: Model) -> None:
        """Validate that ``model`` satisfies the declared protocol and owns a mask token.

        Args:
            model: Object to validate against ``compatible_model_protocol`` and
                supports model.vit.embeddings.mask_token.

        Raises:
            TypeError: If ``model`` is not compatible with the declared protocol,
                does not expose ``vit.embeddings.mask_token``, or leaves it unset.
        """
        super().validate_model(model)
        try:
            mask_token = model.vit.embeddings.mask_token
        except AttributeError:
            msg = f"{cls.__name__} requires a model exposing ``vit.embeddings.mask_token``."
            raise TypeError(msg) from None
        if mask_token is None:
            msg = (
                f"{cls.__name__} requires a model built with ``use_mask_token=True`` (e.g. "
                "``ViTForMaskedImageModeling``), but ``vit.embeddings.mask_token`` is None. "
                "Use MaskTokenStrategy(model) instead, which initialises the mask token itself."
            )
            raise TypeError(msg)

    def apply(self, coalitions: torch.Tensor, token_masks: torch.Tensor) -> torch.Tensor:
        """Apply boolean masking by converting coalitions to a ``bool_masked_pos`` tensor."""
        return self._to_token_mask(coalitions, token_masks)


class MaskTokenStrategy(LatentBasedMaskingStrategy):
    """Masks tokens by zeroing the mask_token embedding before the forward pass."""

    def __init__(self, model: Model) -> None:
        """Initialise with the ViT model whose mask token will be zeroed.

        Args:
            model: A ViT model with a ``vit.embeddings.mask_token``
                parameter.
        """
        self._model = model
        type(self).validate_model(model)

    @classmethod
    def validate_model(cls, model: Model) -> None:
        """Validate that ``model`` satisfies the declared protocol and exposes required attributes.

        Args:
            model: Object to validate against ``compatible_model_protocol`` and supports
                model.vit.embeddings.mask_token and model.config.hidden_size.

        Raises:
            TypeError: If ``model`` does not support the required attributes or is not
                compatible with the declared protocol.
        """
        super().validate_model(model)
        try:
            embeddings = model.vit.embeddings
            _ = embeddings.mask_token
        except AttributeError:
            msg = f"{cls.__name__} requires a model exposing ``vit.embeddings.mask_token``."
            raise TypeError(msg) from None
        validate_config_attributes(model, ("hidden_size",), cls.__name__)

    def apply(self, coalitions: torch.Tensor, token_masks: torch.Tensor) -> torch.Tensor:
        """Apply masking by setting the model's mask_token embedding to zero."""
        self._model.vit.embeddings.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, self._model.config.hidden_size)
        )
        return self._to_token_mask(coalitions, token_masks)
