"""Architecture strategies for vision model inference.

Each strategy encapsulates a model type (CNN-like or ViT-like), its
default player and masking strategies and batched coalition evaluation.
Use :func:`~shapiq.vision.dispatch.resolve_architecture` to pick a strategy
automatically for a raw model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .custom_types import CoalitionDomain
from .dispatch import (
    _processed_dummy,
    extract_logits,
    masking_changes_output,
    resolve_patch_grid,
)
from .masking import MaskTokenStrategy, MeanColorMasking
from .players import PatchStrategy, SuperpixelStrategy
from .utils import get_torch_device, to_tensor_chw

try:
    import torch
except ImportError as err:
    from ._error import _vision_import_error

    raise _vision_import_error from err

if TYPE_CHECKING:
    from shapiq.typing import Model

    from .masking import (
        CNNMaskingStrategy,
        MaskingStrategy,
        TransformerMaskingStrategy,
    )
    from .players import (
        CNNPlayerStrategy,
        PlayerStrategy,
        TransformerPlayerStrategy,
    )


class ModelArchitectureStrategy(ABC):
    """Encapsulates model-specific inference logic.

    Subclasses bind a player strategy and a masking strategy to a concrete
    model type and implement batched coalition evaluation via
    :meth:`value_function`. Input images are converted to tensors after player masks are generated.
    """

    _model: Model
    _player_strategy: PlayerStrategy
    _masking_strategy: MaskingStrategy

    coalition_domain: CoalitionDomain
    """The coalition domain this architecture natively operates in."""

    def _validate_configuration(self) -> None:
        """Validate that model, player strategy, and masking strategy are compatible.

        Raises:
            TypeError: If the player and masking strategies live in different
                coalition domains, or if their (consistent) domain is not the
                one this architecture operates in.
        """
        type(self._player_strategy).validate_model(self._model)
        type(self._masking_strategy).validate_model(self._model)

        player_domain = self._player_strategy.coalition_domain
        masking_domain = self._masking_strategy.accepted_coalition_domain

        if player_domain is not masking_domain:
            msg = (
                "Player strategy and masking strategy are incompatible: "
                f"{type(self._player_strategy).__name__} uses coalition domain "
                f"{player_domain.value!r}, but "
                f"{type(self._masking_strategy).__name__} expects "
                f"{masking_domain.value!r}."
            )
            raise TypeError(msg)

        if player_domain is not self.coalition_domain:
            hint = (
                "Token-space strategies require TransformerArchitecture and a model "
                "that honors bool_masked_pos."
                if self.coalition_domain is CoalitionDomain.PIXEL
                else "Pixel-space masking is provided by CNNArchitecture(model=model, "
                "processor=processor, ...), which supports any classification model, "
                "including ViT and Swin."
            )
            msg = (
                f"{type(self).__name__} operates in coalition domain "
                f"{self.coalition_domain.value!r}, but "
                f"{type(self._player_strategy).__name__} and "
                f"{type(self._masking_strategy).__name__} use {player_domain.value!r}. "
                f"{hint}"
            )
            raise TypeError(msg)

    @abstractmethod
    def default_player_strategy(self) -> PlayerStrategy:
        """Return the default player strategy for this architecture."""
        ...

    @abstractmethod
    def default_masking_strategy(self) -> CNNMaskingStrategy | TransformerMaskingStrategy:
        """Return the default masking strategy for this architecture."""
        ...

    @abstractmethod
    def prepare(self, image: np.ndarray, class_index: int | None = None) -> None:
        """Cache image-dependent state. Called before value_function.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array.
            class_index: Index of the class to explain.
        """
        ...

    @abstractmethod
    def value_function(self, coalitions: torch.Tensor) -> torch.Tensor:
        """Return model predictions for each coalition.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Float tensor of shape ``(n_coalitions,)``.
        """
        ...

    @property
    @abstractmethod
    def player_masks(self) -> torch.Tensor:
        """Boolean pixel masks of shape ``(n_players, H, W)`` for visualization."""
        ...

    @property
    @abstractmethod
    def n_players(self) -> int:
        """Number of players defined by the player strategy."""
        ...

    @property
    @abstractmethod
    def model(self) -> Model:
        """Return the underlying model."""
        ...


class CNNArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for CNN-like models using pixel-space masking.

    Players are defined in pixel space. Absent players are replaced by the
    masking strategy before the image batch is forwarded through the model.

    This is also the fallback path for Hugging Face models that do not
    support token masking (e.g. Swin, BEiT, MobileViT, LeViT, CvT,
    SegFormer): pass the matching ``processor`` and each masked image is
    preprocessed with it before the forward pass, with logits read from the
    output object.
    """

    _masking_strategy: CNNMaskingStrategy
    _player_strategy: CNNPlayerStrategy

    coalition_domain = CoalitionDomain.PIXEL

    def __init__(
        self,
        model: Model,
        masking_strategy: CNNMaskingStrategy | None = None,
        player_strategy: CNNPlayerStrategy | None = None,
        processor: Model | None = None,
    ) -> None:
        """Initialize the CNN architecture strategy.

        Args:
            model: A model evaluated on image batches — a PyTorch CNN
                (e.g. :class:`torchvision.models.ResNet`) called directly on
                the masked tensor, or any Hugging Face image classification
                model when ``processor`` is given.
            masking_strategy: Pixel-space masking strategy. Defaults to
                :class:`~shapiq.vision.masking.MeanColorMasking`.
            player_strategy: Player definition strategy. Defaults to
                :class:`~shapiq.vision.players.SuperpixelStrategy` with 10
                segments.
            processor: Optional Hugging Face image processor. When given,
                masking happens on the original image and every masked image
                is preprocessed with the processor (resize, normalize) before
                being forwarded as ``pixel_values``.
        """
        self._model = model
        self._processor = processor
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._validate_configuration()
        self._player_masks: torch.Tensor
        self._image_tensor: torch.Tensor
        self._class_id: int | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        """Return a superpixel player strategy."""
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        """Return a mean-color masking strategy."""
        return MeanColorMasking()

    def prepare(self, image: np.ndarray, class_index: int | None = None) -> None:
        """Cache the image tensor, player masks, and predicted class index.

        Runs one forward pass on the unmasked image to determine the class
        index that will be tracked across all coalition evaluations.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array.
            class_index: Index of the class to explain.
        """
        device = get_torch_device(self._model)
        if self._processor is not None:
            # Keep the image in its natural 0-255 range so masked images can
            # round-trip through the processor as uint8 arrays.
            arr = image.astype(np.float32)
            if image.dtype != np.uint8 and arr.size > 0 and arr.max() <= 1.0:
                arr = arr * 255.0
            self._image_tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)
        else:
            self._image_tensor = to_tensor_chw(image, device=device)
        self._player_masks = torch.from_numpy(self._player_strategy.get_masks(image)).to(device)

        if class_index is not None:
            self._class_id = class_index
        elif self._class_id is None:
            with torch.no_grad():
                logits = self._forward(self._image_tensor.unsqueeze(0))
            self._class_id = int(logits.argmax(dim=1).item())

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward a ``(B, C, H, W)`` image batch and return ``(B, n_classes)`` logits.

        Without a processor the batch goes straight into the model. With a
        processor, each image is converted back to a uint8 ``(H, W, C)`` array
        and preprocessed before the forward pass.
        """
        if self._processor is None:
            return extract_logits(self._model(batch))

        arrays = batch.clamp(0.0, 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        inputs = self._processor(images=list(arrays), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(get_torch_device(self._model))
        return extract_logits(self._model(pixel_values=pixel_values))

    def value_function(self, coalitions: torch.Tensor) -> torch.Tensor:
        """Evaluate the model for a batch of coalitions.

        Creates masked image tensors via the masking strategy in a single
        batched model call.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Float tensor of shape ``(n_coalitions,)`` with the logit for the
            predicted class for each coalition.
        """
        with torch.no_grad():
            masked_batch = self._masking_strategy.apply(
                self._image_tensor,
                self._player_masks,
                coalitions.to(self._player_masks.device),
            )
            logits = self._forward(masked_batch)
        return logits[:, self._class_id]

    @property
    def player_masks(self) -> torch.Tensor:
        """Boolean pixel masks of shape ``(n_players, H, W)``."""
        return self._player_masks

    @property
    def n_players(self) -> int:
        """Number of players defined by the player strategy."""
        return self._player_strategy.n_players

    @property
    def model(self) -> Model:
        """Return the underlying model."""
        return self._model


class TransformerArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for ViT-like models using latent-space masking.

    Players correspond to groups of patch tokens. Absent players are masked
    in token space via ``bool_masked_pos`` before the forward pass.
    """

    _masking_strategy: TransformerMaskingStrategy
    _player_strategy: TransformerPlayerStrategy

    coalition_domain = CoalitionDomain.TOKEN

    def __init__(
        self,
        model: Model,
        processor: Model,
        masking_strategy: TransformerMaskingStrategy | None = None,
        player_strategy: TransformerPlayerStrategy | None = None,
        *,
        verified: bool = False,
    ) -> None:
        """Initialize the Transformer architecture strategy.

        Args:
            model: A vision transformer model whose output responds to
                ``bool_masked_pos``.
            processor: The matching image processor used to preprocess
                the image into ``pixel_values``.
            masking_strategy: Token-space masking strategy. Defaults to
                :class:`~shapiq.vision.masking.MaskTokenStrategy`.
            player_strategy: Player definition strategy. Defaults to
                :class:`~shapiq.vision.players.PatchStrategy` sized to the model's
                patch grid.
            verified: Set to ``True`` by callers that have already verified that
                the model honors ``bool_masked_pos`` (e.g.
                :func:`~shapiq.vision.dispatch.resolve_architecture` after its
                token-masking probe) to skip the verification forward passes.

        Raises:
            ValueError: If the model accepts ``bool_masked_pos`` but ignores it
                (e.g. Swin, BEiT, FocalNet, or CNN classification heads), which
                would yield constant, meaningless attributions.
        """
        self._model = model
        self.processor = processor
        # Resolve the player default first: on a non-ViT model its unresolvable
        # token grid explains the problem better than a missing mask_token slot.
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._validate_configuration()
        if not verified:
            self._verify_token_masking()
        self._pixel_values: torch.Tensor
        self._player_masks: torch.Tensor
        self._token_masks: torch.Tensor
        self._class_id: int | None = None

    def default_player_strategy(self) -> PatchStrategy:
        """Return a patch player strategy sized to the model's patch grid.

        The grid is resolved from the processor's output size and the model
        config (including nested configs such as ``config.vision_config``).

        Raises:
            TypeError: If the patch grid cannot be determined; pass a
                ``player_strategy`` explicitly in that case.
        """
        grid_size = resolve_patch_grid(self._model, self.processor)
        if grid_size is None:
            msg = (
                f"Could not determine the token grid of {type(self._model).__name__} from its "
                "config/processor. Pass player_strategy=PatchStrategy(grid_size=..., "
                "n_players=...) explicitly."
            )
            raise TypeError(msg)
        return PatchStrategy(
            grid_size=grid_size, n_players=PatchStrategy.default_n_players(grid_size)
        )

    def default_masking_strategy(self) -> MaskTokenStrategy:
        """Return a token-masking strategy.

        Note:
            Classification models usually have ``mask_token=None``;
            :class:`~shapiq.vision.masking.MaskTokenStrategy` initialises it.
        """
        return MaskTokenStrategy(self._model)

    def _verify_token_masking(self) -> None:
        """Verify on a dummy image that ``bool_masked_pos`` changes the model output.

        Compares an unmasked forward pass against a fully masked one (through
        the configured masking strategy). Many classification heads accept
        ``bool_masked_pos`` via ``**kwargs`` but silently drop it (e.g. Swin,
        BEiT, FocalNet, ResNet) — token-space masking would then produce
        constant, meaningless attributions.

        Raises:
            ValueError: If masking all tokens does not change the model
                output, i.e. the model ignores ``bool_masked_pos``.
        """
        device = get_torch_device(self._model)
        pixel_values = _processed_dummy(self.processor, device)
        token_masks = torch.from_numpy(self._player_strategy.get_token_masks()).to(device)
        empty_coalition = torch.zeros(1, self.n_players, dtype=torch.bool, device=device)
        token_mask = self._masking_strategy.apply(empty_coalition, token_masks)
        if not masking_changes_output(self._model, pixel_values, token_mask):
            msg = (
                f"{type(self._model).__name__} ignores bool_masked_pos: masking all tokens "
                "does not change its output, so token-space masking would produce constant "
                "attributions. Use pixel-space masking instead, e.g. "
                "CNNArchitecture(model=model, processor=processor) or "
                "resolve_architecture(model, processor)."
            )
            raise ValueError(msg)

    def prepare(self, image: np.ndarray, class_index: int | None = None) -> None:
        """Cache pixel values, token masks, pixel masks, and predicted class index.

        Passes ``image`` directly to the image processor (which expects
        a numpy ``(H, W, C)`` or PIL image), places the resulting
        ``pixel_values`` tensor on the model's device, and runs one forward
        pass to determine the predicted class index. That the model honors
        ``bool_masked_pos`` was already verified at construction.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array.
            class_index: Index of the class to explain.
        """
        device = get_torch_device(self._model)
        inputs = self.processor(images=image, return_tensors="pt")
        self._pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            logits = extract_logits(self._model(pixel_values=self._pixel_values))
        if class_index is not None:
            self._class_id = class_index
        elif self._class_id is None:
            self._class_id = int(logits.argmax(-1).item())

        self._player_masks = torch.from_numpy(self._player_strategy.get_pixel_masks(image)).to(
            device
        )
        self._token_masks = torch.from_numpy(self._player_strategy.get_token_masks()).to(device)

    def value_function(self, coalitions: torch.Tensor) -> torch.Tensor:
        """Evaluate the ViT for a batch of coalitions.

        Converts coalition membership to a ``bool_masked_pos`` tensor and
        runs a single batched forward pass.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Float tensor of shape ``(n_coalitions,)`` with the softmax
            probability for the predicted class for each coalition.
        """
        with torch.no_grad():
            token_mask = self._masking_strategy.apply(
                coalitions.to(self._token_masks.device), self._token_masks
            )
            batch = self._pixel_values.repeat(token_mask.shape[0], 1, 1, 1)
            logits = extract_logits(self._model(pixel_values=batch, bool_masked_pos=token_mask))
            probs = torch.softmax(logits, dim=-1)

        return probs[:, self._class_id]

    @property
    def player_masks(self) -> torch.Tensor:
        """Boolean pixel masks of shape ``(n_players, H, W)``."""
        return self._player_masks

    @property
    def n_players(self) -> int:
        """Number of players defined by the player strategy."""
        return self._player_strategy.n_players

    @property
    def model(self) -> Model:
        """Return the underlying model."""
        return self._model
