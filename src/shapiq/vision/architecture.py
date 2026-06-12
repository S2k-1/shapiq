from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .masking import (
    BoolMaskedPosStrategy,
    CNNMaskingStrategy,
    LayerMasking,
    ManifoldMaskingStrategy,
    MaskTokenStrategy,
    MeanColorMasking,
    TransformerMaskingStrategy,
)
from .players import (
    CNNPlayerStrategy,
    PatchStrategy,
    PlayerStrategy,
    SuperpixelStrategy,
    TransformerPlayerStrategy,
)
from .utils import get_torch_device, tensor_to_numpy, to_tensor_chw

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import torch


def _mask_images_hwc(
    image: np.ndarray,
    player_masks: np.ndarray,
    coalitions: np.ndarray,
    masking: CNNMaskingStrategy,
) -> list[np.ndarray]:
    """Apply a CNN masking strategy and return HWC images for HF processors."""
    import torch

    image_t = to_tensor_chw(image)
    masks_t = torch.from_numpy(player_masks)
    coalitions_t = torch.from_numpy(coalitions).bool()
    masked = masking.apply(image_t, masks_t, coalitions_t)
    hwc = tensor_to_numpy(masked.permute(0, 2, 3, 1))
    if image.dtype == np.uint8:
        hwc = (hwc * 255.0).clip(0, 255).astype(np.uint8)
    return [hwc[i] for i in range(hwc.shape[0])]


def _extract_hf_features(output: torch.Tensor | object) -> torch.Tensor:
    """Return a feature tensor from a HuggingFace model output or plain tensor."""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    return output.last_hidden_state[:, 0]


class ModelArchitectureStrategy(ABC):
    """Encapsulates model-specific inference logic, decoupling it from
    :class:`~shapiq.vision.imputer.ImageImputer`.

    Subclasses bind a player strategy and a masking strategy to a concrete
    model type and implement batched coalition evaluation via
    :meth:`value_function`.
    Input images are converted to tensors after player masks are generated.
    """

    @abstractmethod
    def default_player_strategy(self) -> PlayerStrategy: ...

    @abstractmethod
    def default_masking_strategy(self): ...

    @abstractmethod
    def prepare(self, image: np.ndarray) -> None:
        """Cache image-dependent state. Called once by ImageImputer before value_function."""
        ...

    @abstractmethod
    def value_function(self, coalitions: torch.BoolTensor) -> torch.Tensor:
        """Return model predictions for each coalition. Returns (n_coalitions,)."""
        ...

    @property
    def player_masks(self) -> torch.Tensor:
        """(n_players, H, W) boolean array of pixel masks for visualization."""


class CNNArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for CNN models (e.g. ResNet) using pixel-space masking.

    Players are defined in pixel space. Absent players are
    replaced by the masking strategy before the image batch is forwarded
    through the model.

    Args:
        model: A PyTorch CNN model (e.g. :class:`torchvision.models.ResNet`).
        masking_strategy: Pixel-space masking strategy. Defaults to
            :class:`~shapiq.vision.masking.MeanColorMasking`.
        player_strategy: Player definition strategy. Defaults to
            :class:`~shapiq.vision.players.SuperpixelStrategy` with 10
            segments.
    """

    def __init__(
        self,
        model,
        masking_strategy: CNNMaskingStrategy | None = None,
        player_strategy: CNNPlayerStrategy | None = None,
    ):
        self.model = model
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._player_masks: torch.Tensor | None = None
        self._image_tensor: torch.Tensor | None = None
        self._class_id: int | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        return MeanColorMasking()

    def prepare(self, image: np.array) -> None:
        """Cache the image tensor, player masks, and predicted class index.

        Runs one forward pass on the unmasked image to determine the class
        index that will be tracked across all coalition evaluations.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array.
        """
        import torch

        from .utils import get_torch_device, to_tensor_chw

        device = get_torch_device(self.model)
        self._image_tensor = to_tensor_chw(image, device=device)
        self._player_masks = torch.from_numpy(self._player_strategy.get_masks(image)).to(device)

        with torch.no_grad():
            logits = self.model(self._image_tensor.unsqueeze(0))
            self._class_id = int(logits.argmax(dim=1).item())

    def value_function(self, coalitions: torch.BoolTensor) -> torch.Tensor:
        """Evaluate the CNN for a batch of coalitions.

        Creates masked image tensors via the masking strategy in a single
        batched model call.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Float tensor of shape ``(n_coalitions,)`` with the logit for the
            predicted class for each coalition.
        """
        import torch

        with torch.no_grad():
            masked_batch = self._masking_strategy.apply(
                self._image_tensor, self._player_masks, coalitions
            )
            logits = self.model(masked_batch)
        return logits[:, self._class_id]

    @property
    def player_masks(self) -> torch.Tensor:
        return self._player_masks


class TransformerArchitecture(ModelArchitectureStrategy):
    """Architecture strategy for Vision Transformer models using latent-space masking.

    Players correspond to groups of patch tokens. Absent players are masked
    in token space via ``bool_masked_pos`` before the forward pass.

    Args:
        model: A vision transformer model.
        vit_processor: The matching processor used to preprocess
            the image into ``pixel_values``.
        masking_strategy: Token-space masking strategy. Defaults to
            :class:`~shapiq.vision.masking.MaskTokenStrategy`.
        player_strategy: Player definition strategy. Defaults to
            :class:`~shapiq.vision.players.PatchStrategy` with 9 players
            derived from the model's patch grid.
    """

    def __init__(
        self,
        model,
        vit_processor,
        masking_strategy: TransformerMaskingStrategy | None = None,
        player_strategy: TransformerPlayerStrategy | None = None,
    ):
        self.model = model
        self.processor = vit_processor
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._pixel_values: torch.Tensor | None = None
        self._player_masks: torch.Tensor | None = None
        self._token_masks: torch.Tensor | None = None
        self._class_id: int | None = None

    def default_player_strategy(self) -> PatchStrategy:
        grid_size = self.model.config.image_size // self.model.config.patch_size
        return PatchStrategy(grid_size=grid_size, n_players=9)

    def default_masking_strategy(self) -> MaskTokenStrategy:
        # ViTForImageClassification has mask_token=None by default; MaskTokenStrategy initialises it
        return MaskTokenStrategy(self.model)

    def prepare(self, image: np.ndarray) -> None:
        """Cache pixel values, token masks, pixel masks, and predicted class index.

        Passes ``image`` directly to the ViT processor (which expects
        a numpy ``(H, W, C)`` or PIL image), places the resulting
        ``pixel_values`` tensor on the model's device, and runs one forward
        pass to determine the predicted class index.

        Args:
            image: Input image as a ``(H, W, C)`` numpy array.
        """
        import torch

        from .utils import get_torch_device

        device = get_torch_device(self.model)
        inputs = self.processor(images=image, return_tensors="pt")
        self._pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            logits = self.model(pixel_values=self._pixel_values).logits
        self._class_id = int(logits.argmax(-1).item())

        self._player_masks = torch.from_numpy(self._player_strategy.get_pixel_masks(image)).to(
            device
        )
        self._token_masks = torch.from_numpy(self._player_strategy.get_token_masks()).to(device)

    def value_function(self, coalitions: torch.BoolTensor) -> torch.Tensor:
        """Evaluate the ViT for a batch of coalitions.

        Converts coalition membership to a ``bool_masked_pos`` tensor and
        runs a single batched forward pass.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Float tensor of shape ``(n_coalitions,)`` with the softmax
            probability for the predicted class for each coalition.
        """
        import torch

        with torch.no_grad():
            token_mask = self._masking_strategy.apply(coalitions, self._token_masks)
            batch = self._pixel_values.repeat(token_mask.shape[0], 1, 1, 1)
            logits = self.model(pixel_values=batch, bool_masked_pos=token_mask).logits
            probs = torch.softmax(logits, dim=-1)

        return probs[:, self._class_id]

    @property
    def player_masks(self) -> torch.Tensor:
        return self._player_masks


class LayerMaskedCNNArchitecture(ModelArchitectureStrategy):
    """CNN architecture that masks intermediate activations instead of input pixels.

    Pairs a raw ``torch.nn.Module`` with a
    :class:`~shapiq.vision.masking.ManifoldMaskingStrategy` such as
    :class:`~shapiq.vision.masking.LayerMasking`. A forward hook attenuates
    activations at the chosen layer so the model never receives a pixel-space
    replacement.

    Args:
        model: A raw ``torch.nn.Module``. Layer hooks need direct submodule access.
        preprocess: Callable mapping a ``(H, W, C)`` image array to a
            ``(C, H, W)`` tensor (e.g. ``ResNet18_Weights.DEFAULT.transforms()``).
        class_id: Class index whose softmax probability is returned.
        masking_strategy: Manifold masking strategy. Defaults to
            :class:`~shapiq.vision.masking.LayerMasking` hooking ``"layer2"``.
        player_strategy: Player definition strategy. Defaults to
            :class:`~shapiq.vision.players.SuperpixelStrategy` with 10 segments.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        preprocess: Callable[[np.ndarray], torch.Tensor],
        class_id: int,
        masking_strategy: ManifoldMaskingStrategy | None = None,
        player_strategy: CNNPlayerStrategy | None = None,
    ) -> None:
        """Initialize layer-masked CNN architecture strategy."""
        self.model = model
        self.preprocess = preprocess
        self.class_id = class_id
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._image_tensor: torch.Tensor | None = None
        self._player_masks: torch.Tensor | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        """Return the default superpixel player strategy."""
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> LayerMasking:
        """Return the default layer-masking strategy."""
        return LayerMasking(layer_name="layer2")

    def prepare(self, image: np.ndarray) -> None:
        """Cache the preprocessed image tensor and player masks."""
        import torch

        device = get_torch_device(self.model)
        image_tensor = self.preprocess(image)
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.as_tensor(image_tensor)
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)
        self._image_tensor = image_tensor.to(device)
        self._player_masks = torch.from_numpy(self._player_strategy.get_masks(image)).to(device)

    def value_function(self, coalitions: torch.BoolTensor) -> torch.Tensor:
        """Evaluate the model for a batch of coalitions via layer masking."""
        return self._masking_strategy.evaluate(
            self.model,
            self._image_tensor,
            self._player_masks,
            coalitions,
            self.class_id,
        )

    @property
    def player_masks(self) -> torch.Tensor:
        """Return cached spatial player masks."""
        return self._player_masks


class CustomViTArchitecture(ModelArchitectureStrategy):
    """ViT latent-space masking without a HuggingFace processor.

    Useful for custom ViT models where ``pixel_values`` and the token grid are
    supplied directly by the caller.

    Args:
        model: A ViT-like model that accepts ``pixel_values`` and
            ``bool_masked_pos``, and returns logits or an object with ``.logits``.
        pixel_values: Pre-processed ``(1, C, H, W)`` tensor or numpy array.
        class_id: The class index to score.
        n_tokens: Number of patch tokens (e.g. 196 for a 14x14 grid). Used to
            derive the default :class:`~shapiq.vision.players.PatchStrategy`.
        masking_strategy: Token-space masking strategy. Defaults to
            :class:`~shapiq.vision.masking.BoolMaskedPosStrategy`.
        player_strategy: Player definition strategy. Defaults to a
            :class:`~shapiq.vision.players.PatchStrategy` derived from
            ``n_tokens``.
    """

    def __init__(
        self,
        model: object,
        pixel_values: np.ndarray | torch.Tensor,
        class_id: int,
        n_tokens: int,
        masking_strategy: TransformerMaskingStrategy | None = None,
        player_strategy: TransformerPlayerStrategy | None = None,
    ) -> None:
        """Initialize custom ViT architecture strategy."""
        self.model = model
        self._pixel_values_input = pixel_values
        self.class_id = class_id
        self.n_tokens = n_tokens
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._pixel_values: torch.Tensor | None = None
        self._player_masks: torch.Tensor | None = None
        self._token_masks: torch.Tensor | None = None

    def default_player_strategy(self) -> PatchStrategy:
        """Return a patch player strategy derived from ``n_tokens``."""
        side = int(math.sqrt(self.n_tokens))
        if side * side != self.n_tokens:
            msg = (
                f"n_tokens ({self.n_tokens}) must be a perfect square for the "
                "default PatchStrategy. Provide an explicit player_strategy instead."
            )
            raise ValueError(msg)
        return PatchStrategy(grid_size=side, n_players=self.n_tokens)

    def default_masking_strategy(self) -> BoolMaskedPosStrategy:
        """Return the default bool-masked-pos masking strategy."""
        return BoolMaskedPosStrategy()

    def prepare(self, image: np.ndarray) -> None:
        """Cache pixel values, token masks, and pixel masks."""
        import torch

        device = get_torch_device(self.model)
        if isinstance(self._pixel_values_input, np.ndarray):
            pixel_values = torch.from_numpy(self._pixel_values_input)
        else:
            pixel_values = self._pixel_values_input
        self._pixel_values = pixel_values.to(device)
        self._player_masks = torch.from_numpy(self._player_strategy.get_pixel_masks(image)).to(
            device
        )
        self._token_masks = torch.from_numpy(self._player_strategy.get_token_masks()).to(device)

    def value_function(self, coalitions: torch.BoolTensor) -> torch.Tensor:
        """Evaluate the ViT for a batch of coalitions."""
        import torch

        with torch.no_grad():
            token_mask = self._masking_strategy.apply(coalitions, self._token_masks)
            batch = self._pixel_values.repeat(token_mask.shape[0], 1, 1, 1)
            logits = self.model(pixel_values=batch, bool_masked_pos=token_mask).logits
            probs = torch.softmax(logits, dim=-1)
        return probs[:, self.class_id]

    @property
    def player_masks(self) -> torch.Tensor:
        """Return cached spatial player masks."""
        return self._player_masks


class HuggingFacePixelArchitecture(ModelArchitectureStrategy):
    """Pixel-space masking for HuggingFace classification models.

    Works with any HuggingFace vision model whose forward pass accepts
    ``pixel_values`` and returns an object with a ``.logits`` attribute.

    Args:
        model: A HuggingFace model returning an output with ``.logits``.
        processor: A HuggingFace image processor (e.g. ``AutoImageProcessor``).
        class_id: Class index to score. Auto-detected from the unmasked image
            during :meth:`prepare` when ``None``.
        masking_strategy: Pixel-space masking strategy. Defaults to
            :class:`~shapiq.vision.masking.MeanColorMasking`.
        player_strategy: Player definition strategy. Defaults to
            :class:`~shapiq.vision.players.SuperpixelStrategy` with 10 segments.
    """

    def __init__(
        self,
        model,
        processor,
        class_id: int | None = None,
        masking_strategy: CNNMaskingStrategy | None = None,
        player_strategy: CNNPlayerStrategy | None = None,
    ):
        self.model = model
        self.processor = processor
        self._class_id = class_id
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._image: np.ndarray | None = None
        self._player_masks: torch.Tensor | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        return MeanColorMasking()

    def prepare(self, image: np.ndarray) -> None:
        """Cache the image, player masks, and predicted class index."""
        import torch

        self._image = image
        player_masks = self._player_strategy.get_masks(image)
        device = get_torch_device(self.model)
        self._player_masks = torch.from_numpy(player_masks).to(device)

        if self._class_id is None:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                logits = self.model(pixel_values=pixel_values).logits
            self._class_id = int(logits.argmax(-1).item())

    def value_function(self, coalitions: torch.BoolTensor) -> torch.Tensor:
        """Evaluate the model for a batch of coalitions."""
        import torch

        masked_images = _mask_images_hwc(
            self._image,
            tensor_to_numpy(self._player_masks),
            tensor_to_numpy(coalitions),
            self._masking_strategy,
        )
        device = get_torch_device(self.model)
        inputs = self.processor(images=masked_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            logits = self.model(pixel_values=pixel_values).logits
            probs = torch.softmax(logits, dim=-1)
        return probs[:, self._class_id]

    @property
    def player_masks(self) -> torch.Tensor:
        return self._player_masks


class ConvNeXtArchitecture(HuggingFacePixelArchitecture):
    """Architecture strategy for HuggingFace ConvNeXt classification models.

    Thin specialization of :class:`HuggingFacePixelArchitecture` for explicit
    model-type naming in user code and demos.
    """


class DINOv2Architecture(ModelArchitectureStrategy):
    """Architecture for DINOv2 (and similar SSL backbones) via feature similarity.

    The value function returns the cosine similarity between the embedding of
    each masked image and the embedding of the original unmasked image computed
    in :meth:`prepare`.

    Args:
        model: A backbone whose forward pass returns ``pooler_output``,
            ``last_hidden_state``, or a plain feature tensor.
        processor: A HuggingFace image processor.
        masking_strategy: Pixel-space masking strategy. Defaults to
            :class:`~shapiq.vision.masking.MeanColorMasking`.
        player_strategy: Player definition strategy. Defaults to
            :class:`~shapiq.vision.players.SuperpixelStrategy` with 10 segments.
    """

    def __init__(
        self,
        model,
        processor,
        masking_strategy: CNNMaskingStrategy | None = None,
        player_strategy: CNNPlayerStrategy | None = None,
    ):
        self.model = model
        self.processor = processor
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._image: np.ndarray | None = None
        self._player_masks: torch.Tensor | None = None
        self._reference_embedding: torch.Tensor | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        return MeanColorMasking()

    def _embed(self, images: list[np.ndarray]) -> torch.Tensor:
        import torch

        device = get_torch_device(self.model)
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            features = _extract_hf_features(self.model(pixel_values=pixel_values))
        return torch.nn.functional.normalize(features, dim=-1)

    def prepare(self, image: np.ndarray) -> None:
        """Cache player masks and the reference embedding for the image."""
        import torch

        self._image = image
        player_masks = self._player_strategy.get_masks(image)
        device = get_torch_device(self.model)
        self._player_masks = torch.from_numpy(player_masks).to(device)
        self._reference_embedding = self._embed([image])[0]

    def value_function(self, coalitions: torch.BoolTensor) -> torch.Tensor:
        """Return cosine similarity scores for each coalition."""
        masked_images = _mask_images_hwc(
            self._image,
            tensor_to_numpy(self._player_masks),
            tensor_to_numpy(coalitions),
            self._masking_strategy,
        )
        embeddings = self._embed(masked_images)
        return embeddings @ self._reference_embedding

    @property
    def player_masks(self) -> torch.Tensor:
        return self._player_masks


class CLIPArchitecture(ModelArchitectureStrategy):
    """Architecture for CLIP zero-shot image classification.

    The value function returns the softmax probability over ``text_prompts`` for
    the target prompt index, given each masked image.

    Args:
        model: A CLIP model with ``get_image_features``, ``get_text_features``,
            and a ``logit_scale`` parameter.
        processor: A HuggingFace CLIP processor.
        text_prompts: Text prompts to compare against.
        target_prompt_idx: Index of the prompt to score. Defaults to ``0``.
        masking_strategy: Pixel-space masking strategy. Defaults to
            :class:`~shapiq.vision.masking.MeanColorMasking`.
        player_strategy: Player definition strategy. Defaults to
            :class:`~shapiq.vision.players.SuperpixelStrategy` with 10 segments.
    """

    def __init__(
        self,
        model,
        processor,
        text_prompts: Sequence[str],
        target_prompt_idx: int = 0,
        masking_strategy: CNNMaskingStrategy | None = None,
        player_strategy: CNNPlayerStrategy | None = None,
    ):
        self.model = model
        self.processor = processor
        self.text_prompts = list(text_prompts)
        self.target_prompt_idx = target_prompt_idx
        self._masking_strategy = masking_strategy or self.default_masking_strategy()
        self._player_strategy = player_strategy or self.default_player_strategy()
        self._image: np.ndarray | None = None
        self._player_masks: torch.Tensor | None = None
        self._text_features: torch.Tensor | None = None

    def default_player_strategy(self) -> SuperpixelStrategy:
        return SuperpixelStrategy(n_segments=10)

    def default_masking_strategy(self) -> MeanColorMasking:
        return MeanColorMasking()

    def prepare(self, image: np.ndarray) -> None:
        """Cache player masks and normalized text features."""
        import torch

        self._image = image
        player_masks = self._player_strategy.get_masks(image)
        device = get_torch_device(self.model)
        self._player_masks = torch.from_numpy(player_masks).to(device)

        text_inputs = self.processor(text=self.text_prompts, return_tensors="pt", padding=True)
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        with torch.no_grad():
            text_features = _extract_hf_features(self.model.get_text_features(**text_inputs))
            self._text_features = torch.nn.functional.normalize(text_features, dim=-1)

    def value_function(self, coalitions: torch.BoolTensor) -> torch.Tensor:
        """Return CLIP similarity scores for each coalition."""
        import torch

        masked_images = _mask_images_hwc(
            self._image,
            tensor_to_numpy(self._player_masks),
            tensor_to_numpy(coalitions),
            self._masking_strategy,
        )
        device = get_torch_device(self.model)
        image_inputs = self.processor(images=masked_images, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device)
        with torch.no_grad():
            image_features = _extract_hf_features(
                self.model.get_image_features(pixel_values=pixel_values)
            )
            image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
            logits = self.model.logit_scale.exp() * image_features @ self._text_features.T
            probs = torch.softmax(logits, dim=-1)
        return probs[:, self.target_prompt_idx]

    @property
    def player_masks(self) -> torch.Tensor:
        return self._player_masks
