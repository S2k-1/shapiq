from abc import ABC, abstractmethod

import numpy as np
import torch

from .masking import LatentMaskingStrategy, MaskTokenStrategy, MeanColorMasking, PixelMaskingStrategy
from .players import LatentPlayerStrategy, PatchStrategy, PixelPlayerStrategy, PlayerStrategy, SuperpixelStrategy


class ModelArchitectureStrategy(ABC):
    """Encapsulates model-specific inference logic, decoupling it from ImageImputer."""

    @abstractmethod
    def default_player_strategy(self) -> PlayerStrategy: ...

    @abstractmethod
    def default_masking_strategy(self): ...

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
        return np.asarray(self.model(masked)).squeeze()

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
        bool_masks = torch.stack(
            [self._player_strategy_ref.get_latent_mask(c) for c in coalitions]
        )
        with torch.no_grad():
            logits = self._masking_strategy.predict_logits(self.model, self._pixel_values, bool_masks)
            probs = torch.softmax(logits, dim=-1)
        return probs[:, self._class_id].cpu().numpy()

    def calc_empty_prediction(self, image: np.ndarray) -> float:
        n = self._player_strategy_ref.n_players
        return float(self.value_function(image, np.zeros((1, n), dtype=bool))[0])
