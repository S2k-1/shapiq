"""Tests for ``shapiq.vision.architecture``.

The current package exposes two concrete architecture strategies:
:class:`CNNArchitecture` (pixel-space masking) and
:class:`TransformerArchitecture` (token-space masking).  Both cache
image-dependent state in :meth:`prepare` and evaluate coalitions in
:meth:`value_function`.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shapiq.vision.architecture import (
    CLIPArchitecture,
    CNNArchitecture,
    ConvNeXtArchitecture,
    CustomViTArchitecture,
    DINOv2Architecture,
    HuggingFacePixelArchitecture,
    LayerMaskedCNNArchitecture,
    ModelArchitectureStrategy,
    TransformerArchitecture,
)
from shapiq.vision.masking import (
    BoolMaskedPosStrategy,
    MaskTokenStrategy,
    MeanColorMasking,
    ZeroMasking,
)
from shapiq.vision.players import PatchStrategy, SuperpixelStrategy
from shapiq.vision.utils import to_tensor_chw

from .conftest import ChannelSumModel, FixedMasksStrategy, MockViT, MockViTProcessor


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class TestCNNArchitecture:
    def test_is_architecture_strategy(self) -> None:
        arch = CNNArchitecture(model=ChannelSumModel())
        assert isinstance(arch, ModelArchitectureStrategy)

    def test_default_player_strategy(self) -> None:
        arch = CNNArchitecture(model=ChannelSumModel())
        strategy = arch.default_player_strategy()
        assert isinstance(strategy, SuperpixelStrategy)
        assert strategy.n_segments == 10

    def test_default_masking_strategy(self) -> None:
        arch = CNNArchitecture(model=ChannelSumModel())
        assert isinstance(arch.default_masking_strategy(), MeanColorMasking)

    def test_explicit_masking_strategy_used(self) -> None:
        zero = ZeroMasking()
        arch = CNNArchitecture(model=ChannelSumModel(), masking_strategy=zero)
        assert arch._masking_strategy is zero

    def test_prepare_caches_player_masks(self, tiny_image, two_player_masks) -> None:
        arch = CNNArchitecture(
            model=ChannelSumModel(), player_strategy=FixedMasksStrategy(two_player_masks)
        )
        assert arch._player_masks is None
        arch.prepare(tiny_image)
        assert arch._player_masks is not None
        np.testing.assert_array_equal(_to_numpy(arch.player_masks), two_player_masks)

    def test_prepare_sets_class_id(self, tiny_image, two_player_masks) -> None:
        arch = CNNArchitecture(
            model=ChannelSumModel(), player_strategy=FixedMasksStrategy(two_player_masks)
        )
        arch.prepare(tiny_image)
        # ChannelSumModel class-0 logit (positive sum) wins.
        assert arch._class_id == 0

    def test_value_function_returns_value_per_coalition(self, tiny_image, two_player_masks) -> None:
        arch = CNNArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image)
        coalitions = torch.tensor(
            [
                [False, False],
                [True, False],
                [False, True],
                [True, True],
            ]
        )
        out = _to_numpy(arch.value_function(coalitions))
        assert out.shape == (4,)
        # Full coalition (no masking) equals the model's sum over the whole image.
        np.testing.assert_allclose(out[3], tiny_image.sum())
        # Empty coalition under ZeroMasking is exactly zero.
        assert out[0] == pytest.approx(0.0)

    def test_value_function_linear_decomposition(self, tiny_image, two_player_masks) -> None:
        arch = CNNArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image)
        coalitions = torch.tensor([[True, False], [False, True]])
        out = _to_numpy(arch.value_function(coalitions))
        np.testing.assert_allclose(out[0], tiny_image[:, :2].sum())
        np.testing.assert_allclose(out[1], tiny_image[:, 2:].sum())

    def test_default_superpixel_strategy_end_to_end(self) -> None:
        pytest.importorskip("skimage")
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(32, 32, 3)).astype(np.float64)
        arch = CNNArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=SuperpixelStrategy(n_segments=4),
        )
        arch.prepare(image)
        assert arch._player_masks is not None
        assert arch._player_masks.shape[0] == arch._player_strategy.n_players
        assert (arch._player_masks.cpu().numpy().sum(axis=0) == 1).all()
        out = _to_numpy(
            arch.value_function(torch.tensor([[True] * arch._player_strategy.n_players]))
        )
        assert np.isfinite(out).all()


class TestTransformerArchitecture:
    def test_is_architecture_strategy(self) -> None:
        arch = TransformerArchitecture(model=MockViT(), vit_processor=MockViTProcessor())
        assert isinstance(arch, ModelArchitectureStrategy)

    def test_default_player_strategy_uses_model_config(self) -> None:
        arch = TransformerArchitecture(model=MockViT(), vit_processor=MockViTProcessor())
        strategy = arch.default_player_strategy()
        assert isinstance(strategy, PatchStrategy)
        assert strategy.grid_size == 3  # 24 // 8
        assert strategy.n_players == 9

    def test_default_masking_strategy(self) -> None:
        arch = TransformerArchitecture(model=MockViT(), vit_processor=MockViTProcessor())
        assert isinstance(arch.default_masking_strategy(), MaskTokenStrategy)

    def test_init_fails_for_standard_vit_without_custom_players(self) -> None:
        """ViT-B/16 uses a 14x14 token grid; default 9 macro-patches do not divide 14."""
        model = SimpleNamespace(
            config=SimpleNamespace(image_size=224, patch_size=16, hidden_size=768)
        )
        with pytest.raises(ValueError, match="divisible"):
            TransformerArchitecture(model=model, vit_processor=object())

    def test_default_player_strategy_fails_for_standard_vit_grid(self) -> None:
        """``default_player_strategy()`` is invalid for ViT-B/16's 14x14 token grid."""
        model = SimpleNamespace(
            config=SimpleNamespace(image_size=224, patch_size=16, hidden_size=768)
        )
        arch = TransformerArchitecture(
            model=model,
            vit_processor=object(),
            player_strategy=PatchStrategy(grid_size=6, n_players=4),
        )
        with pytest.raises(ValueError, match="divisible"):
            arch.default_player_strategy()

    def test_prepare_sets_class_id_and_caches_state(self, image_24x24) -> None:
        arch = TransformerArchitecture(
            model=MockViT(),
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24)
        assert arch._class_id == 0  # logits [2.0, 0.5] -> argmax 0
        assert arch._pixel_values is not None
        assert arch._pixel_values.shape == (1, 3, 24, 24)
        assert arch._token_masks is not None

    def test_value_function_shape_and_monotonicity(self, image_24x24) -> None:
        arch = TransformerArchitecture(
            model=MockViT(),
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24)
        coalitions = torch.tensor(
            [
                [False] * 9,
                [True] + [False] * 8,
                [True] * 5 + [False] * 4,
                [True] * 9,
            ]
        )
        out = _to_numpy(arch.value_function(coalitions))
        assert out.shape == (4,)
        assert np.isfinite(out).all()
        # More visible tokens -> higher class-0 probability.
        assert out[0] < out[1] < out[2] < out[3]

    def test_value_function_empty_coalition_is_half(self, image_24x24) -> None:
        arch = TransformerArchitecture(
            model=MockViT(),
            vit_processor=MockViTProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24)
        out = _to_numpy(arch.value_function(torch.tensor([[False] * 9])))
        # No visible tokens -> logits (0, 0) -> softmax 0.5 for class 0.
        assert out[0] == pytest.approx(0.5, abs=1e-5)

    def test_value_function_with_mask_token_strategy(self, image_24x24) -> None:
        model = MockViT()
        arch = TransformerArchitecture(
            model=model,
            vit_processor=MockViTProcessor(),
            masking_strategy=MaskTokenStrategy(model),
        )
        arch.prepare(image_24x24)
        coalitions = torch.tensor(
            [
                [False] * 9,
                [True] + [False] * 8,
                [True] * 5 + [False] * 4,
                [True] * 9,
            ]
        )
        out = _to_numpy(arch.value_function(coalitions))
        assert out.shape == (4,)
        assert out[0] < out[1] < out[2] < out[3]
        assert torch.allclose(model.vit.embeddings.mask_token.data, torch.zeros(1, 1, 4))


class TestArchitectureOutputScale:
    def test_cnn_returns_logits_transformer_returns_probabilities(
        self,
        tiny_image,
        two_player_masks,
        transformer_architecture,
        image_24x24,
    ) -> None:
        cnn = CNNArchitecture(
            model=ChannelSumModel(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        cnn.prepare(tiny_image)
        cnn_full = float(_to_numpy(cnn.value_function(torch.tensor([[True, True]])))[0])
        cnn_empty = float(_to_numpy(cnn.value_function(torch.tensor([[False, False]])))[0])
        # Channel-sum logits are unbounded and not confined to [0, 1].
        assert cnn_full > 1.0
        assert cnn_empty == pytest.approx(0.0)

        transformer_architecture.prepare(image_24x24)
        vit_full = float(
            _to_numpy(transformer_architecture.value_function(torch.tensor([[True] * 9])))[0]
        )
        vit_empty = float(
            _to_numpy(transformer_architecture.value_function(torch.tensor([[False] * 9])))[0]
        )
        assert 0.0 <= vit_full <= 1.0
        assert 0.0 <= vit_empty <= 1.0


class MockHFClassifier:
    """HF-style classifier whose class-0 logit equals the pixel sum."""

    def __call__(self, pixel_values=None, **_):
        total = pixel_values.double().sum(dim=(1, 2, 3))
        return SimpleNamespace(logits=torch.stack([total, -total], dim=1))


class MockHFProcessor:
    """Mimics a HF image processor for single images or batches."""

    def __call__(self, images=None, return_tensors="pt", **_):
        if isinstance(images, list):
            tensors = [
                torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1).copy())
                for img in images
            ]
            return {"pixel_values": torch.stack(tensors)}
        arr = np.asarray(images, dtype=np.float32)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0)
        return {"pixel_values": tensor}


class MockDINOv2:
    """Backbone returning one feature per image proportional to the pixel sum."""

    def __call__(self, pixel_values=None, **_):
        total = pixel_values.double().sum(dim=(1, 2, 3))
        return SimpleNamespace(last_hidden_state=total[:, None, None])


class MockCLIP:
    """Minimal CLIP mock with fixed text features and image-sum logits."""

    def __init__(self) -> None:
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

    def get_text_features(self, **_):
        return torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def get_image_features(self, pixel_values=None, **_):
        total = pixel_values.double().sum(dim=(1, 2, 3))
        return torch.stack([total, -total], dim=1)


class MockCLIPProcessor(MockHFProcessor):
    """CLIP processor that also accepts text inputs."""

    def __call__(self, images=None, text=None, return_tensors="pt", *, padding=True, **_):
        if text is not None:
            return {"input_ids": torch.zeros((len(text), 4), dtype=torch.long)}
        return super().__call__(images=images, return_tensors=return_tensors)


class TestHuggingFacePixelArchitecture:
    def test_prepare_auto_detects_class_id(self, tiny_image, two_player_masks) -> None:
        arch = HuggingFacePixelArchitecture(
            model=MockHFClassifier(),
            processor=MockHFProcessor(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image)
        assert arch._class_id == 0

    def test_value_function_returns_probabilities(self, tiny_image, two_player_masks) -> None:
        arch = HuggingFacePixelArchitecture(
            model=MockHFClassifier(),
            processor=MockHFProcessor(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image)
        coalitions = torch.tensor([[False, False], [True, True]])
        out = _to_numpy(arch.value_function(coalitions))
        assert out.shape == (2,)
        assert 0.0 <= out[0] <= 1.0
        assert 0.0 <= out[1] <= 1.0
        assert out[1] > out[0]


class TestConvNeXtArchitecture:
    def test_is_huggingface_pixel_subclass(self) -> None:
        arch = ConvNeXtArchitecture(model=MockHFClassifier(), processor=MockHFProcessor())
        assert isinstance(arch, HuggingFacePixelArchitecture)


class TestDINOv2Architecture:
    def test_full_coalition_similarity_is_one(self, tiny_image, two_player_masks) -> None:
        arch = DINOv2Architecture(
            model=MockDINOv2(),
            processor=MockHFProcessor(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image)
        out = _to_numpy(arch.value_function(torch.tensor([[True, True]])))
        assert out[0] == pytest.approx(1.0, abs=1e-5)

    def test_empty_coalition_similarity_is_lower(self, tiny_image, two_player_masks) -> None:
        arch = DINOv2Architecture(
            model=MockDINOv2(),
            processor=MockHFProcessor(),
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image)
        full = float(_to_numpy(arch.value_function(torch.tensor([[True, True]])))[0])
        empty = float(_to_numpy(arch.value_function(torch.tensor([[False, False]])))[0])
        assert empty < full


class TestCLIPArchitecture:
    def test_value_function_returns_target_prompt_probability(
        self, tiny_image, two_player_masks
    ) -> None:
        arch = CLIPArchitecture(
            model=MockCLIP(),
            processor=MockCLIPProcessor(),
            text_prompts=["a dog", "a cat"],
            target_prompt_idx=0,
            masking_strategy=ZeroMasking(),
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image)
        coalitions = torch.tensor([[False, False], [True, True]])
        out = _to_numpy(arch.value_function(coalitions))
        assert out.shape == (2,)
        assert 0.0 <= out[0] <= 1.0
        assert 0.0 <= out[1] <= 1.0
        assert out[1] > out[0]


class _HookableCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 3, padding=1)
        self.layer2 = torch.nn.Conv2d(2, 2, 3, padding=1)
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.layer2(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


class TestLayerMaskedCNNArchitecture:
    def test_value_function_returns_probabilities(self, tiny_image, two_player_masks) -> None:
        arch = LayerMaskedCNNArchitecture(
            model=_HookableCNN().eval(),
            preprocess=to_tensor_chw,
            class_id=0,
            player_strategy=FixedMasksStrategy(two_player_masks),
        )
        arch.prepare(tiny_image)
        coalitions = torch.tensor([[True, False], [False, True], [True, True]])
        out = _to_numpy(arch.value_function(coalitions))
        assert out.shape == (3,)
        assert np.isfinite(out).all()
        assert (out >= 0.0).all() and (out <= 1.0).all()
        assert out[0] != out[1]


class TestCustomViTArchitecture:
    def test_value_function_shape_and_monotonicity(self, image_24x24) -> None:
        pixel_values = MockViTProcessor()(images=image_24x24, return_tensors="pt")["pixel_values"]
        arch = CustomViTArchitecture(
            model=MockViT(),
            pixel_values=pixel_values,
            class_id=0,
            n_tokens=9,
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24)
        coalitions = torch.tensor(
            [
                [False] * 9,
                [True] + [False] * 8,
                [True] * 9,
            ]
        )
        out = _to_numpy(arch.value_function(coalitions))
        assert out.shape == (3,)
        assert out[0] < out[1] < out[2]

    def test_accepts_numpy_pixel_values(self, image_24x24) -> None:
        pixel_values = MockViTProcessor()(images=image_24x24, return_tensors="pt")[
            "pixel_values"
        ].numpy()
        arch = CustomViTArchitecture(
            model=MockViT(),
            pixel_values=pixel_values,
            class_id=0,
            n_tokens=9,
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24)
        out = _to_numpy(arch.value_function(torch.tensor([[True] * 9])))
        assert out.shape == (1,)
        assert 0.0 <= out[0] <= 1.0
