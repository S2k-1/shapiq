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
    CNNArchitecture,
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

from .conftest import ChannelSumModel, FixedMasksStrategy


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


class _MockViT:
    """HF-style ViT mock used by :class:`TransformerArchitecture` tests.

    ``config.image_size=24`` and ``config.patch_size=8`` produce a grid_size of
    3, which is compatible with the default 9-player patch grid.  Without
    ``bool_masked_pos`` (the class-detection forward in ``prepare``) the model
    favours class 0; with ``bool_masked_pos`` the class-0 logit equals the
    number of visible tokens, giving deterministic monotonicity.
    """

    class _Config:
        image_size = 24
        patch_size = 8
        hidden_size = 4

    config = _Config()

    def __init__(self) -> None:
        self.vit = SimpleNamespace(
            embeddings=SimpleNamespace(mask_token=torch.nn.Parameter(torch.zeros(1, 1, 4)))
        )

    def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
        b = pixel_values.shape[0]
        if bool_masked_pos is None:
            return SimpleNamespace(logits=torch.tensor([[2.0, 0.5]]).expand(b, -1).clone())
        visible = (~bool_masked_pos).sum(dim=1).float()
        return SimpleNamespace(logits=torch.stack([visible, -visible], dim=1))


class _MockProcessor:
    """Mimics a HF image processor turning an HWC image into (1, C, H, W)."""

    def __call__(self, images=None, return_tensors="pt"):
        arr = np.asarray(images, dtype=np.float32)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy()).unsqueeze(0)
        return {"pixel_values": tensor}


@pytest.fixture
def image_24x24() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(24, 24, 3)).astype(np.float64)


class TestTransformerArchitecture:
    def test_is_architecture_strategy(self) -> None:
        arch = TransformerArchitecture(model=_MockViT(), vit_processor=_MockProcessor())
        assert isinstance(arch, ModelArchitectureStrategy)

    def test_default_player_strategy_uses_model_config(self) -> None:
        arch = TransformerArchitecture(model=_MockViT(), vit_processor=_MockProcessor())
        strategy = arch.default_player_strategy()
        assert isinstance(strategy, PatchStrategy)
        assert strategy.grid_size == 3  # 24 // 8
        assert strategy.n_players == 9

    def test_default_masking_strategy(self) -> None:
        arch = TransformerArchitecture(model=_MockViT(), vit_processor=_MockProcessor())
        assert isinstance(arch.default_masking_strategy(), MaskTokenStrategy)

    def test_prepare_sets_class_id_and_caches_state(self, image_24x24) -> None:
        arch = TransformerArchitecture(
            model=_MockViT(),
            vit_processor=_MockProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24)
        assert arch._class_id == 0  # logits [2.0, 0.5] -> argmax 0
        assert arch._pixel_values is not None
        assert arch._pixel_values.shape == (1, 3, 24, 24)
        assert arch._token_masks is not None

    def test_value_function_shape_and_monotonicity(self, image_24x24) -> None:
        arch = TransformerArchitecture(
            model=_MockViT(),
            vit_processor=_MockProcessor(),
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
            model=_MockViT(),
            vit_processor=_MockProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        arch.prepare(image_24x24)
        out = _to_numpy(arch.value_function(torch.tensor([[False] * 9])))
        # No visible tokens -> logits (0, 0) -> softmax 0.5 for class 0.
        assert out[0] == pytest.approx(0.5, abs=1e-5)
