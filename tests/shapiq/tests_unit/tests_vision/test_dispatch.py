"""Tests for the automatic architecture dispatch in ``shapiq.vision.dispatch``.

The dispatcher must classify models by *behavior*, not by signature: recent
``transformers`` classification heads accept ``bool_masked_pos`` via
``**kwargs`` and some families (BEiT, Swin, FocalNet) silently drop it. The
mocks below reproduce both behaviors without requiring ``transformers``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

from shapiq.vision.architecture import (
    CNNArchitecture,
    TransformerArchitecture,
)
from shapiq.vision.dispatch import (
    resolve_architecture,
    resolve_patch_grid,
)
from shapiq.vision.probing import (
    ensure_zero_mask_token,
    probe_token_masking,
    resolve_embed_dim,
)
from shapiq.vision.utils import extract_logits

from .conftest import ChannelSumModel


class FakeProcessor:
    """HF-style processor mock: HWC uint8 image(s) -> (B, C, H, W) float pixel_values."""

    size: ClassVar[dict[str, int]] = {"height": 32}

    def __call__(self, images, return_tensors="pt"):
        if not isinstance(images, list):
            images = [images]
        batch = np.stack(
            [np.asarray(image, dtype=np.float32).transpose(2, 0, 1) for image in images]
        )
        return {"pixel_values": torch.from_numpy(batch / 255.0)}


class FakeMaskableViT:
    """ViT-like mock whose logits depend on how many tokens stay visible."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(image_size=32, patch_size=8, hidden_size=6)
        self.embeddings = SimpleNamespace(mask_token=None)

    def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
        batch_size = pixel_values.shape[0]
        if bool_masked_pos is None:
            visible = torch.full((batch_size,), 16.0)
        else:
            visible = (~bool_masked_pos).sum(dim=1).float()
        return SimpleNamespace(logits=torch.stack([visible, -visible], dim=1))


class FakeSwallowingModel:
    """HF-style mock that accepts but ignores ``bool_masked_pos`` (like BEiT/Swin heads)."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(image_size=32, patch_size=8, hidden_size=6)
        self.embeddings = SimpleNamespace(mask_token=None)

    def __call__(self, pixel_values=None, **_):
        total = pixel_values.sum(dim=(1, 2, 3))
        return SimpleNamespace(logits=torch.stack([total, -total], dim=1))


class FakeEncoderOnlyModel:
    """HF-style mock without classification logits (like ViT-MAE)."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(image_size=32, patch_size=8)

    def __call__(self, pixel_values=None, **_):
        return SimpleNamespace(last_hidden_state=torch.zeros(pixel_values.shape[0], 16, 6))


class TestResolveArchitecture:
    def test_architecture_passthrough(self) -> None:
        arch = CNNArchitecture(model=ChannelSumModel())
        assert resolve_architecture(arch) is arch

    def test_plain_module_dispatches_to_cnn(self) -> None:
        arch = resolve_architecture(ChannelSumModel())
        assert isinstance(arch, CNNArchitecture)
        assert arch._processor is None

    def test_token_maskable_model_dispatches_to_transformer(self) -> None:
        processor = FakeProcessor()
        arch = resolve_architecture(FakeMaskableViT(), processor=processor)
        assert isinstance(arch, TransformerArchitecture)
        assert arch.processor is processor

    def test_swallowing_model_falls_back_to_pixel_masking(self) -> None:
        """A model that ignores bool_masked_pos must NOT get token masking."""
        processor = FakeProcessor()
        arch = resolve_architecture(FakeSwallowingModel(), processor=processor)
        assert isinstance(arch, CNNArchitecture)
        assert arch._processor is processor

    def test_encoder_only_model_raises_clear_error(self) -> None:
        with pytest.raises(TypeError, match="logits"):
            resolve_architecture(FakeEncoderOnlyModel(), processor=FakeProcessor())

    def test_hf_model_without_processor_raises(self) -> None:
        """A config-carrying model with no loadable processor needs an explicit one."""
        model = FakeMaskableViT()  # has .config but no name_or_path to load from
        with pytest.raises(TypeError, match="processor"):
            resolve_architecture(model)


class TestResolvePatchGrid:
    def test_processor_size_preferred_over_config(self) -> None:
        """DINOv2-style mismatch: the processor output size wins over config.image_size."""
        model = SimpleNamespace(config=SimpleNamespace(image_size=518, patch_size=8))
        assert resolve_patch_grid(model, FakeProcessor()) == 4  # 32 // 8

    def test_nested_vision_config(self) -> None:
        """CLIP-style nesting: patch_size lives under config.vision_config."""
        config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=16, image_size=224))
        model = SimpleNamespace(config=config)
        assert resolve_patch_grid(model) == 14

    def test_missing_patch_size_returns_none(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(image_size=224))
        assert resolve_patch_grid(model, FakeProcessor()) is None

    def test_crop_size_preferred_over_size(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(patch_size=16))
        processor = SimpleNamespace(size={"shortest_edge": 256}, crop_size={"height": 224})
        assert resolve_patch_grid(model, processor) == 14

    def test_tuple_valued_entries_resolve(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(image_size=(64, 64), patch_size=(8, 8)))
        assert resolve_patch_grid(model) == 8


class TestMaskTokenHelpers:
    def test_ensure_zero_mask_token_creates_from_config_dim(self) -> None:
        model = FakeMaskableViT()
        assert ensure_zero_mask_token(model)
        token = model.embeddings.mask_token
        assert isinstance(token, torch.nn.Parameter)
        assert token.shape == (1, 1, 6)
        assert (token == 0).all()

    def test_ensure_zero_mask_token_zeroes_existing_in_place(self) -> None:
        embeddings = SimpleNamespace(mask_token=torch.nn.Parameter(torch.ones(1, 1, 4)))
        model = SimpleNamespace(base_model=SimpleNamespace(embeddings=embeddings), config=None)
        original = embeddings.mask_token
        assert ensure_zero_mask_token(model)
        assert embeddings.mask_token is original  # zeroed, not replaced
        assert (embeddings.mask_token == 0).all()

    def test_ensure_zero_mask_token_false_without_slot(self) -> None:
        assert not ensure_zero_mask_token(SimpleNamespace(embeddings=SimpleNamespace()))

    def test_resolve_embed_dim_from_projection(self) -> None:
        embeddings = SimpleNamespace(
            mask_token=None,
            patch_embeddings=SimpleNamespace(projection=SimpleNamespace(out_channels=96)),
        )
        model = SimpleNamespace(embeddings=embeddings)
        assert resolve_embed_dim(model) == 96


class TestProbeTokenMasking:
    def test_probe_true_for_maskable_model(self) -> None:
        assert probe_token_masking(FakeMaskableViT(), FakeProcessor(), grid_size=4)

    def test_probe_false_for_swallowing_model(self) -> None:
        assert not probe_token_masking(FakeSwallowingModel(), FakeProcessor(), grid_size=4)

    def test_probe_false_on_processor_error(self) -> None:
        assert not probe_token_masking(FakeMaskableViT(), object(), grid_size=4)


class TestExtractLogits:
    def test_tensor_passthrough(self) -> None:
        logits = torch.ones(2, 3)
        assert extract_logits(logits) is logits

    def test_logits_attribute(self) -> None:
        logits = torch.ones(2, 3)
        assert extract_logits(SimpleNamespace(logits=logits)) is logits

    def test_raises_without_logits(self) -> None:
        with pytest.raises(TypeError, match="logits"):
            extract_logits(SimpleNamespace(last_hidden_state=torch.ones(1, 4)))


class TestPixelFallbackEndToEnd:
    def test_cnn_architecture_with_processor_evaluates_coalitions(self) -> None:
        """Pixel-space masking + processor round-trip yields per-coalition logits."""
        from .conftest import FixedMasksStrategy

        masks = np.zeros((2, 8, 8), dtype=bool)
        masks[0, :, :4] = True
        masks[1, :, 4:] = True
        arch = CNNArchitecture(
            model=FakeSwallowingModel(),
            processor=FakeProcessor(),
            player_strategy=FixedMasksStrategy(masks),
        )
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
        arch.prepare(image)

        coalitions = torch.tensor([[True, True], [False, False], [True, False]])
        out = arch.value_function(coalitions)
        assert out.shape == (3,)
        # Full coalition: unmasked image sum; mean-color masking keeps the sum
        # constant, so compare against the exact unmasked total instead.
        expected_full = float(image.astype(np.float64).sum() / 255.0)
        assert out[0].item() == pytest.approx(expected_full, rel=1e-4)

    def test_transformer_init_raises_for_masking_ignoring_model(self) -> None:
        """Constructing the transformer path around a swallowing model fails at init."""
        from shapiq.vision.masking import BoolMaskedPosStrategy
        from shapiq.vision.players import PatchStrategy

        with pytest.raises(ValueError, match="bool_masked_pos"):
            TransformerArchitecture(
                model=FakeSwallowingModel(),
                processor=FakeProcessor(),
                masking_strategy=BoolMaskedPosStrategy(),
                player_strategy=PatchStrategy(grid_size=4, n_players=4),
            )


class TestExplainerAutoDispatch:
    def test_explainer_accepts_raw_vit_like_model(self) -> None:
        from shapiq.interaction_values import InteractionValues
        from shapiq.vision.explainer import ImageExplainer

        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
        explainer = ImageExplainer(
            model=FakeMaskableViT(),
            data=image,
            processor=FakeProcessor(),
            batch_size=8,
            random_state=0,
        )
        assert isinstance(explainer.architecture, TransformerArchitecture)
        result = explainer.explain_function(x=None, budget=16)
        assert isinstance(result, InteractionValues)
        assert np.isfinite(result.values).all()
