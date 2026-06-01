"""Tests for the HuggingFace/CLIP/DINOv2/custom-ViT architectures.

These tests use lightweight mock objects that mimic the HuggingFace model and
processor interface so we don't need ``transformers`` or any real weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shapiq.vision.architecture import (
    CLIPArchitecture,
    ConvNeXtArchitecture,
    CustomViTArchitecture,
    DINOv2Architecture,
    HuggingFacePixelArchitecture,
    ModelArchitectureStrategy,
    ViTArchitecture,
)
from shapiq.vision.imputer import ImageImputer
from shapiq.vision.masking import (
    BoolMaskedPosStrategy,
    MaskTokenStrategy,
    MeanColorMasking,
    ZeroMasking,
)
from shapiq.vision.players import PatchStrategy

from .conftest import FixedMasksStrategy

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockImageProcessor:
    """Mimics a HF image processor: stacks HWC images into (B, C, H, W) tensors."""

    def __call__(self, images=None, text=None, return_tensors="pt", *, padding=True):
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            arr = np.stack([np.asarray(img, dtype=np.float32) for img in images], axis=0)
            pixel_values = torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())
            return {"pixel_values": pixel_values}
        if text is not None:
            # Single integer "token id" per prompt is enough for the mock CLIP.
            ids = torch.arange(len(text)).reshape(-1, 1)
            return {"input_ids": ids}
        msg = "either images or text must be supplied"
        raise ValueError(msg)


class _MockHFClassifier:
    """Mimics ``ConvNeXtForImageClassification``-style models.

    The forward returns a namespace with ``.logits`` of shape (B, n_classes).
    Class ``0`` logit equals the per-image mean pixel value (so a brighter image
    scores higher), class ``1`` logit is its negation. This makes the resulting
    probabilities a deterministic function of the masked images.
    """

    def __init__(self, n_classes: int = 2) -> None:
        self.n_classes = n_classes
        self.calls: list[torch.Tensor] = []

    def __call__(self, pixel_values=None, **_):
        self.calls.append(pixel_values)
        means = pixel_values.float().mean(dim=(1, 2, 3))  # (B,)
        logits = torch.stack(
            [means, -means] + [torch.zeros_like(means)] * (self.n_classes - 2),
            dim=1,
        )
        return SimpleNamespace(logits=logits)


class _MockBackbone:
    """Mimics a DINOv2-style backbone.

    Returns a namespace with ``.last_hidden_state`` of shape (B, n_tokens, dim).
    The CLS token (index 0) is the mean of the image flattened to a 4-dim vector
    so the cosine similarity is well-defined and depends on the image content.
    """

    def __call__(self, pixel_values=None, **_):
        b = pixel_values.shape[0]
        flat = pixel_values.float().reshape(b, -1)
        # Project to a 4-dim "feature" via deterministic sums over chunks of the flat vector.
        feats = torch.stack(
            [
                flat.mean(dim=1),
                flat.std(dim=1) + 1e-6,
                flat[:, : flat.shape[1] // 2].mean(dim=1),
                flat[:, flat.shape[1] // 2 :].mean(dim=1),
            ],
            dim=1,
        )
        # Pad to (B, n_tokens, dim) with the CLS token at index 0.
        hidden = torch.zeros(b, 2, 4)
        hidden[:, 0] = feats
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=None)


class _MockCLIPModel:
    """Mimics ``CLIPModel``: provides get_image_features, get_text_features, logit_scale."""

    def __init__(self) -> None:
        self.logit_scale = torch.tensor(np.log(1.0))  # exp(.) == 1.0 → no scaling
        # Two fixed text features (2-D for simplicity).
        self._text_table = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def get_text_features(self, input_ids=None, **_):
        # input_ids has shape (n_prompts, 1) — pick the matching row.
        return self._text_table[input_ids.squeeze(-1)]

    def get_image_features(self, pixel_values=None, **_):
        b = pixel_values.shape[0]
        flat = pixel_values.float().reshape(b, -1)
        return torch.stack([flat.mean(dim=1), flat.std(dim=1) + 1e-6], dim=1)


class _MockMaskedViT:
    """Mimics a ViT that accepts ``bool_masked_pos`` and returns logits.

    The logit for class 0 is the sum of ``~bool_masked_pos`` (i.e. number of
    visible tokens), so coalitions with more visible patches score higher.
    """

    def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
        visible = (~bool_masked_pos).sum(dim=1).float()
        logits = torch.stack([visible, -visible], dim=1)
        return SimpleNamespace(logits=logits)


@pytest.fixture
def hwc_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(8, 8, 3)).astype(np.float64)


@pytest.fixture
def two_player_masks_8x8() -> np.ndarray:
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, :, :4] = True
    masks[1, :, 4:] = True
    return masks


# ---------------------------------------------------------------------------
# HuggingFacePixelArchitecture (and ConvNeXtArchitecture)
# ---------------------------------------------------------------------------


class TestHuggingFacePixelArchitecture:
    def test_is_architecture_strategy(self) -> None:
        arch = HuggingFacePixelArchitecture(_MockHFClassifier(), _MockImageProcessor())
        assert isinstance(arch, ModelArchitectureStrategy)

    def test_default_masking_strategy(self) -> None:
        arch = HuggingFacePixelArchitecture(_MockHFClassifier(), _MockImageProcessor())
        assert isinstance(arch.default_masking_strategy(), MeanColorMasking)

    def test_prepare_autodetects_class_id(self, hwc_image, two_player_masks_8x8) -> None:
        arch = HuggingFacePixelArchitecture(
            _MockHFClassifier(), _MockImageProcessor(), masking_strategy=ZeroMasking()
        )
        assert arch.class_id is None
        arch.prepare(hwc_image, FixedMasksStrategy(two_player_masks_8x8))
        # Mean pixel value of the random uint8 image is positive → class 0 wins.
        assert arch.class_id == 0

    def test_prepare_respects_explicit_class_id(self, hwc_image, two_player_masks_8x8) -> None:
        arch = HuggingFacePixelArchitecture(
            _MockHFClassifier(),
            _MockImageProcessor(),
            class_id=1,
            masking_strategy=ZeroMasking(),
        )
        arch.prepare(hwc_image, FixedMasksStrategy(two_player_masks_8x8))
        assert arch.class_id == 1

    def test_value_function_shape_and_range(self, hwc_image, two_player_masks_8x8) -> None:
        arch = HuggingFacePixelArchitecture(
            _MockHFClassifier(), _MockImageProcessor(), masking_strategy=ZeroMasking()
        )
        arch.prepare(hwc_image, FixedMasksStrategy(two_player_masks_8x8))
        coalitions = np.array(
            [
                [False, False],
                [True, False],
                [False, True],
                [True, True],
            ],
        )
        out = arch.value_function(hwc_image, coalitions)
        assert out.shape == (4,)
        assert ((out >= 0.0) & (out <= 1.0)).all()
        # More visible pixels → higher logit-0 → higher prob of class 0.
        assert out[3] >= max(out[1], out[2])
        assert out[1] >= out[0]

    def test_empty_prediction_is_in_unit_interval(self, hwc_image, two_player_masks_8x8) -> None:
        arch = HuggingFacePixelArchitecture(
            _MockHFClassifier(), _MockImageProcessor(), masking_strategy=ZeroMasking()
        )
        arch.prepare(hwc_image, FixedMasksStrategy(two_player_masks_8x8))
        empty = arch.calc_empty_prediction(hwc_image)
        assert 0.0 <= empty <= 1.0
        # With ZeroMasking the input is all zeros → logits are (0, 0) → softmax 0.5.
        assert empty == pytest.approx(0.5)

    def test_works_inside_image_imputer(self, hwc_image, two_player_masks_8x8) -> None:
        arch = HuggingFacePixelArchitecture(
            _MockHFClassifier(), _MockImageProcessor(), masking_strategy=ZeroMasking()
        )
        imputer = ImageImputer(
            architecture=arch,
            image=hwc_image,
            player_strategy=FixedMasksStrategy(two_player_masks_8x8),
        )
        assert imputer.n_players == 2
        coalitions = np.array([[False, False], [True, True]])
        out = imputer.value_function(coalitions)
        assert out.shape == (2,)


class TestConvNeXtArchitecture:
    def test_subclass_of_hf_pixel(self) -> None:
        assert issubclass(ConvNeXtArchitecture, HuggingFacePixelArchitecture)

    def test_construction_and_prepare(self, hwc_image, two_player_masks_8x8) -> None:
        arch = ConvNeXtArchitecture(
            _MockHFClassifier(), _MockImageProcessor(), masking_strategy=ZeroMasking()
        )
        arch.prepare(hwc_image, FixedMasksStrategy(two_player_masks_8x8))
        assert arch.class_id == 0


# ---------------------------------------------------------------------------
# DINOv2Architecture
# ---------------------------------------------------------------------------


class TestDINOv2Architecture:
    def test_is_architecture_strategy(self) -> None:
        arch = DINOv2Architecture(_MockBackbone(), _MockImageProcessor())
        assert isinstance(arch, ModelArchitectureStrategy)

    def test_prepare_caches_reference_embedding_and_masks(
        self, hwc_image, two_player_masks_8x8
    ) -> None:
        arch = DINOv2Architecture(_MockBackbone(), _MockImageProcessor())
        arch.prepare(hwc_image, FixedMasksStrategy(two_player_masks_8x8))
        assert arch._reference_embedding is not None
        assert arch._reference_embedding.shape == (4,)
        assert arch._player_masks is not None

    def test_full_coalition_similarity_is_one(self, hwc_image, two_player_masks_8x8) -> None:
        arch = DINOv2Architecture(
            _MockBackbone(), _MockImageProcessor(), masking_strategy=MeanColorMasking()
        )
        arch.prepare(hwc_image, FixedMasksStrategy(two_player_masks_8x8))
        out = arch.value_function(hwc_image, np.array([[True, True]]))
        # Original image vs. itself → cosine similarity == 1.
        assert out.shape == (1,)
        assert out[0] == pytest.approx(1.0, abs=1e-5)

    def test_value_function_returns_finite_floats(self, hwc_image, two_player_masks_8x8) -> None:
        arch = DINOv2Architecture(_MockBackbone(), _MockImageProcessor())
        arch.prepare(hwc_image, FixedMasksStrategy(two_player_masks_8x8))
        out = arch.value_function(
            hwc_image,
            np.array([[False, False], [True, False], [False, True], [True, True]]),
        )
        assert out.shape == (4,)
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# CLIPArchitecture
# ---------------------------------------------------------------------------


class TestCLIPArchitecture:
    def test_is_architecture_strategy(self) -> None:
        arch = CLIPArchitecture(_MockCLIPModel(), _MockImageProcessor(), text_prompts=["a", "b"])
        assert isinstance(arch, ModelArchitectureStrategy)

    def test_prepare_caches_normalized_text_features(self, hwc_image, two_player_masks_8x8) -> None:
        arch = CLIPArchitecture(_MockCLIPModel(), _MockImageProcessor(), text_prompts=["a", "b"])
        arch.prepare(hwc_image, FixedMasksStrategy(two_player_masks_8x8))
        assert arch._text_features is not None
        assert arch._text_features.shape == (2, 2)
        # Already unit-norm in the mock; check post-normalization.
        norms = arch._text_features.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2))

    def test_value_function_returns_probabilities(self, hwc_image, two_player_masks_8x8) -> None:
        arch = CLIPArchitecture(
            _MockCLIPModel(),
            _MockImageProcessor(),
            text_prompts=["a", "b"],
            target_prompt_idx=0,
        )
        arch.prepare(hwc_image, FixedMasksStrategy(two_player_masks_8x8))
        out = arch.value_function(hwc_image, np.array([[False, False], [True, True]]))
        assert out.shape == (2,)
        assert ((out >= 0.0) & (out <= 1.0)).all()

    def test_target_prompt_idx_changes_output(self, hwc_image, two_player_masks_8x8) -> None:
        coalitions = np.array([[True, True]])
        common = {
            "model": _MockCLIPModel(),
            "processor": _MockImageProcessor(),
            "text_prompts": ["a", "b"],
        }
        arch_a = CLIPArchitecture(**common, target_prompt_idx=0)
        arch_b = CLIPArchitecture(**common, target_prompt_idx=1)
        arch_a.prepare(hwc_image, FixedMasksStrategy(two_player_masks_8x8))
        arch_b.prepare(hwc_image, FixedMasksStrategy(two_player_masks_8x8))
        # The two probabilities must sum to 1 (only 2 prompts).
        assert arch_a.value_function(hwc_image, coalitions)[0] + arch_b.value_function(
            hwc_image, coalitions
        )[0] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# CustomViTArchitecture
# ---------------------------------------------------------------------------


class TestCustomViTArchitecture:
    def _arch(self, n_tokens: int = 4, class_id: int = 0):
        pixel_values = torch.zeros(1, 3, 4, 4)
        return CustomViTArchitecture(
            model=_MockMaskedViT(),
            pixel_values=pixel_values,
            class_id=class_id,
            n_tokens=n_tokens,
        )

    def test_is_architecture_strategy(self) -> None:
        assert isinstance(self._arch(), ModelArchitectureStrategy)

    def test_default_player_strategy_is_patch_grid(self) -> None:
        arch = self._arch(n_tokens=9)
        strategy = arch.default_player_strategy()
        assert isinstance(strategy, PatchStrategy)
        assert strategy.n_players == 9
        assert strategy.grid_size == 3

    def test_default_masking_strategy_is_bool_masked_pos(self) -> None:
        arch = self._arch()
        assert isinstance(arch.default_masking_strategy(), BoolMaskedPosStrategy)

    def test_default_player_strategy_rejects_non_square_tokens(self) -> None:
        arch = self._arch(n_tokens=7)
        with pytest.raises(ValueError, match="perfect square"):
            arch.default_player_strategy()

    def test_value_function_more_visible_higher_score(self) -> None:
        arch = self._arch(n_tokens=4, class_id=0)
        # PatchStrategy with grid_size=2, 4 players: each player == one "patch".
        strategy = PatchStrategy(grid_size=2, n_players=4)
        arch.prepare(image=np.zeros((4, 4, 3)), player_strategy=strategy)

        coalitions = np.array(
            [
                [False, False, False, False],
                [True, False, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ],
        )
        out = arch.value_function(np.zeros((4, 4, 3)), coalitions)
        assert out.shape == (4,)
        # Monotonic in number of visible players (mock model's class-0 logit == #visible).
        assert out[0] < out[1] < out[2] < out[3]

    def test_value_function_accepts_1d_coalition(self) -> None:
        arch = self._arch(n_tokens=4, class_id=0)
        strategy = PatchStrategy(grid_size=2, n_players=4)
        arch.prepare(image=np.zeros((4, 4, 3)), player_strategy=strategy)
        out = arch.value_function(np.zeros((4, 4, 3)), np.array([True, True, True, True]))
        assert out.shape == (1,)

    def test_works_inside_image_imputer(self) -> None:
        arch = self._arch(n_tokens=4, class_id=0)
        imputer = ImageImputer(
            architecture=arch,
            image=np.zeros((4, 4, 3)),
            player_strategy=PatchStrategy(grid_size=2, n_players=4),
        )
        assert imputer.n_players == 4
        out = imputer.value_function(np.eye(4, dtype=bool))
        assert out.shape == (4,)


# ---------------------------------------------------------------------------
# ViTArchitecture — full functional tests
# ---------------------------------------------------------------------------


class _MockViTWithConfig:
    """HF-style ViT mock: has model.config and handles bool_masked_pos.

    config.image_size=16, config.patch_size=8 → grid_size=2 → 4 tokens.

    When called *without* bool_masked_pos (during prepare's class detection),
    the model returns a fixed logit favouring class 0.  When called *with*
    bool_masked_pos the class-0 logit equals the number of visible tokens,
    which lets us write deterministic monotonicity assertions.
    """

    class _Config:
        image_size = 16
        patch_size = 8  # → grid_size = image_size / patch_size = 2, n_tokens = 4
        hidden_size = 4

    config = _Config()

    def __init__(self) -> None:
        # Expose vit.embeddings.mask_token so MaskTokenStrategy tests can also use this mock.
        self.vit = SimpleNamespace(
            embeddings=SimpleNamespace(mask_token=torch.nn.Parameter(torch.zeros(1, 1, 4)))
        )

    def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
        b = pixel_values.shape[0]
        if bool_masked_pos is None:
            # Initial forward during prepare — class 0 wins (logit 2.0 vs 0.5).
            return SimpleNamespace(logits=torch.tensor([[2.0, 0.5]]).expand(b, -1).clone())
        visible = (~bool_masked_pos).sum(dim=1).float()
        return SimpleNamespace(logits=torch.stack([visible, -visible], dim=1))


class TestViTArchitectureFull:
    """Functional tests for ViTArchitecture: prepare / value_function / calc_empty_prediction."""

    @pytest.fixture
    def arch(self):
        """ViTArchitecture wired with BoolMaskedPosStrategy for easy mocking."""
        return ViTArchitecture(
            model=_MockViTWithConfig(),
            processor=_MockImageProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )

    @pytest.fixture
    def image_16x16(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.integers(0, 255, size=(16, 16, 3)).astype(np.float64)

    @pytest.fixture
    def patch_strategy_2x2(self) -> PatchStrategy:
        """2x2 macro-grid -> 4 players on a 4-token ViT."""
        return PatchStrategy(grid_size=2, n_players=4)

    def test_is_architecture_strategy(self) -> None:
        arch = ViTArchitecture(model=object(), processor=object())
        assert isinstance(arch, ModelArchitectureStrategy)

    def test_prepare_sets_class_id(self, arch, image_16x16, patch_strategy_2x2) -> None:
        assert arch._class_id is None
        arch.prepare(image_16x16, patch_strategy_2x2)
        # Mock returns logits [2.0, 0.5] → argmax == 0.
        assert arch._class_id == 0

    def test_prepare_caches_pixel_values(self, arch, image_16x16, patch_strategy_2x2) -> None:
        arch.prepare(image_16x16, patch_strategy_2x2)
        assert arch._pixel_values is not None
        assert arch._pixel_values.ndim == 4
        assert arch._pixel_values.shape[0] == 1

    def test_prepare_stores_player_strategy_ref(
        self, arch, image_16x16, patch_strategy_2x2
    ) -> None:
        arch.prepare(image_16x16, patch_strategy_2x2)
        assert arch._player_strategy_ref is patch_strategy_2x2

    def test_value_function_shape_and_finite(self, arch, image_16x16, patch_strategy_2x2) -> None:
        arch.prepare(image_16x16, patch_strategy_2x2)
        coalitions = np.array(
            [
                [False, False, False, False],
                [True, False, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        out = arch.value_function(image_16x16, coalitions)
        assert out.shape == (4,)
        assert np.isfinite(out).all()

    def test_value_function_monotone_in_visible_tokens(
        self, arch, image_16x16, patch_strategy_2x2
    ) -> None:
        """More visible tokens → higher class-0 probability (mock design guarantee)."""
        arch.prepare(image_16x16, patch_strategy_2x2)
        coalitions = np.array(
            [
                [False, False, False, False],
                [True, False, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        out = arch.value_function(image_16x16, coalitions)
        assert out[0] < out[1] < out[2] < out[3]

    def test_value_function_accepts_1d_coalition(
        self, arch, image_16x16, patch_strategy_2x2
    ) -> None:
        arch.prepare(image_16x16, patch_strategy_2x2)
        out = arch.value_function(image_16x16, np.array([True, True, True, True]))
        assert out.shape == (1,)

    def test_calc_empty_prediction_is_float(self, arch, image_16x16, patch_strategy_2x2) -> None:
        arch.prepare(image_16x16, patch_strategy_2x2)
        empty = arch.calc_empty_prediction(image_16x16)
        assert isinstance(empty, float)
        assert np.isfinite(empty)

    def test_calc_empty_prediction_all_masked(self, arch, image_16x16, patch_strategy_2x2) -> None:
        """All tokens masked → logits [0, 0] → softmax 0.5 for class 0."""
        arch.prepare(image_16x16, patch_strategy_2x2)
        empty = arch.calc_empty_prediction(image_16x16)
        # visible=0 → logits (0, 0) → softmax (0.5, 0.5)
        assert empty == pytest.approx(0.5, abs=1e-5)

    def test_works_inside_image_imputer(self, image_16x16, patch_strategy_2x2) -> None:
        arch = ViTArchitecture(
            model=_MockViTWithConfig(),
            processor=_MockImageProcessor(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        imputer = ImageImputer(
            architecture=arch,
            image=image_16x16,
            player_strategy=patch_strategy_2x2,
        )
        assert imputer.n_players == 4
        out = imputer.value_function(np.eye(4, dtype=bool))
        assert out.shape == (4,)
        assert np.isfinite(out).all()

    def test_works_with_mask_token_strategy(self, image_16x16, patch_strategy_2x2) -> None:
        """ViTArchitecture + MaskTokenStrategy runs through ImageImputer end-to-end.

        ``_MockViTWithConfig`` exposes ``vit.embeddings.mask_token`` and
        ``config.hidden_size``, satisfying ``MaskTokenStrategy``'s requirements.
        """
        arch = ViTArchitecture(
            model=_MockViTWithConfig(),
            processor=_MockImageProcessor(),
            masking_strategy=MaskTokenStrategy(),
        )
        imputer = ImageImputer(
            architecture=arch,
            image=image_16x16,
            player_strategy=patch_strategy_2x2,
        )
        assert imputer.n_players == 4
        out = imputer.value_function(np.eye(4, dtype=bool))
        assert out.shape == (4,)
        assert np.isfinite(out).all()
        # Monotonicity: more visible tokens → higher class-0 probability.
        coalitions = np.array(
            [
                [False, False, False, False],
                [True, False, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        vals = imputer.value_function(coalitions)
        assert vals[0] < vals[1] < vals[2] < vals[3]
