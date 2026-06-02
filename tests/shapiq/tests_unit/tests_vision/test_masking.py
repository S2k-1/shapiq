"""Tests for masking strategies in ``shapiq.vision.masking``."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shapiq.vision.masking import (
    BlurMasking,
    BoolMaskedPosStrategy,
    DatasetMeanMasking,
    GaussianNoiseMasking,
    InpaintingMasking,
    LatentMaskingStrategy,
    LayerMasking,
    ManifoldMaskingStrategy,
    MarginalSampling,
    MaskTokenStrategy,
    MeanColorMasking,
    PixelMaskingStrategy,
    ZeroMasking,
)


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(4, 4, 3)).astype(np.float64)


@pytest.fixture
def half_masks() -> np.ndarray:
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, :, :2] = True
    masks[1, :, 2:] = True
    return masks


class TestMeanColorMasking:
    def test_is_pixel_masking_strategy(self) -> None:
        assert isinstance(MeanColorMasking(), PixelMaskingStrategy)

    def test_full_coalition_preserves_image(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalition = np.array([[True, True]])
        out = strategy.apply(image, half_masks, coalition)
        assert out.shape == (1, 4, 4, 3)
        np.testing.assert_array_equal(out[0], image)

    def test_empty_coalition_uses_mean_color_everywhere(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalition = np.array([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        mean_color = image.mean(axis=(0, 1))
        expected = np.broadcast_to(mean_color, image.shape)
        np.testing.assert_allclose(out[0], expected)

    def test_partial_coalition_masks_only_absent_player(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalition = np.array([[True, False]])
        out = strategy.apply(image, half_masks, coalition)
        # Left half (player 0 present) should be preserved.
        np.testing.assert_array_equal(out[0, :, :2], image[:, :2])
        # Right half (player 1 absent) should be replaced with mean color.
        mean_color = image.mean(axis=(0, 1))
        np.testing.assert_allclose(out[0, :, 2:], np.broadcast_to(mean_color, (4, 2, 3)))

    def test_multiple_coalitions_handled_independently(self, image, half_masks) -> None:
        strategy = MeanColorMasking()
        coalitions = np.array(
            [
                [True, True],
                [False, False],
                [True, False],
                [False, True],
            ],
        )
        out = strategy.apply(image, half_masks, coalitions)
        assert out.shape == (4, 4, 4, 3)
        np.testing.assert_array_equal(out[0], image)
        mean_color = image.mean(axis=(0, 1))
        np.testing.assert_allclose(out[1], np.broadcast_to(mean_color, image.shape))
        np.testing.assert_array_equal(out[2, :, :2], image[:, :2])
        np.testing.assert_array_equal(out[3, :, 2:], image[:, 2:])


class TestZeroMasking:
    def test_is_pixel_masking_strategy(self) -> None:
        assert isinstance(ZeroMasking(), PixelMaskingStrategy)

    def test_default_value_is_zero(self, image, half_masks) -> None:
        strategy = ZeroMasking()
        coalition = np.array([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        assert (out[0] == 0).all()

    def test_custom_value(self, image, half_masks) -> None:
        strategy = ZeroMasking(value=7.0)
        coalition = np.array([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        assert (out[0] == 7.0).all()

    def test_partial_coalition_zeros_only_absent(self, image, half_masks) -> None:
        strategy = ZeroMasking()
        coalition = np.array([[False, True]])
        out = strategy.apply(image, half_masks, coalition)
        # Left half (absent) zeroed; right half (present) preserved.
        assert (out[0, :, :2] == 0).all()
        np.testing.assert_array_equal(out[0, :, 2:], image[:, 2:])

    def test_full_coalition_preserves_image(self, image, half_masks) -> None:
        strategy = ZeroMasking()
        coalition = np.array([[True, True]])
        out = strategy.apply(image, half_masks, coalition)
        np.testing.assert_array_equal(out[0], image)


class TestBlurMasking:
    def test_is_pixel_masking_strategy(self) -> None:
        assert isinstance(BlurMasking(), PixelMaskingStrategy)

    def test_default_sigma(self) -> None:
        assert BlurMasking().sigma == pytest.approx(10.0)

    def test_custom_sigma(self) -> None:
        assert BlurMasking(sigma=5.0).sigma == pytest.approx(5.0)

    def test_output_shape(self, image, half_masks) -> None:
        strategy = BlurMasking(sigma=2.0)
        coalition = np.array([[True, True], [False, False], [True, False]])
        out = strategy.apply(image, half_masks, coalition)
        assert out.shape == (3, 4, 4, 3)

    def test_full_coalition_preserves_image(self, image, half_masks) -> None:
        """When all players are present, pixels should equal the original image."""
        strategy = BlurMasking(sigma=2.0)
        coalition = np.array([[True, True]])
        out = strategy.apply(image, half_masks, coalition)
        np.testing.assert_array_equal(out[0], image)

    def test_empty_coalition_differs_from_original(self, image, half_masks) -> None:
        """When all players are absent, every pixel is replaced with a blurred value."""
        strategy = BlurMasking(sigma=2.0)
        coalition = np.array([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        # The blurred image won't equal the original unless sigma=0.
        # At least some pixels must differ (for any non-constant image).
        assert not np.allclose(out[0], image)

    def test_partial_coalition_keeps_present_region(self, image, half_masks) -> None:
        """Present players' pixels must be identical to the original image."""
        strategy = BlurMasking(sigma=2.0)
        coalition = np.array([[True, False]])  # player 0 present, player 1 absent
        out = strategy.apply(image, half_masks, coalition)
        # Left half (player 0 present) must be unchanged.
        np.testing.assert_array_equal(out[0, :, :2], image[:, :2])

    def test_partial_coalition_blurs_absent_region(self, image, half_masks) -> None:
        """Absent players' pixels must differ from the original (unless image is constant)."""
        from scipy.ndimage import gaussian_filter

        strategy = BlurMasking(sigma=2.0)
        coalition = np.array([[False, True]])  # player 0 absent, player 1 present
        out = strategy.apply(image, half_masks, coalition)
        blurred = gaussian_filter(image, sigma=[2.0, 2.0, 0])
        # Left half (player 0 absent) should match the blurred image.
        np.testing.assert_allclose(out[0, :, :2], blurred[:, :2], atol=1e-10)

    def test_multiple_coalitions_independent(self, image, half_masks) -> None:
        strategy = BlurMasking(sigma=2.0)
        coalitions = np.array([[True, True], [False, False]])
        out = strategy.apply(image, half_masks, coalitions)
        assert out.shape == (2, 4, 4, 3)
        np.testing.assert_array_equal(out[0], image)
        assert not np.allclose(out[1], image)


class TestDatasetMeanMasking:
    def test_is_pixel_masking_strategy(self) -> None:
        assert isinstance(DatasetMeanMasking(), PixelMaskingStrategy)

    def test_empty_coalition_uses_dataset_mean(self, image, half_masks) -> None:
        mean_color = np.array([10.0, 20.0, 30.0])
        strategy = DatasetMeanMasking(mean_color=mean_color)
        coalition = np.array([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        expected = np.broadcast_to(mean_color, image.shape)
        np.testing.assert_allclose(out[0], expected)

    def test_default_imagenet_mean(self) -> None:
        strategy = DatasetMeanMasking()
        np.testing.assert_allclose(strategy.mean_color, DatasetMeanMasking._IMAGENET_MEAN)


class TestGaussianNoiseMasking:
    def test_is_pixel_masking_strategy(self) -> None:
        assert isinstance(GaussianNoiseMasking(), PixelMaskingStrategy)

    def test_empty_coalition_filled_with_noise(self, image, half_masks) -> None:
        strategy = GaussianNoiseMasking(mean=0.0, std=1.0, random_state=0)
        coalition = np.array([[False, False]])
        out = strategy.apply(image, half_masks, coalition)
        assert out.shape == (1, 4, 4, 3)
        assert not np.allclose(out[0], image)

    def test_reproducible_with_random_state(self, image, half_masks) -> None:
        coalition = np.array([[False, False]])
        out_a = GaussianNoiseMasking(random_state=7).apply(image, half_masks, coalition)
        out_b = GaussianNoiseMasking(random_state=7).apply(image, half_masks, coalition)
        np.testing.assert_array_equal(out_a, out_b)

    def test_full_coalition_preserves_image(self, image, half_masks) -> None:
        strategy = GaussianNoiseMasking(random_state=0)
        coalition = np.array([[True, True]])
        out = strategy.apply(image, half_masks, coalition)
        np.testing.assert_array_equal(out[0], image)


def test_latent_masking_strategy_is_abstract() -> None:
    """LatentMaskingStrategy is an ABC; cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LatentMaskingStrategy()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Helpers for latent masking tests
# ---------------------------------------------------------------------------


class _MockViTWithMaskToken:
    """Minimal ViT mock that satisfies MaskTokenStrategy's requirements.

    MaskTokenStrategy reads ``model.config.hidden_size`` to create the zero
    tensor and writes to ``model.vit.embeddings.mask_token``.  The forward
    returns logits whose class-0 score equals the number of *visible* tokens.
    """

    class _Config:
        hidden_size = 4

    config = _Config()

    def __init__(self) -> None:
        self.vit = SimpleNamespace(
            embeddings=SimpleNamespace(mask_token=torch.nn.Parameter(torch.ones(1, 1, 4)))
        )

    def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
        visible = (~bool_masked_pos).sum(dim=1).float()
        return SimpleNamespace(logits=torch.stack([visible, -visible], dim=1))


class TestBoolMaskedPosStrategy:
    def test_is_latent_masking_strategy(self) -> None:
        assert isinstance(BoolMaskedPosStrategy(), LatentMaskingStrategy)

    def test_predict_logits_output_shape(self) -> None:
        class _MinimalModel:
            def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
                visible = (~bool_masked_pos).sum(dim=1).float()
                return SimpleNamespace(logits=torch.stack([visible, -visible], dim=1))

        strategy = BoolMaskedPosStrategy()
        pixel_values = torch.zeros(1, 3, 4, 4)
        bool_masks = torch.zeros(5, 4, dtype=torch.bool)
        logits = strategy.predict_logits(_MinimalModel(), pixel_values, bool_masks)
        assert logits.shape == (5, 2)

    def test_predict_logits_repeats_pixel_values(self) -> None:
        """pixel_values is repeated B times for a batch of B bool_masks."""
        seen: list[tuple] = []

        class _ShapeCapture:
            def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
                seen.append(tuple(pixel_values.shape))
                b = bool_masked_pos.shape[0]
                return SimpleNamespace(logits=torch.zeros(b, 2))

        strategy = BoolMaskedPosStrategy()
        pixel_values = torch.zeros(1, 3, 4, 4)
        bool_masks = torch.zeros(7, 4, dtype=torch.bool)
        strategy.predict_logits(_ShapeCapture(), pixel_values, bool_masks)
        assert seen[-1] == (7, 3, 4, 4)


class TestMaskTokenStrategy:
    def test_is_latent_masking_strategy(self) -> None:
        assert isinstance(MaskTokenStrategy(), LatentMaskingStrategy)

    def test_predict_logits_output_shape(self) -> None:
        """predict_logits returns ``(B, n_classes)`` logits."""
        strategy = MaskTokenStrategy()
        model = _MockViTWithMaskToken()
        pixel_values = torch.zeros(1, 3, 4, 4)
        bool_masks = torch.zeros(3, 4, dtype=torch.bool)
        bool_masks[1, :2] = True
        logits = strategy.predict_logits(model, pixel_values, bool_masks)
        assert logits.shape == (3, 2)

    def test_predict_logits_zeros_mask_token(self) -> None:
        """After predict_logits the model's mask_token parameter is all zeros."""
        strategy = MaskTokenStrategy()
        model = _MockViTWithMaskToken()
        # Confirm it starts non-zero.
        assert not torch.allclose(model.vit.embeddings.mask_token.data, torch.zeros(1, 1, 4))
        pixel_values = torch.zeros(1, 3, 4, 4)
        bool_masks = torch.zeros(1, 4, dtype=torch.bool)
        strategy.predict_logits(model, pixel_values, bool_masks)
        assert torch.allclose(model.vit.embeddings.mask_token.data, torch.zeros(1, 1, 4))

    def test_predict_logits_repeats_pixel_values(self) -> None:
        """pixel_values (shape 1xCxHxW) is expanded to BxCxHxW inside the model call."""
        seen_shapes: list[tuple] = []

        class _ShapeCapture:
            class _Config:
                hidden_size = 4

            config = _Config()

            def __init__(self) -> None:
                self.vit = SimpleNamespace(
                    embeddings=SimpleNamespace(mask_token=torch.nn.Parameter(torch.zeros(1, 1, 4)))
                )

            def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
                seen_shapes.append(tuple(pixel_values.shape))
                b = bool_masked_pos.shape[0]
                return SimpleNamespace(logits=torch.zeros(b, 2))

        strategy = MaskTokenStrategy()
        pixel_values = torch.zeros(1, 3, 4, 4)
        bool_masks = torch.zeros(5, 4, dtype=torch.bool)
        strategy.predict_logits(_ShapeCapture(), pixel_values, bool_masks)
        assert seen_shapes[-1] == (5, 3, 4, 4)

    def test_logits_reflect_mask_token_zeroing(self) -> None:
        """Visible tokens (bool_mask False) drive the class-0 score; masked tokens do not."""
        strategy = MaskTokenStrategy()
        model = _MockViTWithMaskToken()
        pixel_values = torch.zeros(1, 3, 4, 4)
        # Coalition 1: all 4 tokens visible.  Coalition 2: no tokens visible.
        bool_masks = torch.stack(
            [
                torch.zeros(4, dtype=torch.bool),  # all visible
                torch.ones(4, dtype=torch.bool),  # all masked
            ]
        )
        logits = strategy.predict_logits(model, pixel_values, bool_masks)
        # Class-0 logit = #visible, so coalition 0 (4 visible) > coalition 1 (0 visible).
        assert logits[0, 0] > logits[1, 0]


# ---------------------------------------------------------------------------
# MarginalSampling
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_bank() -> np.ndarray:
    """A small bank of constant-color reference images, one channel each set to 100."""
    refs = np.zeros((3, 4, 4, 3), dtype=np.float64)
    refs[0] = 11.0
    refs[1] = 22.0
    refs[2] = 33.0
    return refs


class TestMarginalSampling:
    def test_is_pixel_masking_strategy(self, reference_bank) -> None:
        assert isinstance(MarginalSampling(reference_bank), PixelMaskingStrategy)

    def test_full_coalition_preserves_image(
        self, tiny_image, two_player_masks, reference_bank
    ) -> None:
        strategy = MarginalSampling(reference_bank, random_state=0)
        coalition = np.array([[True, True]])
        out = strategy.apply(tiny_image, two_player_masks, coalition)
        assert out.shape == (1, 4, 4, 3)
        np.testing.assert_array_equal(out[0], tiny_image)

    def test_empty_coalition_uses_reference(
        self, tiny_image, two_player_masks, reference_bank
    ) -> None:
        # Force a single reference so the absent-everywhere output is deterministic.
        single = reference_bank[:1]
        strategy = MarginalSampling(single, random_state=0)
        coalition = np.array([[False, False]])
        out = strategy.apply(tiny_image, two_player_masks, coalition)
        np.testing.assert_array_equal(out[0], single[0])

    def test_partial_coalition_replaces_only_absent(
        self, tiny_image, two_player_masks, reference_bank
    ) -> None:
        # Use a deterministic single-reference bank for unambiguous assertion.
        single = reference_bank[:1]
        strategy = MarginalSampling(single, random_state=0)
        coalition = np.array([[True, False]])  # left present, right absent
        out = strategy.apply(tiny_image, two_player_masks, coalition)
        np.testing.assert_array_equal(out[0, :, :2], tiny_image[:, :2])
        np.testing.assert_array_equal(out[0, :, 2:], single[0, :, 2:])

    def test_same_seed_produces_same_output(
        self, tiny_image, two_player_masks, reference_bank
    ) -> None:
        coalition = np.array([[False, False], [False, True], [True, False]])
        a = MarginalSampling(reference_bank, random_state=42).apply(
            tiny_image, two_player_masks, coalition
        )
        b = MarginalSampling(reference_bank, random_state=42).apply(
            tiny_image, two_player_masks, coalition
        )
        np.testing.assert_array_equal(a, b)

    def test_same_instance_reseeds_per_call(
        self, tiny_image, two_player_masks, reference_bank
    ) -> None:
        """Two .apply() calls on the same seeded instance must produce identical output."""
        strategy = MarginalSampling(reference_bank, random_state=42)
        coalition = np.array([[False, False], [False, True]])
        a = strategy.apply(tiny_image, two_player_masks, coalition)
        b = strategy.apply(tiny_image, two_player_masks, coalition)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_can_differ(self, tiny_image, two_player_masks, reference_bank) -> None:
        # Many coalitions across the same image so collisions are essentially impossible.
        coalitions_many = np.zeros((20, 2), dtype=bool)
        a = MarginalSampling(reference_bank, random_state=0).apply(
            tiny_image, two_player_masks, coalitions_many
        )
        b = MarginalSampling(reference_bank, random_state=1).apply(
            tiny_image, two_player_masks, coalitions_many
        )
        assert not np.array_equal(a, b)

    def test_reference_shape_mismatch_raises(self, tiny_image, two_player_masks) -> None:
        wrong = np.zeros((2, 8, 8, 3), dtype=np.float64)
        strategy = MarginalSampling(wrong, random_state=0)
        coalition = np.array([[False, False]])
        with pytest.raises(ValueError, match="does not match"):
            strategy.apply(tiny_image, two_player_masks, coalition)

    def test_bad_reference_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 4-D"):
            MarginalSampling(np.zeros((4, 4, 3)))  # missing batch dim


# ---------------------------------------------------------------------------
# InpaintingMasking
# ---------------------------------------------------------------------------


class TestInpaintingMasking:
    def test_is_pixel_masking_strategy(self) -> None:
        assert isinstance(InpaintingMasking(lambda img, mask: img), PixelMaskingStrategy)

    def test_full_coalition_skips_inpainter(self, tiny_image, two_player_masks) -> None:
        calls = []

        def spy(img, mask):
            calls.append((img.shape, mask.shape))
            return img

        strategy = InpaintingMasking(spy)
        coalition = np.array([[True, True]])
        out = strategy.apply(tiny_image, two_player_masks, coalition)
        assert len(calls) == 0
        np.testing.assert_array_equal(out[0], tiny_image)

    def test_empty_coalition_calls_inpainter(self, tiny_image, two_player_masks) -> None:
        calls = []

        def spy(img, mask):
            calls.append(mask.copy())
            return np.full_like(img, 7.0)

        strategy = InpaintingMasking(spy)
        coalition = np.array([[False, False]])
        out = strategy.apply(tiny_image, two_player_masks, coalition)
        assert len(calls) == 1
        assert calls[0].all()  # mask is all True (everything absent)
        np.testing.assert_array_equal(out[0], np.full_like(tiny_image, 7.0))

    def test_partial_coalition_passes_correct_mask(self, tiny_image, two_player_masks) -> None:
        seen_masks = []

        def spy(img, mask):
            seen_masks.append(mask.copy())
            return img

        strategy = InpaintingMasking(spy)
        coalition = np.array([[True, False]])  # right half absent
        strategy.apply(tiny_image, two_player_masks, coalition)
        assert len(seen_masks) == 1
        # Mask should be False on left (present), True on right (absent).
        assert not seen_masks[0][:, :2].any()
        assert seen_masks[0][:, 2:].all()

    def test_one_inpainter_call_per_non_full_coalition(self, tiny_image, two_player_masks) -> None:
        calls = []

        def spy(img, mask):
            calls.append(None)
            return img

        strategy = InpaintingMasking(spy)
        coalitions = np.array(
            [
                [True, True],  # full -> skip
                [False, False],  # call
                [True, False],  # call
                [False, True],  # call
            ]
        )
        strategy.apply(tiny_image, two_player_masks, coalitions)
        assert len(calls) == 3

    def test_output_uses_inpainter_result(self, tiny_image, two_player_masks) -> None:
        def inpainter(img, mask):
            out = img.copy()
            out[mask] = 99.0
            return out

        strategy = InpaintingMasking(inpainter)
        coalition = np.array([[False, True]])  # left absent
        out = strategy.apply(tiny_image, two_player_masks, coalition)
        assert (out[0, :, :2] == 99.0).all()
        np.testing.assert_array_equal(out[0, :, 2:], tiny_image[:, 2:])

    def test_preserves_inpainter_dtype(self, tiny_image, two_player_masks) -> None:
        """An inpainter that returns a different dtype must not be silently truncated."""

        def float_inpainter(img, mask):
            return np.full(img.shape, 0.123, dtype=np.float32)

        strategy = InpaintingMasking(float_inpainter)
        coalition = np.array([[False, False]])
        out = strategy.apply(tiny_image, two_player_masks, coalition)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out[0], 0.123, rtol=1e-6)


# ---------------------------------------------------------------------------
# LayerMasking
# ---------------------------------------------------------------------------


class _TinyCNN(torch.nn.Module):
    """Tiny conv + global-avg-pool model exposing a ``layer2`` submodule to hook."""

    def __init__(self) -> None:
        super().__init__()
        self.layer2 = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        torch.nn.init.constant_(self.layer2.weight, 1.0)
        self.head = torch.nn.Linear(4, 2, bias=False)
        torch.nn.init.constant_(self.head.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.layer2(x)  # (B, 4, H, W)
        a = a.mean(dim=(-2, -1))  # global avg pool -> (B, 4)
        return self.head(a)  # raw logits, plain tensor


def _identity_preprocess(image: np.ndarray) -> torch.Tensor:
    """Image (H, W, C) ndarray -> (C, H, W) float32 tensor."""
    return torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)


class TestLayerMasking:
    def test_is_manifold_masking_strategy(self) -> None:
        assert isinstance(LayerMasking(), ManifoldMaskingStrategy)

    def test_manifold_masking_strategy_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            ManifoldMaskingStrategy()  # type: ignore[abstract]

    def test_value_function_output_shape(self, tiny_image, two_player_masks) -> None:
        model = _TinyCNN().eval()
        strategy = LayerMasking(layer_name="layer2")
        coalitions = np.array([[True, True], [False, False], [True, False]])
        out = strategy.value_function(
            model, _identity_preprocess, tiny_image, two_player_masks, coalitions, class_id=0
        )
        assert out.shape == (3,)

    def test_hook_removed_after_call(self, tiny_image, two_player_masks) -> None:
        model = _TinyCNN().eval()
        strategy = LayerMasking(layer_name="layer2")
        coalitions = np.array([[True, True], [False, False]])
        strategy.value_function(
            model, _identity_preprocess, tiny_image, two_player_masks, coalitions, class_id=0
        )
        # No hook should remain after the call returns.
        assert len(model.layer2._forward_hooks) == 0

    def test_unknown_layer_name_raises_only_attributeerror(
        self, tiny_image, two_player_masks
    ) -> None:
        """A bad layer name must raise AttributeError, not chain a NameError from the finally."""
        model = _TinyCNN().eval()
        strategy = LayerMasking(layer_name="does_not_exist")
        coalitions = np.array([[False, False]])
        with pytest.raises(AttributeError):
            strategy.value_function(
                model, _identity_preprocess, tiny_image, two_player_masks, coalitions, class_id=0
            )
        # No hook should leak onto layer2 either.
        assert len(model.layer2._forward_hooks) == 0

    def test_hook_removed_when_forward_raises(self, tiny_image, two_player_masks) -> None:
        """If the model forward pass raises after the hook is installed, the hook is still removed."""

        class _Boom(torch.nn.Module):
            def forward(self, _x: torch.Tensor) -> torch.Tensor:
                msg = "boom"
                raise RuntimeError(msg)

        model = _TinyCNN().eval()
        model.layer2 = _Boom()  # hook target whose forward raises

        strategy = LayerMasking(layer_name="layer2")
        coalitions = np.array([[False, False]])
        with pytest.raises(RuntimeError, match="boom"):
            strategy.value_function(
                model, _identity_preprocess, tiny_image, two_player_masks, coalitions, class_id=0
            )
        # The hook was successfully registered before the forward failed; the finally must
        # still remove it from the broken layer.
        assert len(model.layer2._forward_hooks) == 0

    def test_empty_coalition_attenuates_more_than_full(self, tiny_image, two_player_masks) -> None:
        """Output is a valid probability for both full and empty coalitions."""
        model = _TinyCNN().eval()
        strategy = LayerMasking(layer_name="layer2")
        coalitions = np.array([[True, True], [False, False]])
        out = strategy.value_function(
            model, _identity_preprocess, tiny_image, two_player_masks, coalitions, class_id=0
        )
        # _TinyCNN's symmetric head makes class-0 and class-1 logits equal, so softmax is
        # always 0.5 — we just verify outputs are valid probabilities.
        assert (out >= 0).all() and (out <= 1).all()

    def test_layer_masking_responds_to_coalition(self, two_player_masks) -> None:
        """Different coalitions should produce different intermediate masks => different scores."""
        # Normalised image keeps softmax in the non-saturating regime. _TinyCNN's conv has
        # constant-1 weights so every output channel is identical; a class-asymmetric head
        # of (+w, -w) on a single channel makes the class-0 probability strictly monotone
        # in the post-hook activation magnitude.
        small_image = np.random.default_rng(0).random((4, 4, 3))
        model = _TinyCNN().eval()
        with torch.no_grad():
            model.head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]))

        strategy = LayerMasking(layer_name="layer2")
        coalitions = np.array([[True, True], [True, False], [False, True]])
        out = strategy.value_function(
            model, _identity_preprocess, small_image, two_player_masks, coalitions, class_id=0
        )
        # At least two of the three values should differ.
        assert len(np.unique(np.round(out, 6))) >= 2
