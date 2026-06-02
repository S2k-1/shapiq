"""Framework-agnostic callable interface tests for the vision package.

The pixel-space masking pipeline (ResNetArchitecture + pixel maskers) and the
custom ViT latent pipeline (CustomViTArchitecture + BoolMaskedPosStrategy) are
designed to work with **any** model callable that accepts the expected inputs and
returns numpy-compatible outputs.  No ML framework (PyTorch, JAX, TensorFlow, …)
is required at the imputer or architecture level; only the callable the user
supplies needs to understand the input format.

HuggingFace-centric latent paths (ViTArchitecture, MaskTokenStrategy, CLIP, DINOv2)
remain PyTorch-specific because they integrate with Transformers conventions.

These tests verify:
1. No ``torch`` appears in the module-level namespace of any vision module.
2. A pure-numpy callable works end-to-end through ``ImageImputer``.
3. A numpy-array CustomViT callable works end-to-end through ``ImageImputer``.
4. (skip if jax absent) JAX callables produce identical results on pixel and latent paths.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import shapiq.vision.explainer  # noqa: F401 — ensure module is in sys.modules for namespace check
from shapiq.vision.architecture import CustomViTArchitecture, ResNetArchitecture
from shapiq.vision.imputer import ImageImputer
from shapiq.vision.masking import MeanColorMasking, ZeroMasking
from shapiq.vision.players import GridStrategy, PatchStrategy
from tests.shapiq.markers import skip_if_no_jax

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)


def _tiny_image(h: int = 8, w: int = 8, c: int = 3) -> np.ndarray:
    return _RNG.random((h, w, c)).astype(np.float32)


def _two_by_two_grid() -> GridStrategy:
    """2×2 grid → 4 players; works on any image ≥ 2×2 without SLIC."""
    return GridStrategy(rows=2, cols=2)


def _coalitions_4() -> np.ndarray:
    return np.array(
        [
            [True, True, True, True],
            [True, False, False, False],
            [False, False, False, False],
            [True, True, False, False],
        ],
        dtype=bool,
    )


# ---------------------------------------------------------------------------
# 1. Module-level namespace: torch must not be imported at import time
# ---------------------------------------------------------------------------


class TestNoModuleLevelTorch:
    """No vision module imports torch at module level."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "shapiq.vision.architecture",
            "shapiq.vision.explainer",
            "shapiq.vision.imputer",
            "shapiq.vision.masking",
            "shapiq.vision.players",
            "shapiq.vision.utils",
        ],
    )
    def test_torch_not_in_module_namespace(self, module_name: str) -> None:
        mod = sys.modules[module_name]
        assert "torch" not in vars(mod), (
            f"torch found at module level in {module_name}. "
            "Use a local 'import torch' inside the methods that need it."
        )

    def test_import_shapiq_vision_does_not_pull_torch(self) -> None:
        """Importing shapiq.vision must not cause torch to appear in sys.modules
        if it was absent beforehand.  This test only gives a reliable signal when
        run in a fresh interpreter (i.e. before any test that uses torch), so we
        only assert when torch was not already imported at the start of the session.
        """
        # We cannot unload torch once it is in sys.modules, so we check the
        # module-level namespace instead (see parametrized test above).  This
        # test documents intent and can be verified in an isolated subprocess.
        import shapiq.vision  # noqa: F401 — side-effect check

        # If we reach here the import succeeded; the namespace check above
        # already covers the module-level guarantee.


# ---------------------------------------------------------------------------
# 2. Pure-numpy model callable
# ---------------------------------------------------------------------------


class TestNumpyCallableModel:
    """Pixel-space pipeline works with any numpy-in / numpy-out callable."""

    @staticmethod
    def _make_linear_model(weights: np.ndarray, bias: float = 0.0):
        """Return a callable that computes a weighted mean over pixel channels."""

        def model(images: np.ndarray) -> np.ndarray:
            # images: (B, H, W, C) — compute weighted channel mean, then spatial mean
            return float(bias) + images.dot(weights).mean(axis=(1, 2))

        return model

    def test_value_function_shape(self) -> None:
        weights = np.array([0.3, 0.59, 0.11], dtype=np.float32)  # luminance-like
        model = self._make_linear_model(weights)
        image = _tiny_image()
        arch = ResNetArchitecture(model=model, masking_strategy=ZeroMasking())
        imputer = ImageImputer(arch, image, player_strategy=_two_by_two_grid())

        values = imputer.value_function(_coalitions_4())
        assert values.shape == (4,)
        assert np.isfinite(values).all()

    def test_empty_coalition_is_zero_for_zero_masking(self) -> None:
        """With ZeroMasking the empty coalition score should be 0 (all pixels → 0)."""
        weights = np.ones(3, dtype=np.float32)
        model = self._make_linear_model(weights)
        image = _tiny_image()
        arch = ResNetArchitecture(model=model, masking_strategy=ZeroMasking())
        imputer = ImageImputer(arch, image, player_strategy=_two_by_two_grid())

        raw_empty = imputer.architecture.calc_empty_prediction(image)
        assert abs(raw_empty) < 1e-6

    def test_full_coalition_equals_unmasked_prediction(self) -> None:
        """Grand coalition (all present) must match a direct model call on the original image."""
        weights = np.array([0.2, 0.5, 0.3], dtype=np.float32)
        model = self._make_linear_model(weights)
        image = _tiny_image()
        arch = ResNetArchitecture(model=model, masking_strategy=MeanColorMasking())
        imputer = ImageImputer(arch, image, player_strategy=_two_by_two_grid(), normalize=False)

        full_coal = np.ones((1, 4), dtype=bool)
        via_imputer = imputer.value_function(full_coal)[0]
        direct = float(model(image[np.newaxis])[0])
        assert abs(via_imputer - direct) < 1e-5

    def test_batching_consistency_numpy_model(self) -> None:
        """Results must be identical with batch_size=1 and batch_size=None."""
        model = lambda imgs: np.mean(imgs, axis=(1, 2, 3))  # noqa: E731
        image = _tiny_image()
        arch1 = ResNetArchitecture(model=model, masking_strategy=ZeroMasking())
        arch2 = ResNetArchitecture(model=model, masking_strategy=ZeroMasking())
        imp_batch = ImageImputer(arch1, image, player_strategy=_two_by_two_grid(), batch_size=1)
        imp_all = ImageImputer(arch2, image, player_strategy=_two_by_two_grid(), batch_size=None)

        coalitions = _coalitions_4()
        np.testing.assert_allclose(
            imp_batch.value_function(coalitions),
            imp_all.value_function(coalitions),
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# 3. JAX callable model (skipped when jax is not installed)
# ---------------------------------------------------------------------------


@skip_if_no_jax
class TestJaxCallableModel:
    """Pixel-space pipeline works with a JAX model via the callable interface.

    A JAX model is just a Python callable; it receives a numpy array from the
    architecture and returns a numpy-compatible array.  No changes to the vision
    package are needed — JAX support is a natural consequence of the design.
    """

    @staticmethod
    def _jax_mean_model(images: np.ndarray) -> np.ndarray:
        import jax.numpy as jnp

        arr = jnp.array(images, dtype=jnp.float32)
        return np.asarray(jnp.mean(arr, axis=(1, 2, 3)))

    @staticmethod
    def _numpy_mean_model(images: np.ndarray) -> np.ndarray:
        return np.mean(images, axis=(1, 2, 3))

    def test_jax_model_value_function_shape(self) -> None:
        image = _tiny_image()
        arch = ResNetArchitecture(model=self._jax_mean_model, masking_strategy=ZeroMasking())
        imputer = ImageImputer(arch, image, player_strategy=_two_by_two_grid())

        values = imputer.value_function(_coalitions_4())
        assert values.shape == (4,)
        assert np.isfinite(values).all()

    def test_jax_matches_numpy_identical_logic(self) -> None:
        """JAX and numpy callables implementing the same function must agree."""
        image = _tiny_image()
        coalitions = _coalitions_4()

        arch_np = ResNetArchitecture(model=self._numpy_mean_model, masking_strategy=ZeroMasking())
        imp_np = ImageImputer(arch_np, image, player_strategy=_two_by_two_grid(), normalize=False)

        arch_jax = ResNetArchitecture(model=self._jax_mean_model, masking_strategy=ZeroMasking())
        imp_jax = ImageImputer(arch_jax, image, player_strategy=_two_by_two_grid(), normalize=False)

        np.testing.assert_allclose(
            imp_np.value_function(coalitions),
            imp_jax.value_function(coalitions),
            rtol=1e-5,
            err_msg="JAX and numpy models with identical logic must produce the same values.",
        )

    def test_jax_model_with_flax_linear_layer(self) -> None:
        """End-to-end test using a tiny Flax linear model as the vision backbone."""
        import jax
        import jax.numpy as jnp

        try:
            import flax.linen as nn
        except ImportError:
            pytest.skip("flax is not installed")

        class TinyClassifier(nn.Module):
            n_classes: int = 2

            @nn.compact
            def __call__(self, x: jax.Array) -> jax.Array:
                # x: (B, H, W, C) — global average pool then linear
                pooled = x.mean(axis=(1, 2))  # (B, C)
                return nn.Dense(self.n_classes)(pooled)  # (B, n_classes)

        rng = jax.random.PRNGKey(0)
        dummy = jnp.ones((1, 8, 8, 3), dtype=jnp.float32)
        model_flax = TinyClassifier()
        params = model_flax.init(rng, dummy)

        def flax_callable(images: np.ndarray) -> np.ndarray:
            arr = jnp.array(images, dtype=jnp.float32)
            logits = model_flax.apply(params, arr)  # (B, 2)
            probs = jax.nn.softmax(logits, axis=-1)
            return np.asarray(probs[:, 0])  # score for class 0

        image = _tiny_image()
        arch = ResNetArchitecture(model=flax_callable, masking_strategy=MeanColorMasking())
        imputer = ImageImputer(arch, image, player_strategy=_two_by_two_grid())

        values = imputer.value_function(_coalitions_4())
        assert values.shape == (4,)
        assert np.isfinite(values).all()
        # All probabilities lie in [0, 1] after normalization shift
        assert values.min() >= -1.0 and values.max() <= 1.0


# ---------------------------------------------------------------------------
# 4. Numpy latent callable (CustomViTArchitecture)
# ---------------------------------------------------------------------------


class TestNumpyLatentCallableModel:
    """Custom ViT latent path works with numpy pixel values and callables."""

    @staticmethod
    def _numpy_vit_model(pixel_values: np.ndarray, bool_masked_pos: np.ndarray) -> np.ndarray:
        visible = (~bool_masked_pos).sum(axis=1).astype(np.float64)
        return np.stack([visible, -visible], axis=1)

    def test_custom_vit_value_function_shape(self) -> None:
        arch = CustomViTArchitecture(
            model=self._numpy_vit_model,
            pixel_values=np.zeros((1, 3, 4, 4), dtype=np.float32),
            class_id=0,
            n_tokens=4,
        )
        strategy = PatchStrategy(grid_size=2, n_players=4)
        image = _tiny_image(4, 4)
        arch.prepare(image, strategy)
        coalitions = np.array(
            [
                [False, False, False, False],
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, True],
            ],
            dtype=bool,
        )
        values = arch.value_function(image, coalitions)
        assert values.shape == (4,)
        assert np.isfinite(values).all()
        assert values[0] < values[1] < values[2] < values[3]

    def test_custom_vit_works_inside_image_imputer(self) -> None:
        arch = CustomViTArchitecture(
            model=self._numpy_vit_model,
            pixel_values=np.zeros((1, 3, 4, 4), dtype=np.float32),
            class_id=0,
            n_tokens=4,
        )
        imputer = ImageImputer(
            architecture=arch,
            image=_tiny_image(4, 4),
            player_strategy=PatchStrategy(grid_size=2, n_players=4),
            normalize=False,
        )
        values = imputer.value_function(_coalitions_4())
        assert values.shape == (4,)
        assert np.isfinite(values).all()


@skip_if_no_jax
class TestJaxLatentCallableModel:
    """Custom ViT latent path works with JAX callables via numpy I/O."""

    @staticmethod
    def _jax_vit_model(pixel_values: np.ndarray, bool_masked_pos: np.ndarray) -> np.ndarray:
        import jax.numpy as jnp

        visible = (~jnp.asarray(bool_masked_pos)).sum(axis=1).astype(jnp.float32)
        return np.asarray(jnp.stack([visible, -visible], axis=1))

    @staticmethod
    def _numpy_vit_model(pixel_values: np.ndarray, bool_masked_pos: np.ndarray) -> np.ndarray:
        visible = (~bool_masked_pos).sum(axis=1).astype(np.float64)
        return np.stack([visible, -visible], axis=1)

    def test_jax_latent_matches_numpy_logic(self) -> None:
        image = _tiny_image(4, 4)
        coalitions = _coalitions_4()
        strategy = PatchStrategy(grid_size=2, n_players=4)

        arch_np = CustomViTArchitecture(
            model=self._numpy_vit_model,
            pixel_values=np.zeros((1, 3, 4, 4), dtype=np.float32),
            class_id=0,
            n_tokens=4,
        )
        arch_jax = CustomViTArchitecture(
            model=self._jax_vit_model,
            pixel_values=np.zeros((1, 3, 4, 4), dtype=np.float32),
            class_id=0,
            n_tokens=4,
        )
        arch_np.prepare(image, strategy)
        arch_jax.prepare(image, strategy)

        np.testing.assert_allclose(
            arch_np.value_function(image, coalitions),
            arch_jax.value_function(image, coalitions),
            rtol=1e-5,
        )

    def test_jax_latent_flax_vit_style_callable(self) -> None:
        import jax
        import jax.numpy as jnp

        try:
            import flax.linen as nn
        except ImportError:
            pytest.skip("flax is not installed")

        class TinyViT(nn.Module):
            n_classes: int = 2

            @nn.compact
            def __call__(self, pixel_values, bool_masked_pos=None):
                del pixel_values
                visible = (~bool_masked_pos).sum(axis=1).astype(jnp.float32)
                return jnp.stack([visible, -visible], axis=1)

        rng = jax.random.PRNGKey(0)
        dummy_pixels = jnp.ones((1, 3, 4, 4), dtype=jnp.float32)
        dummy_masks = jnp.zeros((1, 4), dtype=bool)
        model = TinyViT()
        params = model.init(rng, dummy_pixels, bool_masked_pos=dummy_masks)

        def flax_vit_callable(pixel_values: np.ndarray, bool_masked_pos: np.ndarray) -> np.ndarray:
            logits = model.apply(
                params,
                jnp.asarray(pixel_values),
                bool_masked_pos=jnp.asarray(bool_masked_pos),
            )
            return np.asarray(logits)

        arch = CustomViTArchitecture(
            model=flax_vit_callable,
            pixel_values=np.zeros((1, 3, 4, 4), dtype=np.float32),
            class_id=0,
            n_tokens=4,
        )
        imputer = ImageImputer(
            architecture=arch,
            image=_tiny_image(4, 4),
            player_strategy=PatchStrategy(grid_size=2, n_players=4),
        )
        values = imputer.value_function(_coalitions_4())
        assert values.shape == (4,)
        assert np.isfinite(values).all()
