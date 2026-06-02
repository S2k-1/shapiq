"""Tests for the ``batch_size`` parameter on ImageImputer and ImageExplainer.

The contract for batching:
- ``batch_size=None`` and ``batch_size >= n_coalitions`` are equivalent (one model call).
- For ``batch_size < n_coalitions``, the architecture is called ``ceil(n / batch_size)`` times.
- Output values are bit-for-bit identical regardless of batch size.
"""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.vision import ImageExplainer
from shapiq.vision.architecture import ResNetArchitecture
from shapiq.vision.imputer import ImageImputer
from shapiq.vision.masking import ZeroMasking

from .conftest import FixedMasksStrategy, make_linear_pixel_model


class _CountingResNet(ResNetArchitecture):
    """ResNetArchitecture that counts ``value_function`` calls and the per-call batch size."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.call_batch_sizes: list[int] = []

    def value_function(self, image, coalitions):
        self.call_batch_sizes.append(int(coalitions.shape[0]))
        return super().value_function(image, coalitions)


def _build(image, masks, *, batch_size, weights=None):
    weights = weights if weights is not None else np.ones((4, 4))
    arch = _CountingResNet(model=make_linear_pixel_model(weights), masking_strategy=ZeroMasking())
    imputer = ImageImputer(
        architecture=arch,
        image=image,
        player_strategy=FixedMasksStrategy(masks),
        batch_size=batch_size,
        normalize=False,
    )
    return imputer, arch


@pytest.fixture
def setup_image_and_masks():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(4, 4, 3)).astype(np.float64)
    masks = np.zeros((4, 4, 4), dtype=bool)
    masks[0, :2, :2] = True
    masks[1, :2, 2:] = True
    masks[2, 2:, :2] = True
    masks[3, 2:, 2:] = True
    return image, masks


def _all_coalitions(n: int) -> np.ndarray:
    return np.array([[(i >> j) & 1 == 1 for j in range(n)] for i in range(1 << n)], dtype=bool)


class TestBatchingNumerics:
    def test_output_identical_to_non_batched(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        coalitions = _all_coalitions(4)

        imp_none, _ = _build(image, masks, batch_size=None)
        imp_3, _ = _build(image, masks, batch_size=3)
        imp_5, _ = _build(image, masks, batch_size=5)
        imp_1, _ = _build(image, masks, batch_size=1)

        v_none = imp_none.value_function(coalitions)
        v_3 = imp_3.value_function(coalitions)
        v_5 = imp_5.value_function(coalitions)
        v_1 = imp_1.value_function(coalitions)

        np.testing.assert_array_equal(v_none, v_3)
        np.testing.assert_array_equal(v_none, v_5)
        np.testing.assert_array_equal(v_none, v_1)

    def test_batch_size_larger_than_n_is_one_call(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        coalitions = _all_coalitions(4)  # 16 coalitions
        imp, arch = _build(image, masks, batch_size=100)
        # Reset counter (the constructor call for empty_prediction also bumps it).
        arch.call_batch_sizes.clear()
        imp.value_function(coalitions)
        assert arch.call_batch_sizes == [16]

    def test_chunking_call_pattern(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        coalitions = _all_coalitions(4)  # 16 coalitions
        imp, arch = _build(image, masks, batch_size=5)
        arch.call_batch_sizes.clear()
        imp.value_function(coalitions)
        # 16 split by 5 → chunks of 5, 5, 5, 1.
        assert arch.call_batch_sizes == [5, 5, 5, 1]

    def test_batch_size_one_splits_per_coalition(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        coalitions = _all_coalitions(4)
        imp, arch = _build(image, masks, batch_size=1)
        arch.call_batch_sizes.clear()
        imp.value_function(coalitions)
        assert arch.call_batch_sizes == [1] * 16

    def test_1d_coalition_still_works_under_batching(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        imp, _ = _build(image, masks, batch_size=2)
        out = imp.value_function(np.array([True, True, True, True]))
        # Linear model, full coalition with ZeroMasking → original image sum.
        assert out.shape == (1,)
        assert out[0] == pytest.approx(image.sum())


class TestBatchingPropagation:
    def test_explainer_passes_batch_size_to_imputer(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        arch = _CountingResNet(
            model=make_linear_pixel_model(np.ones((4, 4))),
            masking_strategy=ZeroMasking(),
        )
        explainer = ImageExplainer(
            architecture=arch,
            data=image,
            player_strategy=FixedMasksStrategy(masks),
            batch_size=4,
            random_state=0,
        )
        assert explainer.imputer.batch_size == 4

    def test_default_batch_size_is_none(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        arch = _CountingResNet(
            model=make_linear_pixel_model(np.ones((4, 4))),
            masking_strategy=ZeroMasking(),
        )
        imputer = ImageImputer(
            architecture=arch, image=image, player_strategy=FixedMasksStrategy(masks)
        )
        assert imputer.batch_size is None

    def test_explainer_end_to_end_with_batching(self, setup_image_and_masks) -> None:
        image, masks = setup_image_and_masks
        arch = _CountingResNet(
            model=make_linear_pixel_model(np.ones((4, 4))),
            masking_strategy=ZeroMasking(),
        )
        explainer = ImageExplainer(
            architecture=arch,
            data=image,
            player_strategy=FixedMasksStrategy(masks),
            batch_size=3,
            max_order=2,
            random_state=0,
        )
        result = explainer.explain_function(image, budget=16)
        # Sanity check: explainer ran and returned a non-empty result.
        assert result.n_players == 4
        # And — crucially — the architecture saw multiple sub-batch calls.
        # (The exact count depends on the approximator's sampling, so just check >1.)
        assert sum(b > 0 for b in arch.call_batch_sizes) > 1
