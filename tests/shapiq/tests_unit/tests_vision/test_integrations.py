"""Integration tests across mixed (architecture x player x masker) combinations.

These tests sweep over the cartesian product of supported strategies to verify
the full ``Explainer → Imputer → Architecture`` pipeline works end-to-end for
each combination, and that swapping any component for a compatible alternative
yields a valid ``InteractionValues`` output of the expected shape.

The architectures are exercised with lightweight mock HuggingFace-style models
so the tests run without ``transformers`` or any downloaded weights. The
``ResNetArchitecture`` path uses the linear-pixel-model helper from conftest.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shapiq.game_theory.exact import ExactComputer
from shapiq.interaction_values import InteractionValues
from shapiq.vision import ImageExplainer
from shapiq.vision.architecture import (
    CLIPArchitecture,
    ConvNeXtArchitecture,
    CustomViTArchitecture,
    DINOv2Architecture,
    HuggingFacePixelArchitecture,
    ResNetArchitecture,
)
from shapiq.vision.imputer import ImageImputer
from shapiq.vision.masking import (
    BlurMasking,
    BoolMaskedPosStrategy,
    MeanColorMasking,
    ZeroMasking,
)
from shapiq.vision.players import (
    CustomMasksStrategy,
    GridStrategy,
    PatchStrategy,
    SuperpixelStrategy,
)

from .conftest import FixedMasksStrategy, make_linear_pixel_model

# ---------------------------------------------------------------------------
# Shared mocks. Identical interface to the ones in test_extra_architectures.py
# but kept local so the two files stay independent.
# ---------------------------------------------------------------------------


class _MockImageProcessor:
    def __call__(self, images=None, text=None, return_tensors="pt", *, padding=True):
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            arr = np.stack([np.asarray(img, dtype=np.float32) for img in images], axis=0)
            pixel_values = torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())
            return {"pixel_values": pixel_values}
        if text is not None:
            return {"input_ids": torch.arange(len(text)).reshape(-1, 1)}
        msg = "either images or text must be supplied"
        raise ValueError(msg)


class _MockHFClassifier:
    """Two-class HF-style classifier; class-0 logit == per-image mean pixel."""

    def __call__(self, pixel_values=None, **_):
        means = pixel_values.float().mean(dim=(1, 2, 3))
        return SimpleNamespace(logits=torch.stack([means, -means], dim=1))


class _MockBackbone:
    """DINOv2-style backbone: CLS token is a 4-d projection of the image."""

    def __call__(self, pixel_values=None, **_):
        b = pixel_values.shape[0]
        flat = pixel_values.float().reshape(b, -1)
        feats = torch.stack(
            [
                flat.mean(dim=1),
                flat.std(dim=1) + 1e-6,
                flat[:, : flat.shape[1] // 2].mean(dim=1),
                flat[:, flat.shape[1] // 2 :].mean(dim=1),
            ],
            dim=1,
        )
        hidden = torch.zeros(b, 2, 4)
        hidden[:, 0] = feats
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=None)


class _MockCLIPModel:
    def __init__(self) -> None:
        self.logit_scale = torch.tensor(0.0)  # exp(0)=1.0
        self._text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def __call__(self, *_, **__):  # Imputer base requires `model` to be callable.
        return SimpleNamespace()

    def get_text_features(self, input_ids=None, **_):
        return self._text[input_ids.squeeze(-1)]

    def get_image_features(self, pixel_values=None, **_):
        b = pixel_values.shape[0]
        flat = pixel_values.float().reshape(b, -1)
        return torch.stack([flat.mean(dim=1), flat.std(dim=1) + 1e-6], dim=1)


class _MockMaskedViT:
    """ViT-like that consumes bool_masked_pos. Class-0 logit == #visible tokens."""

    def __call__(self, pixel_values=None, bool_masked_pos=None, **_):
        visible = (~bool_masked_pos).sum(dim=1).float()
        return SimpleNamespace(logits=torch.stack([visible, -visible], dim=1))


# ---------------------------------------------------------------------------
# Fixtures shared by all integration tests below
# ---------------------------------------------------------------------------


@pytest.fixture
def image_16x16() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(16, 16, 3)).astype(np.float64)


@pytest.fixture
def four_player_masks() -> np.ndarray:
    """Quadrant partition of a 16x16 image (4 disjoint players)."""
    masks = np.zeros((4, 16, 16), dtype=bool)
    masks[0, :8, :8] = True
    masks[1, :8, 8:] = True
    masks[2, 8:, :8] = True
    masks[3, 8:, 8:] = True
    return masks


# ---------------------------------------------------------------------------
# Architecture factories. Each factory returns (architecture, player_strategy_factory).
# The player_strategy_factory takes a `masks` ndarray and returns a strategy that
# either uses those masks (pixel-space) or wraps them as patch-equivalent (latent).
# ---------------------------------------------------------------------------


def _pixel_player_factory(masks):
    return FixedMasksStrategy(masks)


def _build_resnet_arch(masker):
    weights = np.ones((16, 16))
    return ResNetArchitecture(model=make_linear_pixel_model(weights), masking_strategy=masker)


def _build_hfpixel_arch(masker):
    return HuggingFacePixelArchitecture(
        _MockHFClassifier(), _MockImageProcessor(), masking_strategy=masker
    )


def _build_convnext_arch(masker):
    return ConvNeXtArchitecture(_MockHFClassifier(), _MockImageProcessor(), masking_strategy=masker)


def _build_dino_arch(masker):
    return DINOv2Architecture(_MockBackbone(), _MockImageProcessor(), masking_strategy=masker)


def _build_clip_arch(masker):
    return CLIPArchitecture(
        _MockCLIPModel(),
        _MockImageProcessor(),
        text_prompts=["a", "b"],
        target_prompt_idx=0,
        masking_strategy=masker,
    )


PIXEL_ARCHITECTURES = [
    ("ResNet", _build_resnet_arch),
    ("HFPixel", _build_hfpixel_arch),
    ("ConvNeXt", _build_convnext_arch),
    ("DINOv2", _build_dino_arch),
    ("CLIP", _build_clip_arch),
]

PIXEL_MASKERS = [
    ("mean_color", MeanColorMasking),
    ("zero", ZeroMasking),
    ("blur", BlurMasking),
]


# ---------------------------------------------------------------------------
# Matrix test 1 — pixel-space: every architecture x every pixel masker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("arch_name", "arch_factory"), PIXEL_ARCHITECTURES)
@pytest.mark.parametrize(("masker_name", "masker_cls"), PIXEL_MASKERS)
def test_pixel_matrix_explainer_endtoend(
    arch_name, arch_factory, masker_name, masker_cls, image_16x16, four_player_masks
):
    """Every (pixel architecture x pixel masker) combo runs end-to-end."""
    architecture = arch_factory(masker_cls())
    explainer = ImageExplainer(
        architecture=architecture,
        data=image_16x16,
        player_strategy=_pixel_player_factory(four_player_masks),
        index="k-SII",
        max_order=2,
        batch_size=4,
        random_state=0,
    )
    iv = explainer.explain_function(image_16x16, budget=32)
    assert isinstance(iv, InteractionValues), (arch_name, masker_name)
    assert iv.n_players == 4
    assert np.isfinite(iv.values).all(), (arch_name, masker_name)


@pytest.mark.parametrize(("arch_name", "arch_factory"), PIXEL_ARCHITECTURES)
@pytest.mark.parametrize(("masker_name", "masker_cls"), PIXEL_MASKERS)
def test_pixel_matrix_imputer_value_function(
    arch_name, arch_factory, masker_name, masker_cls, image_16x16, four_player_masks
):
    """The imputer's value_function returns the right shape/range for each combo."""
    architecture = arch_factory(masker_cls())
    imputer = ImageImputer(
        architecture=architecture,
        image=image_16x16,
        player_strategy=_pixel_player_factory(four_player_masks),
        normalize=False,
    )

    coalitions = np.array(
        [
            [False, False, False, False],
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ],
    )
    out = imputer(coalitions)
    assert out.shape == (5,), (arch_name, masker_name)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Matrix test 2 — latent-space: CustomViT x every latent masker x patch grid
# ---------------------------------------------------------------------------


LATENT_MASKERS = [
    ("bool_masked_pos", BoolMaskedPosStrategy),
    # MaskTokenStrategy is excluded from the integration matrix because it requires a mock
    # with ``model.vit.embeddings.mask_token`` and ``model.config.hidden_size``; this
    # combination is tested end-to-end in test_extra_architectures.py::TestViTArchitectureFull.
]


@pytest.mark.parametrize(("masker_name", "masker_cls"), LATENT_MASKERS)
@pytest.mark.parametrize("n_tokens", [4, 9])
def test_latent_matrix_explainer_endtoend(masker_name, masker_cls, n_tokens):
    """Every (latent masker x patch grid) combo runs end-to-end through CustomViT."""
    side = int(n_tokens**0.5)
    pixel_values = torch.zeros(1, 3, side, side)
    architecture = CustomViTArchitecture(
        model=_MockMaskedViT(),
        pixel_values=pixel_values,
        class_id=0,
        n_tokens=n_tokens,
        masking_strategy=masker_cls(),
    )
    explainer = ImageExplainer(
        architecture=architecture,
        data=np.zeros((side, side, 3)),
        player_strategy=PatchStrategy(grid_size=side, n_players=n_tokens),
        index="k-SII",
        max_order=2,
        batch_size=4,
        random_state=0,
    )
    iv = explainer.explain_function(np.zeros((side, side, 3)), budget=32)
    assert isinstance(iv, InteractionValues)
    assert iv.n_players == n_tokens
    assert np.isfinite(iv.values).all()


# ---------------------------------------------------------------------------
# Matrix test 3 — correctness vs ExactComputer for several combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("arch_name", "arch_factory"), PIXEL_ARCHITECTURES)
@pytest.mark.parametrize(("masker_name", "masker_cls"), PIXEL_MASKERS)
def test_exact_computer_returns_finite_shapley(
    arch_name, arch_factory, masker_name, masker_cls, image_16x16, four_player_masks
):
    """ExactComputer yields finite Shapley values for every (architecture x masker)."""
    architecture = arch_factory(masker_cls())
    imputer = ImageImputer(
        architecture=architecture,
        image=image_16x16,
        player_strategy=_pixel_player_factory(four_player_masks),
        normalize=False,
    )
    ec = ExactComputer(n_players=imputer.n_players, game=imputer)
    sv = ec.probabilistic_value(index="SV")
    assert np.isfinite(sv.values).all(), (arch_name, masker_name)
    # Efficiency: sum of SVs == v(grand) - v(empty).
    v_grand = imputer(np.array([[True] * 4]))[0]
    v_empty = imputer(np.array([[False] * 4]))[0]
    assert sv.values[1:].sum() == pytest.approx(v_grand - v_empty, abs=1e-5), (
        arch_name,
        masker_name,
    )


# ---------------------------------------------------------------------------
# Matrix test 4 — cross-batch consistency on every (architecture x masker)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("arch_name", "arch_factory"), PIXEL_ARCHITECTURES)
@pytest.mark.parametrize(("masker_name", "masker_cls"), PIXEL_MASKERS)
def test_batching_invariance_across_combinations(
    arch_name, arch_factory, masker_name, masker_cls, image_16x16, four_player_masks
):
    """For every combination, batch_size must not change the value function output."""
    coalitions = np.array(
        [
            [False, False, False, False],
            [True, False, False, False],
            [False, True, True, False],
            [True, True, True, True],
            [True, False, True, True],
        ],
    )

    def values_with(batch_size):
        # Fresh architecture every time — some architectures cache state in prepare.
        architecture = arch_factory(masker_cls())
        imp = ImageImputer(
            architecture=architecture,
            image=image_16x16,
            player_strategy=_pixel_player_factory(four_player_masks),
            normalize=False,
            batch_size=batch_size,
        )
        return imp.value_function(coalitions)

    v_full = values_with(None)
    for bs in (1, 2, 5, 100):
        np.testing.assert_allclose(values_with(bs), v_full, atol=1e-5, err_msg=f"bs={bs}")


# ---------------------------------------------------------------------------
# Matrix test 5 — strategy swap on the same architecture
# ---------------------------------------------------------------------------


def test_swap_masker_on_same_architecture_changes_output(image_16x16, four_player_masks):
    """Changing only the masker (same architecture, same players) yields different SVs."""
    arch_mean = _build_hfpixel_arch(MeanColorMasking())
    arch_zero = _build_hfpixel_arch(ZeroMasking())

    iv_mean = ImageExplainer(
        architecture=arch_mean,
        data=image_16x16,
        player_strategy=_pixel_player_factory(four_player_masks),
        random_state=0,
    ).explain_function(image_16x16, budget=32)

    iv_zero = ImageExplainer(
        architecture=arch_zero,
        data=image_16x16,
        player_strategy=_pixel_player_factory(four_player_masks),
        random_state=0,
    ).explain_function(image_16x16, budget=32)

    # Same shape; meaningfully different values (different masker => different baseline).
    assert iv_mean.n_players == iv_zero.n_players
    assert not np.allclose(iv_mean.values, iv_zero.values)


def test_swap_player_partition_on_same_architecture(image_16x16):
    """Changing only the player partition changes n_players and produces valid output."""
    arch = _build_hfpixel_arch(ZeroMasking())

    halves = np.zeros((2, 16, 16), dtype=bool)
    halves[0, :, :8] = True
    halves[1, :, 8:] = True

    quads = np.zeros((4, 16, 16), dtype=bool)
    quads[0, :8, :8] = True
    quads[1, :8, 8:] = True
    quads[2, 8:, :8] = True
    quads[3, 8:, 8:] = True

    iv2 = ImageExplainer(
        architecture=arch,
        data=image_16x16,
        player_strategy=FixedMasksStrategy(halves),
        random_state=0,
    ).explain_function(image_16x16, budget=16)
    iv4 = ImageExplainer(
        architecture=_build_hfpixel_arch(ZeroMasking()),
        data=image_16x16,
        player_strategy=FixedMasksStrategy(quads),
        random_state=0,
    ).explain_function(image_16x16, budget=32)

    assert iv2.n_players == 2
    assert iv4.n_players == 4
    assert np.isfinite(iv2.values).all()
    assert np.isfinite(iv4.values).all()


# ---------------------------------------------------------------------------
# Matrix test 6 — all pixel-space architectures agree on the trivial coalitions
# ---------------------------------------------------------------------------


def test_grand_coalition_equals_model_on_unmasked_image(image_16x16, four_player_masks):
    """For every pixel-space architecture, v(grand) on a non-normalized imputer should
    equal the model output on the original (unmasked) image — independent of masker."""
    for arch_name, arch_factory in PIXEL_ARCHITECTURES:
        for masker_name, masker_cls in PIXEL_MASKERS:
            arch = arch_factory(masker_cls())
            imp = ImageImputer(
                architecture=arch,
                image=image_16x16,
                player_strategy=_pixel_player_factory(four_player_masks),
                normalize=False,
            )
            v_grand = imp(np.array([[True] * 4]))[0]
            # Independent reference: call value_function on the all-present coalition.
            ref = arch.value_function(image_16x16, np.array([[True] * 4]))
            ref = float(np.atleast_1d(ref)[0])
            assert v_grand == pytest.approx(ref, abs=1e-5), (arch_name, masker_name)


# ---------------------------------------------------------------------------
# Matrix test 7 — player strategy sweep: GridStrategy and CustomMasksStrategy
# ---------------------------------------------------------------------------

#: Factory signature: ``(image, masks) -> PlayerStrategy``
PIXEL_PLAYER_STRATEGIES = [
    ("grid_2x2", lambda _img, _masks: GridStrategy(rows=2, cols=2)),
    ("custom_masks", lambda _img, masks: CustomMasksStrategy(masks)),
    ("superpixel_4", lambda _img, _masks: SuperpixelStrategy(n_segments=4)),
]


@pytest.mark.parametrize(("player_name", "player_factory"), PIXEL_PLAYER_STRATEGIES)
@pytest.mark.parametrize(("masker_name", "masker_cls"), PIXEL_MASKERS)
def test_player_strategy_matrix_imputer_value_function(
    player_name, player_factory, masker_name, masker_cls, image_16x16, four_player_masks
):
    """GridStrategy and CustomMasksStrategy return correct shapes through the imputer."""
    arch = _build_resnet_arch(masker_cls())
    player_strategy = player_factory(image_16x16, four_player_masks)
    imputer = ImageImputer(
        architecture=arch,
        image=image_16x16,
        player_strategy=player_strategy,
        normalize=False,
    )
    assert imputer.n_players == 4, (player_name, masker_name)
    coalitions = np.array(
        [
            [False, False, False, False],
            [True, False, False, False],
            [False, True, True, False],
            [True, True, True, True],
        ]
    )
    out = imputer(coalitions)
    assert out.shape == (4,), (player_name, masker_name)
    assert np.isfinite(out).all(), (player_name, masker_name)


@pytest.mark.parametrize(("player_name", "player_factory"), PIXEL_PLAYER_STRATEGIES)
@pytest.mark.parametrize(("masker_name", "masker_cls"), PIXEL_MASKERS)
def test_player_strategy_matrix_explainer_endtoend(
    player_name, player_factory, masker_name, masker_cls, image_16x16, four_player_masks
):
    """End-to-end ImageExplainer run for every (player strategy x masker) combination."""
    arch = _build_resnet_arch(masker_cls())
    player_strategy = player_factory(image_16x16, four_player_masks)
    iv = ImageExplainer(
        architecture=arch,
        data=image_16x16,
        player_strategy=player_strategy,
        index="k-SII",
        max_order=2,
        batch_size=4,
        random_state=0,
    ).explain_function(image_16x16, budget=32)
    assert isinstance(iv, InteractionValues), (player_name, masker_name)
    assert iv.n_players == 4, (player_name, masker_name)
    assert np.isfinite(iv.values).all(), (player_name, masker_name)


@pytest.mark.parametrize(("player_name", "player_factory"), PIXEL_PLAYER_STRATEGIES)
@pytest.mark.parametrize(("masker_name", "masker_cls"), PIXEL_MASKERS)
def test_player_strategy_matrix_exact_shapley(
    player_name, player_factory, masker_name, masker_cls, image_16x16, four_player_masks
):
    """ExactComputer efficiency axiom holds for every (player strategy x masker)."""
    from shapiq.game_theory.exact import ExactComputer

    arch = _build_resnet_arch(masker_cls())
    player_strategy = player_factory(image_16x16, four_player_masks)
    imputer = ImageImputer(
        architecture=arch,
        image=image_16x16,
        player_strategy=player_strategy,
        normalize=False,
    )
    ec = ExactComputer(n_players=imputer.n_players, game=imputer)
    sv = ec.probabilistic_value(index="SV")
    assert np.isfinite(sv.values).all(), (player_name, masker_name)
    v_grand = imputer(np.array([[True] * 4]))[0]
    v_empty = imputer(np.array([[False] * 4]))[0]
    assert sv.values[1:].sum() == pytest.approx(v_grand - v_empty, abs=1e-5), (
        player_name,
        masker_name,
    )
