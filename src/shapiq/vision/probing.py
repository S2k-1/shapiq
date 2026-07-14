"""Model reflection and functional probing helpers for vision models.

Static information (protocols, signatures, config attributes) cannot tell
whether a model *actually* honors ``bool_masked_pos``: recent ``transformers``
classification heads accept it via ``**kwargs`` and some families (BEiT, Swin,
FocalNet) silently drop it. The helpers in this module answer such questions
functionally — by inspecting the model object or running forward passes on a
dummy image. They are used by :mod:`~shapiq.vision.dispatch` to *decide* on an
architecture and by the architectures themselves to *verify* a configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .utils import extract_logits, get_torch_device

try:
    import torch
except ImportError as err:
    from ._error import _vision_import_error

    raise _vision_import_error from err

if TYPE_CHECKING:
    from collections.abc import Iterator

    from shapiq.typing import Model


def base_module_of(model: Model) -> Model:
    """Return the backbone of a Hugging Face model, or the model itself.

    Hugging Face task heads expose their backbone as ``base_model`` (e.g.
    ``model.vit``, ``model.deit``, ``model.beit``, ``model.swin``). Plain
    modules are returned unchanged.
    """
    base = getattr(model, "base_model", None)
    return base if base is not None else model


def embeddings_of(model: Model) -> Model | None:
    """Return the embeddings module of a Hugging Face-style model, if any."""
    return getattr(base_module_of(model), "embeddings", None)


def _iter_configs(config: object) -> Iterator[object]:
    """Yield ``config`` followed by nested sub-configs (e.g. ``vision_config``)."""
    yield config
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        yield vision_config
    for value in vars(config).values() if hasattr(config, "__dict__") else ():
        if value is not vision_config and callable(getattr(value, "to_dict", None)):
            yield value


def search_config(config: object, *keys: str) -> int | None:
    """Search ``config`` and its sub-configs for the first positive-int value of ``keys``.

    Tuple/list values (e.g. ``image_size=(224, 224)``) resolve to their first
    element.
    """
    for cfg in _iter_configs(config):
        for key in keys:
            value = getattr(cfg, key, None)
            if isinstance(value, tuple | list) and value:
                value = value[0]
            if isinstance(value, int) and value > 0:
                return value
    return None


def resolve_embed_dim(model: Model) -> int | None:
    """Infer the token embedding dimension of a Hugging Face-style model.

    Tries, in order: the shape of an existing ``mask_token``, the output
    channels of the patch-embedding projection, and the config keys
    ``hidden_size`` / ``embed_dim``. This avoids the hard
    ``config.hidden_size`` requirement that models like FocalNet or
    MobileViT do not satisfy.
    """
    embeddings = embeddings_of(model)
    mask_token = getattr(embeddings, "mask_token", None)
    if isinstance(mask_token, torch.Tensor):
        return int(mask_token.shape[-1])
    projection = getattr(getattr(embeddings, "patch_embeddings", None), "projection", None)
    out_channels = getattr(projection, "out_channels", None)
    if isinstance(out_channels, int) and out_channels > 0:
        return out_channels
    config = getattr(model, "config", None)
    if config is not None:
        return search_config(config, "hidden_size", "embed_dim")
    return None


def ensure_zero_mask_token(model: Model) -> bool:
    """Make sure the model's embeddings carry an all-zero ``mask_token``.

    Classification checkpoints are usually instantiated with
    ``use_mask_token=False`` (``mask_token is None``), which crashes the
    ``bool_masked_pos`` path inside ``transformers``. A zero mask token makes
    masked tokens carry no signal, which is exactly the imputation wanted for
    Shapley evaluation — existing (learned) mask tokens are zeroed for the
    same reason.

    Returns:
        ``True`` if a zero mask token is in place, ``False`` when the model
        has no embeddings with a ``mask_token`` slot or the embedding
        dimension cannot be inferred.
    """
    embeddings = embeddings_of(model)
    if embeddings is None or not hasattr(embeddings, "mask_token"):
        return False
    mask_token = embeddings.mask_token
    if isinstance(mask_token, torch.Tensor):
        with torch.no_grad():
            mask_token.zero_()
        return True
    dim = resolve_embed_dim(model)
    if dim is None:
        return False
    device = get_torch_device(model)
    parameters = getattr(model, "parameters", None)
    first_param = next(parameters(), None) if callable(parameters) else None
    dtype = first_param.dtype if first_param is not None else torch.float32
    embeddings.mask_token = torch.nn.Parameter(
        torch.zeros(1, 1, dim, device=device, dtype=dtype), requires_grad=False
    )
    return True


def processed_dummy(processor: Model, device: torch.device) -> torch.Tensor:
    """Run a small random image through the processor and return ``pixel_values``."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(device)


def masking_changes_output(
    model: Model, pixel_values: torch.Tensor, token_mask: torch.Tensor
) -> bool:
    """Return ``True`` when masking all tokens changes the model output.

    The shared core of :func:`probe_token_masking` and the construction-time
    verification in
    :class:`~shapiq.vision.architecture.TransformerArchitecture`: forwards
    ``pixel_values`` once without and once with ``bool_masked_pos=token_mask``
    and compares the logits.
    """
    with torch.no_grad():
        unmasked = extract_logits(model(pixel_values=pixel_values))
        masked = extract_logits(model(pixel_values=pixel_values, bool_masked_pos=token_mask))
    return unmasked.shape == masked.shape and not torch.allclose(unmasked, masked)


def probe_token_masking(model: Model, processor: Model, grid_size: int) -> bool:
    """Check functionally whether ``bool_masked_pos`` affects the model output.

    Signatures cannot answer this: in recent ``transformers`` versions most
    classification heads accept arbitrary ``**kwargs``, and families like
    BEiT, Swin, or FocalNet silently drop ``bool_masked_pos`` — which would
    make every coalition evaluate to the same prediction. The probe compares
    an unmasked forward pass against a fully masked one (with a zeroed mask
    token in place) and only reports ``True`` when the logits differ.

    Any failure along the way (processor errors, unsupported argument, wrong
    token count, missing logits) counts as "not token-maskable".
    """
    try:
        device = get_torch_device(model)
        pixel_values = processed_dummy(processor, device)
        if not ensure_zero_mask_token(model):
            return False
        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        try:
            mask = torch.ones(1, grid_size * grid_size, dtype=torch.bool, device=device)
            return masking_changes_output(model, pixel_values, mask)
        finally:
            if was_training and hasattr(model, "train"):
                model.train()
    except Exception:  # noqa: BLE001 - any incompatibility means "not ViT-like"
        return False


def validate_classification_model(model: Model, processor: Model) -> None:
    """Raise a clear error when a Hugging Face model yields no logits.

    Catches encoder-only models (ViT-MAE, bare CLIP, feature extractors)
    at dispatch time instead of failing deep inside the value function.
    """
    try:
        pixel_values = processed_dummy(processor, get_torch_device(model))
        with torch.no_grad():
            output = model(pixel_values=pixel_values)
    except Exception as err:
        msg = (
            f"Could not run {type(model).__name__} on processed pixel values: {err} "
            "ImageExplainer requires an image classification model that accepts "
            "`pixel_values` and returns logits."
        )
        raise TypeError(msg) from err
    extract_logits(output)
