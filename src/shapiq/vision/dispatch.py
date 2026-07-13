"""Automatic architecture dispatch for vision models.

:func:`resolve_architecture` inspects a raw model and returns a configured
:class:`~shapiq.vision.architecture.ModelArchitectureStrategy` so users can
pass a model directly to :class:`~shapiq.vision.explainer.ImageExplainer`
without choosing an architecture themselves.

The dispatch is deliberately wide: a model is treated as *ViT-like* only when
token masking via ``bool_masked_pos`` demonstrably changes its output
(verified with a functional probe — many ``transformers`` classification
heads accept the argument via ``**kwargs`` and silently ignore it). Every
other model falls back to pixel-space masking, i.e. the *CNN-like* path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .utils import get_torch_device

try:
    import torch
except ImportError as err:
    from ._error import _vision_import_error

    raise _vision_import_error from err

if TYPE_CHECKING:
    from collections.abc import Iterator

    from shapiq.typing import Model

    from .architecture import ModelArchitectureStrategy


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


def _search_config(config: object, *keys: str) -> int | None:
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


def _processor_input_size(processor: Model) -> int | None:
    """Return the spatial size a processor resizes/crops images to, if declared.

    Prefers ``crop_size`` (the final size after center-cropping) over ``size``.
    Both attributes may be ints or dicts with ``height``/``shortest_edge`` keys.
    """
    for attr in ("crop_size", "size"):
        value = getattr(processor, attr, None)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, dict):
            for key in ("height", "shortest_edge", "width"):
                side = value.get(key)
                if isinstance(side, int) and side > 0:
                    return side
    return None


def resolve_patch_grid(model: Model, processor: Model | None = None) -> int | None:
    """Resolve the token-grid side length of a ViT-like model.

    The patch size is searched in the model config and its sub-configs (so
    nested layouts like CLIP's ``config.vision_config.patch_size`` work). The
    input size preferably comes from the processor, since checkpoints such as
    DINOv2 declare ``config.image_size`` values the processor never produces.

    Returns:
        ``input_size // patch_size``, or ``None`` when either quantity cannot
        be determined.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None
    patch_size = _search_config(config, "patch_size")
    if patch_size is None:
        return None
    input_size = None
    if processor is not None:
        input_size = _processor_input_size(processor)
    if input_size is None:
        input_size = _search_config(config, "image_size")
    if input_size is None or input_size < patch_size:
        return None
    return input_size // patch_size


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
        return _search_config(config, "hidden_size", "embed_dim")
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
    dtype = torch.float32
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            dtype = next(parameters()).dtype
        except StopIteration:
            pass
    embeddings.mask_token = torch.nn.Parameter(
        torch.zeros(1, 1, dim, device=device, dtype=dtype), requires_grad=False
    )
    return True


def extract_logits(output: object) -> torch.Tensor:
    """Return classification logits from a model output.

    Accepts raw tensors (torchvision-style) and output objects exposing
    ``.logits`` (``transformers``-style).

    Raises:
        TypeError: If the output carries neither, e.g. encoder-only models
            such as ViT-MAE or bare CLIP without a classification head.
    """
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, torch.Tensor):
        logits = output
    if not isinstance(logits, torch.Tensor):
        msg = (
            f"Model output of type {type(output).__name__} exposes no classification logits. "
            "Only classification models (returning a tensor or an object with `.logits`) "
            "are supported by ImageExplainer."
        )
        raise TypeError(msg)
    return logits


def _processed_dummy(processor: Model, device: torch.device) -> torch.Tensor:
    """Run a small random image through the processor and return ``pixel_values``."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(device)


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
        pixel_values = _processed_dummy(processor, device)
        if not ensure_zero_mask_token(model):
            return False
        was_training = bool(getattr(model, "training", False))
        if hasattr(model, "eval"):
            model.eval()
        try:
            with torch.no_grad():
                unmasked = extract_logits(model(pixel_values=pixel_values))
                mask = torch.ones(1, grid_size * grid_size, dtype=torch.bool, device=device)
                masked = extract_logits(model(pixel_values=pixel_values, bool_masked_pos=mask))
        finally:
            if was_training and hasattr(model, "train"):
                model.train()
        if unmasked.shape != masked.shape:
            return False
        return not torch.allclose(unmasked, masked)
    except Exception:  # noqa: BLE001 - any incompatibility means "not ViT-like"
        return False


def _load_processor(model: Model) -> Model | None:
    """Try to auto-load the matching ``AutoImageProcessor`` for a model."""
    name = getattr(model, "name_or_path", None) or getattr(
        getattr(model, "config", None), "_name_or_path", None
    )
    if not name:
        return None
    try:
        from transformers import AutoImageProcessor

        return AutoImageProcessor.from_pretrained(name)
    except Exception:  # noqa: BLE001 - fall back to explicit processor argument
        return None


def _validate_classification_model(model: Model, processor: Model) -> None:
    """Raise a clear error when a Hugging Face model yields no logits.

    Catches encoder-only models (ViT-MAE, bare CLIP, feature extractors)
    at dispatch time instead of failing deep inside the value function.
    """
    try:
        pixel_values = _processed_dummy(processor, get_torch_device(model))
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


def resolve_architecture(
    model: Model | ModelArchitectureStrategy,
    processor: Model | None = None,
) -> ModelArchitectureStrategy:
    """Dispatch a raw vision model to a fitting architecture strategy.

    Dispatch rules:

    1. An existing :class:`~shapiq.vision.architecture.ModelArchitectureStrategy`
       is returned unchanged.
    2. Models without a ``config`` attribute (torchvision CNNs, custom
       modules) get a plain :class:`~shapiq.vision.architecture.CNNArchitecture`.
    3. Hugging Face-style models (with ``config``) whose output verifiably
       responds to ``bool_masked_pos`` (see :func:`probe_token_masking`) get a
       :class:`~shapiq.vision.architecture.TransformerArchitecture` with
       token-space masking.
    4. Every other Hugging Face-style model falls back to
       :class:`~shapiq.vision.architecture.CNNArchitecture` with pixel-space
       masking, preprocessing each masked image through the processor.

    Args:
        model: The model to dispatch, or an already-configured architecture.
        processor: Optional matching image processor. When omitted, it is
            auto-loaded via ``AutoImageProcessor.from_pretrained`` using the
            model's ``name_or_path``.

    Returns:
        A configured architecture strategy wrapping ``model``.

    Raises:
        TypeError: If a Hugging Face-style model has no loadable processor or
            does not produce classification logits.
    """
    from .architecture import CNNArchitecture, ModelArchitectureStrategy, TransformerArchitecture

    # Duck-typed on top of isinstance so architectures survive module reloads
    # (isinstance is identity-based and breaks across importlib.reload). The
    # attributes are checked on the type because some are properties that
    # raise before ``prepare`` was called.
    if isinstance(model, ModelArchitectureStrategy) or all(
        hasattr(type(model), attr)
        for attr in ("prepare", "value_function", "n_players", "player_masks")
    ):
        return model

    if not hasattr(model, "config"):
        return CNNArchitecture(model=model)

    processor = processor or _load_processor(model)
    if processor is None:
        msg = (
            f"{type(model).__name__} looks like a Hugging Face model but no matching image "
            "processor could be loaded automatically. Pass one explicitly, e.g. "
            "ImageExplainer(model=model, processor=AutoImageProcessor.from_pretrained(...), ...)."
        )
        raise TypeError(msg)

    grid_size = resolve_patch_grid(model, processor)
    if grid_size is not None and probe_token_masking(model, processor, grid_size):
        return TransformerArchitecture(model=model, processor=processor)

    _validate_classification_model(model, processor)
    return CNNArchitecture(model=model, processor=processor)
