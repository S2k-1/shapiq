"""Automatic architecture dispatch for vision models.

:func:`resolve_architecture` inspects a raw model and returns a configured
:class:`~shapiq.vision.architecture.ModelArchitectureStrategy` so users can
pass a model directly to :class:`~shapiq.vision.explainer.ImageExplainer`
without choosing an architecture themselves.

The dispatch is deliberately wide: a model is treated as *ViT-like* only when
token masking via ``bool_masked_pos`` demonstrably changes its output
(verified with :func:`~shapiq.vision.probing.probe_token_masking` — many
``transformers`` classification heads accept the argument via ``**kwargs``
and silently ignore it). Every other model falls back to pixel-space masking,
i.e. the *CNN-like* path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .probing import probe_token_masking, search_config, validate_classification_model

if TYPE_CHECKING:
    from shapiq.typing import Model

    from .architecture import ModelArchitectureStrategy


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
    patch_size = search_config(config, "patch_size")
    if patch_size is None:
        return None
    input_size = None
    if processor is not None:
        input_size = _processor_input_size(processor)
    if input_size is None:
        input_size = search_config(config, "image_size")
    if input_size is None or input_size < patch_size:
        return None
    return input_size // patch_size


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
       responds to ``bool_masked_pos`` (see
       :func:`~shapiq.vision.probing.probe_token_masking`) get a
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
        # The probe just verified token masking; skip the constructor's own check.
        return TransformerArchitecture(model=model, processor=processor, verified=True)

    validate_classification_model(model, processor)
    return CNNArchitecture(model=model, processor=processor)
