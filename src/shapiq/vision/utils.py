"""Utilities for the vision explanation package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image as PILImage

    from .architecture import ModelArchitectureStrategy

    ImageLike = np.ndarray | PILImage | torch.Tensor
    AutoBatchSize = int | Literal["auto"] | None
else:
    ImageLike = np.ndarray
    AutoBatchSize = int | str | None

__all__ = [
    "AutoBatchSize",
    "ImageLike",
    "as_hwc_array",
    "get_torch_device",
    "infer_default_batch_size",
    "is_image_like",
    "resolve_batch_size",
    "tensor_to_numpy",
]


def is_image_like(data: object) -> bool:
    """Return whether ``data`` looks like a single image rather than tabular background data.

    Used by :class:`~shapiq.explainer.base.Explainer` to auto-dispatch to
    :class:`~shapiq.vision.explainer.ImageExplainer`.

    Args:
        data: Candidate image input (numpy array, PIL image, or PyTorch tensor).

    Returns:
        ``True`` if ``data`` is a PIL image, a PyTorch image tensor, a ``(H, W, C)``
        numpy array, or a ``(H, W)`` grayscale array that is unlikely to be a tabular
        background matrix.
    """
    if data is None:
        return False
    if _try_convert_pil_image(data) is not None:
        return True
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(data, torch.Tensor):
        return data.ndim in (2, 3, 4)
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            return True
        if data.ndim == 2:
            rows, cols = data.shape
            if rows == cols:
                return True
            smaller, larger = sorted((rows, cols))
            if smaller <= 64 and larger / smaller > 2:
                return False
            return True
    return False


def as_hwc_array(image: ImageLike) -> np.ndarray:
    """Convert an image to a ``(H, W, C)`` numpy array.

    Accepts numpy arrays (``(H, W, C)`` or ``(H, W)``), PIL images, and PyTorch
    tensors (``(C, H, W)``, ``(H, W, C)``, or ``(1, C, H, W)``). Other array-like
    objects are coerced with :func:`numpy.asarray` when possible.

    Args:
        image: Input image in a supported format.

    Returns:
        A numpy array with shape ``(H, W, C)``.

    Raises:
        TypeError: If the input type is not supported.
        ValueError: If the resulting array does not have two or three dimensions.
    """
    if isinstance(image, np.ndarray):
        arr = np.asarray(image)
    else:
        pil_image = _try_convert_pil_image(image)
        if pil_image is not None:
            arr = np.asarray(pil_image)
        else:
            tensor = _try_convert_torch_tensor(image)
            if tensor is not None:
                arr = tensor
            else:
                try:
                    arr = np.asarray(image)
                except (TypeError, ValueError) as exc:
                    msg = (
                        "image must be a numpy array, PIL Image, or PyTorch tensor; "
                        f"got {type(image)!r}"
                    )
                    raise TypeError(msg) from exc

    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
        if (
            arr.ndim == 3
            and arr.shape[0] in (1, 3, 4)
            and arr.shape[0]
            not in (
                arr.shape[1],
                arr.shape[2],
            )
        ) or (arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4)):
            arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 0 or arr.dtype == object:
        msg = f"image must be a numpy array, PIL Image, or PyTorch tensor; got {type(image)!r}"
        raise TypeError(msg)
    if arr.ndim != 3:
        msg = f"Expected image with 2 or 3 dimensions after conversion, got shape {arr.shape}"
        raise ValueError(msg)
    return arr


def _try_convert_pil_image(image: object) -> np.ndarray | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    return None


def _try_convert_torch_tensor(image: object) -> np.ndarray | None:
    try:
        import torch
    except ImportError:
        return None
    if not isinstance(image, torch.Tensor):
        return None

    tensor = image.detach().cpu()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 2:
        return tensor.numpy()[..., np.newaxis]
    if tensor.ndim == 3:
        if tensor.shape[-1] in (1, 3, 4) and tensor.shape[0] not in (1, 3, 4):
            pass
        elif tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        return tensor.numpy()
    msg = f"Expected PyTorch tensor with 2, 3, or 4 dimensions, got shape {tuple(tensor.shape)}"
    raise ValueError(msg)


def get_torch_device(obj: object) -> torch.device:
    """Return the ``torch.device`` for a model, module, or tensor.

    Inspects ``.parameters()`` and ``.buffers()`` when present. Falls back to CPU
    when PyTorch is unavailable or no tensors are found.
    """
    try:
        import torch
    except ImportError as exc:
        msg = "PyTorch is required to resolve a torch device"
        raise ImportError(msg) from exc

    if isinstance(obj, torch.Tensor):
        return obj.device

    for accessor in (getattr(obj, "parameters", None), getattr(obj, "buffers", None)):
        if callable(accessor):
            try:
                return next(accessor()).device
            except StopIteration:
                continue

    return torch.device("cpu")


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a numpy array, copying from GPU when needed."""
    return tensor.detach().cpu().numpy()


def infer_default_batch_size(
    architecture: ModelArchitectureStrategy,
    image: np.ndarray,
    n_players: int,
) -> int:
    """Pick a conservative coalition batch size from image size, players, and hardware.

    Heuristics follow the guidance in ``supplementary/batching_details.ipynb``:
    small batches on CPU, moderate batches on consumer GPUs, and larger batches for
    latent-space ViT paths where each coalition reuses cached pixel values.
    """
    from .masking import LatentMaskingStrategy

    is_latent = isinstance(architecture.masking_strategy, LatentMaskingStrategy)
    pixel_count = int(image.shape[0] * image.shape[1])

    cuda_available = False
    vram_gb = 0.0
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = get_torch_device(architecture.model)
            device_index = device.index if device.index is not None else 0
            if device.type != "cuda":
                device_index = 0
            vram_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
    except ImportError:
        pass

    if not cuda_available:
        return 4 if is_latent else 2

    if is_latent:
        if n_players >= 196:
            return 16
        if n_players >= 64:
            return 32
        return 64 if vram_gb >= 24 else 32

    if pixel_count <= 32 * 32:
        return 64 if vram_gb >= 16 else 16
    if pixel_count <= 224 * 224:
        if vram_gb >= 24:
            return 128
        if vram_gb >= 12:
            return 32
        return 16
    return 8 if vram_gb >= 12 else 4


def resolve_batch_size(
    batch_size: AutoBatchSize,
    architecture: ModelArchitectureStrategy,
    image: np.ndarray,
    n_players: int,
) -> int | None:
    """Resolve ``batch_size`` after ``"auto"`` selection.

    ``"auto"`` (the default) picks a hardware-aware batch size. ``None`` keeps the
    legacy behaviour of evaluating all coalitions in one forward pass.
    """
    if batch_size == "auto":
        return infer_default_batch_size(architecture, image, n_players)
    return batch_size
