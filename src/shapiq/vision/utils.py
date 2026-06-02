"""Utilities for the vision explanation package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image as PILImage

    ImageLike = np.ndarray | PILImage | torch.Tensor
else:
    ImageLike = np.ndarray

__all__ = ["ImageLike", "as_hwc_array", "is_image_like"]


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
