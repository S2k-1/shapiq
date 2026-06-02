"""Tests for ``shapiq.vision.utils``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from shapiq.vision.utils import as_hwc_array


class TestAsHwcArray:
    def test_numpy_hwc_passthrough(self) -> None:
        image = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
        out = as_hwc_array(image)
        np.testing.assert_array_equal(out, image)

    def test_numpy_grayscale_adds_channel(self) -> None:
        image = np.arange(4, dtype=np.float64).reshape(2, 2)
        out = as_hwc_array(image)
        assert out.shape == (2, 2, 1)
        np.testing.assert_array_equal(out[..., 0], image)

    def test_pil_rgb_image(self) -> None:
        arr = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        pil = Image.fromarray(arr, mode="RGB")
        out = as_hwc_array(pil)
        np.testing.assert_array_equal(out, arr)

    def test_pil_grayscale_converted_to_rgb(self) -> None:
        gray = np.arange(4, dtype=np.uint8).reshape(2, 2)
        pil = Image.fromarray(gray, mode="L")
        out = as_hwc_array(pil)
        assert out.shape == (2, 2, 3)
        np.testing.assert_array_equal(out[..., 0], gray)
        np.testing.assert_array_equal(out[..., 1], gray)
        np.testing.assert_array_equal(out[..., 2], gray)

    def test_torch_chw_tensor(self) -> None:
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        out = as_hwc_array(tensor)
        expected = tensor.permute(1, 2, 0).numpy()
        np.testing.assert_array_equal(out, expected)

    def test_torch_batched_chw_tensor(self) -> None:
        tensor = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
        out = as_hwc_array(tensor)
        expected = tensor.squeeze(0).permute(1, 2, 0).numpy()
        np.testing.assert_array_equal(out, expected)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="image must be"):
            as_hwc_array({"not": "an image"})
