"""Tests for ``shapiq.vision.utils``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from shapiq.vision.architecture import ResNetArchitecture, ViTArchitecture
from shapiq.vision.masking import BoolMaskedPosStrategy, ZeroMasking
from shapiq.vision.utils import (
    as_hwc_array,
    get_torch_device,
    infer_default_batch_size,
    is_image_like,
    resolve_batch_size,
    tensor_to_numpy,
)


class TestIsImageLike:
    def test_rgb_numpy_array(self) -> None:
        assert is_image_like(np.zeros((8, 8, 3)))

    def test_grayscale_square_array(self) -> None:
        assert is_image_like(np.zeros((16, 16)))

    def test_tabular_background_matrix(self) -> None:
        assert not is_image_like(np.zeros((100, 20)))

    def test_pil_image(self) -> None:
        arr = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        assert is_image_like(Image.fromarray(arr, mode="RGB"))

    def test_torch_tensor(self) -> None:
        assert is_image_like(torch.zeros(3, 8, 8))

    def test_none_is_false(self) -> None:
        assert not is_image_like(None)


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


class TestTorchDeviceHelpers:
    def test_get_torch_device_from_tensor(self) -> None:
        tensor = torch.zeros(2, 3)
        assert get_torch_device(tensor) == torch.device("cpu")

    def test_get_torch_device_from_module(self) -> None:
        module = torch.nn.Linear(2, 1)
        assert get_torch_device(module) == torch.device("cpu")

    def test_tensor_to_numpy_from_cpu(self) -> None:
        tensor = torch.tensor([1.0, 2.0])
        np.testing.assert_array_equal(tensor_to_numpy(tensor), np.array([1.0, 2.0]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_torch_device_from_cuda_module(self) -> None:
        module = torch.nn.Linear(2, 1).cuda()
        assert get_torch_device(module).type == "cuda"


class TestBatchSizeInference:
    def test_cpu_pixel_path_returns_small_batch(self) -> None:
        image = np.zeros((8, 8, 3))
        arch = ResNetArchitecture(
            model=lambda x: np.zeros(x.shape[0]), masking_strategy=ZeroMasking()
        )
        batch_size = infer_default_batch_size(arch, image, n_players=4)
        assert batch_size == 2

    def test_cpu_latent_path_returns_larger_batch(self) -> None:
        image = np.zeros((16, 16, 3))
        arch = ViTArchitecture(
            model=object(),
            processor=object(),
            masking_strategy=BoolMaskedPosStrategy(),
        )
        batch_size = infer_default_batch_size(arch, image, n_players=4)
        assert batch_size == 4

    def test_resolve_auto_returns_int(self) -> None:
        image = np.zeros((8, 8, 3))
        arch = ResNetArchitecture(
            model=lambda x: np.zeros(x.shape[0]), masking_strategy=ZeroMasking()
        )
        assert isinstance(resolve_batch_size("auto", arch, image, 4), int)

    def test_resolve_none_keeps_unbatched(self) -> None:
        image = np.zeros((8, 8, 3))
        arch = ResNetArchitecture(
            model=lambda x: np.zeros(x.shape[0]), masking_strategy=ZeroMasking()
        )
        assert resolve_batch_size(None, arch, image, 4) is None
