"""Masking strategies for vision-based explanations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import torch

    from shapiq.typing import Model


class PixelMaskingStrategy(ABC):
    """Abstract base class for pixel-space masking strategies."""

    @abstractmethod
    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        """Apply the masking strategy to a batch of coalitions.

        Args:
            image:        ``(H, W, C)`` original image.
            player_masks: ``(n_players, H, W)`` boolean masks per player.
            coalition:    ``(n_coalitions, n_players)`` boolean array; ``True`` = player present.

        Returns:
            masked_images: ``(n_coalitions, H, W, C)`` array of masked images.
        """
        ...


class MeanColorMasking(PixelMaskingStrategy):
    """Imputes absent players' regions with the mean color of the entire image."""

    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        """Apply mean-color masking to a batch of coalitions."""
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape

        masked_images = np.stack([image] * n_coalitions, axis=0)

        mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    mask[i] |= player_masks[j]

        masked_images[mask] = image.mean(axis=(0, 1))
        return masked_images


class ZeroMasking(PixelMaskingStrategy):
    """Imputes absent players' regions with a constant fill value (default: ``0.0``)."""

    def __init__(self, value: float = 0.0) -> None:
        """Initialize the ZeroMasking strategy.

        Args:
            value: Fill value for absent regions. Defaults to ``0.0``.
        """
        self.value = value

    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        """Apply constant-value masking to a batch of coalitions."""
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape

        masked_images = np.stack([image] * n_coalitions, axis=0)

        mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    mask[i] |= player_masks[j]

        masked_images[mask] = self.value
        return masked_images


class BlurMasking(PixelMaskingStrategy):
    """Imputes absent players' regions with a Gaussian-blurred version of the image.

    Absent regions are replaced with pixels drawn from a Gaussian-blurred copy
    of the original image.  This provides a smooth, natural-looking baseline
    that retains global color statistics while removing fine-grained spatial
    detail from hidden regions — useful when the model is sensitive to hard
    color discontinuities at region boundaries.

    Args:
        sigma: Standard deviation of the Gaussian kernel in pixels.
            Defaults to ``10.0``.

    Example::

        masking = BlurMasking(sigma=8)
        imputer = ImageImputer(arch, image, masking_strategy=masking)
    """

    def __init__(self, sigma: float = 10.0) -> None:
        """Initialize the BlurMasking strategy.

        Args:
            sigma: Standard deviation of the Gaussian kernel in pixels.
        """
        self.sigma = sigma

    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        """Apply Gaussian-blur masking to a batch of coalitions."""
        from scipy.ndimage import gaussian_filter

        blurred = gaussian_filter(image, sigma=[self.sigma, self.sigma, 0])
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape

        masked_images = np.stack([image] * n_coalitions, axis=0)

        absence_mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    absence_mask[i] |= player_masks[j]

        return np.where(absence_mask[..., np.newaxis], blurred[np.newaxis], masked_images)


class MarginalSampling(PixelMaskingStrategy):
    """Imputes absent regions with samples from a bank of reference images.

    Per coalition, draws one reference image uniformly at random and copies its
    pixels into the absent region. This is the *marginal* feature-removal
    function in the removal-based explanation framework of
    Covert, Lundberg & Lee (JMLR 2021) — absent pixels are drawn from the data
    distribution rather than replaced with a fixed baseline, so the masked
    image stays closer to the natural image manifold than ``MeanColorMasking``
    or ``ZeroMasking``.

    When ``random_state`` is set, calls to :meth:`apply` are reproducible across
    invocations (the RNG is re-seeded at the start of every call). When
    ``random_state`` is ``None``, a fresh non-deterministic RNG is drawn per call.

    Args:
        reference_images: ``(N, H, W, C)`` stack of reference images. Must have
            the same spatial shape ``(H, W, C)`` as the image being explained.
        random_state: Seed for the per-coalition reference draw. Defaults to ``None``
            (non-deterministic).

    Example::

        refs = np.stack([np.array(Image.open(p).resize((H, W))) for p in ref_paths])
        masking = MarginalSampling(reference_images=refs, random_state=0)
    """

    def __init__(
        self,
        reference_images: np.ndarray,
        random_state: int | None = None,
    ) -> None:
        """Initialize the MarginalSampling strategy.

        Args:
            reference_images: ``(N, H, W, C)`` stack of reference images.
            random_state: Seed for the reference draw. Defaults to ``None``.
        """
        refs = np.asarray(reference_images)
        if refs.ndim != 4:
            msg = f"reference_images must be 4-D (N, H, W, C); got shape {refs.shape}."
            raise ValueError(msg)
        self.reference_images = refs
        self.random_state = random_state

    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        """Apply marginal-sampling masking to a batch of coalitions."""
        if self.reference_images.shape[1:] != image.shape:
            msg = (
                f"reference_images shape {self.reference_images.shape[1:]} does not match "
                f"image shape {image.shape}. Resize reference images to match."
            )
            raise ValueError(msg)

        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape

        absence_mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    absence_mask[i] |= player_masks[j]

        # Re-seed each call so reproducibility holds across multiple .apply() invocations
        # on the same strategy instance.
        rng = np.random.default_rng(self.random_state)
        ref_idx = rng.integers(0, len(self.reference_images), size=n_coalitions)
        refs = self.reference_images[ref_idx]  # (B, H, W, C)
        return np.where(absence_mask[..., np.newaxis], refs, image[np.newaxis]).astype(image.dtype)


class InpaintingMasking(PixelMaskingStrategy):
    """Imputes absent regions via a user-supplied inpainter callable.

    The inpainter is invoked once per coalition with the original image and a
    per-coalition boolean absence mask, and must return the inpainted image.
    Keeps ``shapiq`` free of heavy inpainter dependencies — the user wires in
    their own model (e.g. ``diffusers.AutoPipelineForInpainting`` for diffusion
    inpainting, ``cv2.inpaint``, or ``skimage.restoration.inpaint_biharmonic``).

    The inpainter signature:

    .. code-block:: python

        def inpainter(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            # image: (H, W, C) — the original image
            # mask:  (H, W)    — bool; True where pixels should be filled
            # returns: (H, W, C) — the inpainted image
            ...

    This is the *conditional* feature-removal function in Covert et al.'s
    framework — absent pixels are reconstructed conditional on the visible
    surroundings, which is the principled target for image SHAP. Cost is
    dominated by one inpainter call per coalition; batched inpainters can be
    wrapped externally if throughput matters.

    Args:
        inpainter: Callable that maps ``(image, mask) -> image``.

    Example::

        from skimage.restoration import inpaint_biharmonic

        def inpainter(img, mask):
            return (
                inpaint_biharmonic(img / 255.0, mask, channel_axis=-1) * 255
            ).astype(np.uint8)

        masking = InpaintingMasking(inpainter)
    """

    def __init__(
        self,
        inpainter: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        """Initialize the InpaintingMasking strategy.

        Args:
            inpainter: Callable ``f(image, mask) -> image``. ``image`` is
                ``(H, W, C)``; ``mask`` is ``(H, W)`` bool with ``True`` marking
                pixels to fill; the return is ``(H, W, C)``.
        """
        self.inpainter = inpainter

    def apply(
        self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray
    ) -> np.ndarray:
        """Apply inpainting masking to a batch of coalitions.

        Output dtype follows what the inpainter returns: ``np.stack`` promotes to
        the common dtype, so a ``uint8`` image plus a ``float32`` inpainter result
        produces a ``float32`` batch. Pre-allocating with ``image.dtype`` would
        silently truncate floats to ``uint8``.
        """
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape

        absence_mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    absence_mask[i] |= player_masks[j]

        return np.stack(
            [
                image if not absence_mask[i].any() else self.inpainter(image, absence_mask[i])
                for i in range(n_coalitions)
            ],
            axis=0,
        )


class LatentMaskingStrategy(ABC):
    """Defines how tokens are masked in latent/embedding space."""

    @abstractmethod
    def predict_logits(
        self,
        model: Model,
        pixel_values: torch.Tensor,
        bool_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model with masked tokens and return logits.

        Args:
            model: The vision model.
            pixel_values: ``(1, C, H, W)`` pre-processed image tensor.
            bool_masks: ``(B, n_tokens)`` boolean mask; ``True`` = token masked (absent).

        Returns:
            Logit tensor of shape ``(B, n_classes)``.
        """
        ...


class BoolMaskedPosStrategy(LatentMaskingStrategy):
    """Masks tokens via the ``bool_masked_pos`` argument in the model forward pass."""

    def predict_logits(
        self,
        model: Model,
        pixel_values: torch.Tensor,
        bool_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model with ``bool_masked_pos`` masking and return logits."""
        import torch

        # ViTForImageClassification has mask_token=None by default, initialise it so the
        # embedding layer can replace masked patch tokens during the forward pass.
        embeddings = model.vit.embeddings
        if embeddings.mask_token is None:
            embeddings.mask_token = torch.nn.Parameter(torch.zeros(1, 1, model.config.hidden_size))
        batch = pixel_values.repeat(bool_masks.shape[0], 1, 1, 1)
        return model(pixel_values=batch, bool_masked_pos=bool_masks).logits


class MaskTokenStrategy(LatentMaskingStrategy):
    """Masks tokens by zeroing the ``mask_token`` embedding before the forward pass.

    Initialises ``model.vit.embeddings.mask_token`` to a zero vector so that
    masked patches contribute nothing to the CLS representation.  Works with
    HuggingFace ``ViTForImageClassification`` checkpoints where the mask token
    is not pre-initialised.
    """

    def predict_logits(
        self,
        model: Model,
        pixel_values: torch.Tensor,
        bool_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model with zero mask-token masking and return logits."""
        import torch

        model.vit.embeddings.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, model.config.hidden_size)
        )
        batch = pixel_values.repeat(bool_masks.shape[0], 1, 1, 1)
        return model(pixel_values=batch, bool_masked_pos=bool_masks).logits


class ManifoldMaskingStrategy(ABC):
    """Masks model activations rather than input pixels.

    Strategies of this kind run the (unmasked) image through the model and
    intervene on an intermediate representation — typically by attenuating
    activations at spatial locations the absent players cover. The model never
    sees an out-of-distribution input, so the *missingness bias* that plagues
    pixel-space maskers (the model reacting to the unfamiliar fill colour or
    the mask boundary) does not apply.

    See: Balasubramanian & Feizi, *Towards Improved Input Masking for CNNs*,
    ICCV 2023.
    """

    @abstractmethod
    def value_function(
        self,
        model: torch.nn.Module,
        preprocess: Callable[[np.ndarray], torch.Tensor],
        image: np.ndarray,
        player_masks: np.ndarray,
        coalitions: np.ndarray,
        class_id: int,
    ) -> np.ndarray:
        """Run the model on the image with manifold masking and return target-class scores.

        Args:
            model: The torch ``nn.Module`` to run.
            preprocess: Callable mapping a ``(H, W, C)`` image to a ``(C, H, W)`` tensor.
            image: The ``(H, W, C)`` image being explained.
            player_masks: ``(n_players, H, W)`` boolean masks per player.
            coalitions: ``(n_coalitions, n_players)`` boolean array; ``True`` = present.
            class_id: Class index whose softmax probability is returned.

        Returns:
            ``(n_coalitions,)`` array of target-class probabilities.
        """
        ...


class LayerMasking(ManifoldMaskingStrategy):
    """Masks intermediate CNN activations at a chosen layer.

    Implements the *layer masking* of Balasubramanian & Feizi
    (`arXiv:2211.14646 <https://arxiv.org/abs/2211.14646>`_): a forward hook on
    ``layer_name`` multiplies the layer's output by a per-coalition spatial
    mask, downsampled to the feature-map resolution. The input image is never
    altered, so missingness bias from pixel-space fills (mean color, zero,
    blur) is eliminated. This is the CNN analogue of attention masking for ViT.

    Args:
        layer_name: Dotted path to the submodule whose output is masked
            (passed to ``model.get_submodule``). For ``torchvision.models.resnet18``
            sensible choices are ``"layer1"`` (high spatial resolution, strong
            attenuation) through ``"layer4"`` (low resolution, mild attenuation).
            Defaults to ``"layer2"``.

    Example::

        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights).eval()
        preprocess = weights.transforms()
        masking = LayerMasking(layer_name="layer2")
    """

    def __init__(self, layer_name: str = "layer2") -> None:
        """Initialize the LayerMasking strategy.

        Args:
            layer_name: Dotted path to the layer whose activations are masked.
        """
        self.layer_name = layer_name

    def value_function(
        self,
        model: torch.nn.Module,
        preprocess: Callable[[np.ndarray], torch.Tensor],
        image: np.ndarray,
        player_masks: np.ndarray,
        coalitions: np.ndarray,
        class_id: int,
    ) -> np.ndarray:
        """Run the model with layer-masked activations and return class probabilities."""
        import torch
        import torch.nn.functional as F  # noqa: N812

        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        b = coalitions.shape[0]
        _, H, W = player_masks.shape

        # Build the per-coalition absence mask, then convert to a visibility mask
        # in [0, 1] that the hook multiplies into the activations.
        absence_mask = np.zeros((b, H, W), dtype=bool)
        for i, coal in enumerate(coalitions):
            for j, is_present in enumerate(coal):
                if not is_present:
                    absence_mask[i] |= player_masks[j]
        visible = torch.as_tensor(np.logical_not(absence_mask), dtype=torch.float32)

        # Preprocess the single image once, then broadcast to a batch of B copies.
        single = preprocess(image)
        if single.ndim == 3:
            single = single.unsqueeze(0)  # (1, C, H_in, W_in)
        device = next(model.parameters()).device
        batch_input = single.to(device).expand(b, -1, -1, -1).contiguous()
        visible = visible.to(device)

        def hook(_module, _inputs, output):
            spatial = output.shape[-2:]
            m = F.interpolate(visible.unsqueeze(1), size=spatial, mode="area")  # (B, 1, h, w)
            return output * m

        handle = None
        try:
            layer = model.get_submodule(self.layer_name)
            handle = layer.register_forward_hook(hook)
            with torch.no_grad():
                output = model(batch_input)
            logits = output.logits if hasattr(output, "logits") else output
        finally:
            if handle is not None:
                handle.remove()

        probs = torch.softmax(logits, dim=-1)
        return probs[:, class_id].cpu().numpy()
