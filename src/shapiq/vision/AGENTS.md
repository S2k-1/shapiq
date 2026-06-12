# AGENTS.md — `shapiq.vision`

This file documents common mistakes and confusion points agents encounter in the vision subpackage. If you discover something surprising, alert the developer and add a note here.

## Package layout

Core library code lives in these modules (ignore `tutorials/`, `prototypes/`, and `vision_package_test.ipynb` — they are notebooks, not the public API):

| Module | Role |
|--------|------|
| `players.py` | Split an image into cooperative-game players (pixel or token regions) |
| `masking.py` | Replace absent players before a forward pass (pixel, token, or layer space) |
| `architecture.py` | Bind player + masking strategies to a concrete model; implement `value_function` |
| `imputer.py` | `ImageImputer` — batches coalitions and calls the architecture strategy |
| `explainer.py` | `ImageExplainer` — wraps the imputer with a shapiq approximator |
| `utils.py` | Image format conversion (`as_hwc_array`, `to_tensor_chw`, device helpers) |

Plotting lives separately in `shapiq.plot.vision` (`image_attribution_plot`), not in this package.

## Data flow

```
Image → PlayerStrategy.get_masks() → MaskingStrategy.apply() → model forward → scalar value
         (numpy HWC)                  (per coalition batch)        (per architecture)
```

`ImageImputer` sits between the approximator and the architecture: it converts numpy coalitions to torch, chunks them by `batch_size`, and calls `architecture.value_function`.

## Coalition semantics

Across players, masking, and imputer code, **`True` means the player is present** (region stays visible) and **`False` means absent** (region is imputed or masked). Do not invert this when writing tests or new strategies.

## Image format conventions

- **Player strategies** and `prepare()` receive numpy arrays in **`(H, W, C)`** layout.
- **CNN masking** and most model forwards use **`(C, H, W)`** float tensors via `to_tensor_chw`.
- **`as_hwc_array`** accepts PIL, numpy `(H,W)` / `(H,W,C)` / `(C,H,W)`, or torch tensors — but **not 4-D batched inputs**. Pass one image at a time.
- For HuggingFace processors, masked CNN/HF paths convert back to HWC uint8 or float lists before calling the processor.

## Architecture strategy selection

Pick the class that matches how the model should be called and what the value function returns:

| Class | Model type | Masking space | Value returned |
|-------|-----------|---------------|----------------|
| `CNNArchitecture` | Raw `torch.nn.Module` (e.g. torchvision ResNet) | Pixel | Logit of auto-detected class |
| `TransformerArchitecture` | HF ViT + processor | Token (`bool_masked_pos`) | Softmax prob of auto-detected class |
| `CustomViTArchitecture` | Custom ViT (no HF processor) | Token | Softmax prob of given `class_id` |
| `LayerMaskedCNNArchitecture` | Raw module + `preprocess` callable | Intermediate layer (hook) | Softmax prob of given `class_id` |
| `HuggingFacePixelArchitecture` | Any HF classifier with `pixel_values` | Pixel (mask then re-process) | Softmax prob |
| `ConvNeXtArchitecture` | Alias of `HuggingFacePixelArchitecture` | Pixel | Softmax prob |
| `DINOv2Architecture` | SSL backbone + processor | Pixel | Cosine similarity to unmasked embedding |
| `CLIPArchitecture` | CLIP model + processor | Pixel | Softmax prob over `text_prompts` |

`prepare()` always runs one unmasked forward pass (or embedding) to cache image-dependent state and, where applicable, the predicted class index.

## Lazy PyTorch imports

**Do not add top-level `import torch` in any vision module.** Torch is imported inside the methods that need it. This is enforced by `tests_vision/test_framework_agnostic.py`. `TYPE_CHECKING` blocks may reference torch types only.

## Common mistakes

1. **Superpixel non-determinism** — `SuperpixelStrategy` uses SLIC and the actual `n_players` may differ from `n_segments`. For deterministic unit tests, use `FixedMasksStrategy` from `conftest.py` or `CustomPlayerStrategy` with precomputed masks.

2. **Uncovered pixels** — `CustomPlayerStrategy` raises a `UserWarning` (not `ValueError`) when pixels belong to no player. Those pixels stay visible in every coalition.

3. **`ImageExplainer` lazy import** — It is not in `__all__`; access via `from shapiq.vision import ImageExplainer` (resolved by `__getattr__` in `__init__.py`).

4. **ViT token grid mismatch** — `PatchStrategy` and `TransformerArchitecture.default_player_strategy()` derive `grid_size` from `model.config.image_size // model.config.patch_size`. A mismatch between processor output size and model config causes silent wrong masks.

5. **`MaskTokenStrategy` side effect** — It sets `model.vit.embeddings.mask_token` to zeros on every `apply()` call. Tests using a shared model instance should account for this mutation.

6. **Logit vs probability** — `CNNArchitecture` returns raw logits; `TransformerArchitecture` and most HF wrappers return softmax probabilities. Comparisons across architecture types are not directly comparable without normalization.

7. **Batching location** — `batch_size` is handled in `ImageImputer.value_function`, not in architecture classes. Architecture `value_function` receives the full chunk passed by the imputer.

8. **Accessing player masks for plotting** — Use `explainer.imputer.player_masks` (numpy via `tensor_to_numpy`) with `shapiq.plot.vision.image_attribution_plot`.

## Commands

Run vision unit tests (fast, no real models):

```bash
uv run pytest tests/shapiq/tests_unit/tests_vision/ -m 'not integration'
```

Include integration tests (needs `torch`, `torchvision`, `transformers`, `scikit-image`):

```bash
uv sync --group all_ml --group test
uv run pytest tests/shapiq/tests_unit/tests_vision/
```

Run a single test file:

```bash
uv run pytest tests/shapiq/tests_unit/tests_vision/test_masking.py -v
```

Pre-commit (from project root):

```bash
uv run pre-commit run --all-files
```

## Test helpers (`tests_vision/conftest.py`)

Reuse these instead of inventing new mocks:

- `FixedMasksStrategy` — deterministic spatial masks
- `ChannelSumModel` — linear CNN whose output equals the sum of visible pixels (exact correctness checks)
- `MockViT` / `MockViTProcessor` — minimal HF-style ViT for `TransformerArchitecture` tests
- `tiny_image`, `two_player_masks`, `three_player_masks`, `image_24x24` — small fixtures

Integration tests in `test_integration_real_models.py` use randomly initialised weights (no checkpoint download) and are marked `@pytest.mark.integration`.

## Optional dependencies

| Dependency | Needed for |
|------------|-----------|
| `torch` | All `value_function` / imputer paths |
| `scikit-image` | `SuperpixelStrategy` (lazy import inside `get_masks`) |
| `torchvision` | ResNet integration tests |
| `transformers` | ViT / HF integration tests |

Core `shapiq` installs do not include these; use the `all_ml` dependency group for local development.

## Extending the package

When adding a new model type:

1. Subclass `ModelArchitectureStrategy` (or an existing HF/CNN base).
2. Implement `default_player_strategy`, `default_masking_strategy`, `prepare`, and `value_function`.
3. Pick or implement matching player and masking strategy base classes (`CNNPlayerStrategy` + `CNNMaskingStrategy`, or `TransformerPlayerStrategy` + `TransformerMaskingStrategy`, or `ManifoldMaskingStrategy` for layer hooks).
4. Add unit tests with deterministic fixtures first; add `@pytest.mark.integration` tests with real models only when necessary.

When adding a new player or masking strategy, add focused tests in `test_players.py` or `test_masking.py` and wire defaults through the relevant architecture class if appropriate.
