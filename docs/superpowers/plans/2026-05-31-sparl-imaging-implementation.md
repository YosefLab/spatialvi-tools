# SPARL Imaging Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate SPARL as the first model in a new `scviva/imaging/` submodule — inference-only, loading a pre-trained ViT backbone from local path or HuggingFace Hub, writing CLS-token embeddings to `adata.obsm["X_sparl"]`.

**Architecture:** New `ImagingBaseModel(SpatialBaseModel)` base class owns loading, data setup, and inference. `SPARL(ImagingBaseModel)` adds SPARL-specific checkpoint reading and channel handling. `SPARLModule(nn.Module)` wraps SPARL's `DinoVisionTransformer` and handles the channels tensor internally. No scvi-tools training infrastructure is used.

**Tech Stack:** PyTorch, scvi-tools (SpatialBaseModel, AnnDataManager, ObsmField), huggingface-hub, Pillow, tifffile, sparl (optional dep).

> **Branch note:** All commits must be on a new feature branch (e.g. `feat/sparl-imaging`), NOT on main. Create the branch before Task 1.

```bash
git checkout -b feat/sparl-imaging
```

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/scviva/imaging/__init__.py` | Re-exports `ImagingBaseModel`, `SPARL` |
| Create | `src/scviva/imaging/base/__init__.py` | Re-exports `ImagingBaseModel` |
| Create | `src/scviva/imaging/base/_imaging_base.py` | `ImagingBaseModel`, `_ImagePathDataset`, `_load_image_as_tensor` |
| Create | `src/scviva/imaging/sparl/__init__.py` | Re-exports `SPARL` |
| Create | `src/scviva/imaging/sparl/_module.py` | `SPARLModule(nn.Module)` |
| Create | `src/scviva/imaging/sparl/_model.py` | `SPARL(ImagingBaseModel)` |
| Create | `tests/imaging/__init__.py` | Empty |
| Create | `tests/imaging/conftest.py` | Shared fixtures: `_MinimalImagingModel`, `make_imaging_adata`, checkpoints |
| Create | `tests/imaging/base/__init__.py` | Empty |
| Create | `tests/imaging/base/test_imaging_base.py` | Tests for `ImagingBaseModel` via `_MinimalImagingModel` |
| Create | `tests/imaging/sparl/__init__.py` | Empty |
| Create | `tests/imaging/sparl/test_sparl.py` | Tests for `SPARL` (gated: `pytest.importorskip("sparl")`) |
| Modify | `src/scviva/__init__.py` | Add `imaging` to lazy import registry |
| Modify | `pyproject.toml` | Add `[imaging]` optional dependency group |

---

## Task 1: Scaffold — directory structure and pyproject.toml

**Files:**
- Create: all `__init__.py` stubs listed above
- Modify: `pyproject.toml`

- [ ] **Step 1: Create branch**

```bash
git checkout -b feat/sparl-imaging
```

- [ ] **Step 2: Create directory skeleton**

```bash
mkdir -p src/scviva/imaging/base
mkdir -p src/scviva/imaging/sparl
mkdir -p tests/imaging/base
mkdir -p tests/imaging/sparl
touch src/scviva/imaging/__init__.py
touch src/scviva/imaging/base/__init__.py
touch src/scviva/imaging/sparl/__init__.py
touch tests/imaging/__init__.py
touch tests/imaging/base/__init__.py
touch tests/imaging/sparl/__init__.py
```

- [ ] **Step 3: Add `[imaging]` extras to `pyproject.toml`**

In `pyproject.toml`, after the existing `optional-dependencies.hub` line, add:

```toml
optional-dependencies.imaging = [
  "huggingface-hub>=0.20",
  "Pillow>=10",
  "tifffile>=2023",
]
```

Also add `"scviva-tools[imaging]"` to `optional-dependencies.all`:
```toml
optional-dependencies.all = [ "scviva-tools[dev,doc,imaging,rapids,spatial,test,tutorials]" ]
```

- [ ] **Step 4: Create stub source files**

`src/scviva/imaging/base/_imaging_base.py`:
```python
"""ImagingBaseModel and image loading utilities."""
from __future__ import annotations
```

`src/scviva/imaging/sparl/_module.py`:
```python
"""SPARLModule: thin nn.Module wrapper around DinoVisionTransformer."""
from __future__ import annotations
```

`src/scviva/imaging/sparl/_model.py`:
```python
"""SPARL model for scviva-tools."""
from __future__ import annotations
```

`tests/imaging/conftest.py`:
```python
"""Shared fixtures for imaging tests."""
from __future__ import annotations
```

`tests/imaging/base/test_imaging_base.py`:
```python
"""Tests for ImagingBaseModel."""
from __future__ import annotations
```

`tests/imaging/sparl/test_sparl.py`:
```python
"""Tests for SPARL model. Requires sparl package."""
from __future__ import annotations
sparl = pytest.importorskip("sparl")  # skip entire module if sparl not installed
```

- [ ] **Step 5: Verify project still imports cleanly**

```bash
cd /Users/orikr/PycharmProjects/spatialvi-tools2
python -c "import scviva; print(scviva.__version__)"
```

Expected output: `0.1.3` (or current version). No errors.

- [ ] **Step 6: Commit scaffold**

```bash
git add src/scviva/imaging/ tests/imaging/ pyproject.toml
git commit -m "chore: scaffold imaging/ submodule structure"
```

---

## Task 2: `ImagingBaseModel` — `__init__`, `from_pretrained` (local path)

**Files:**
- Create: `src/scviva/imaging/base/_imaging_base.py`
- Create: `tests/imaging/conftest.py`
- Create: `tests/imaging/base/test_imaging_base.py`

- [ ] **Step 1: Write fixtures in `tests/imaging/conftest.py`**

```python
"""Shared fixtures for imaging tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from anndata import AnnData
from PIL import Image

from scviva.imaging.base._imaging_base import ImagingBaseModel


class _MinimalImagingModel(ImagingBaseModel):
    """Concrete subclass using a tiny Linear backbone — no sparl dependency."""

    _default_obsm_key = "X_test_imaging"

    @classmethod
    def _build_module(cls, checkpoint_data: dict) -> nn.Module:
        # Expects checkpoint with state_dict for nn.Sequential(Flatten, Linear(64, 8))
        mod = nn.Sequential(nn.Flatten(), nn.Linear(64, 8))
        mod.load_state_dict(checkpoint_data["state_dict"])
        return mod

    @classmethod
    def setup_anndata(
        cls, adata: AnnData, img_path_col: str, spatial_key: str = "spatial", **kwargs
    ) -> None:
        from scvi.data import AnnDataManager
        from scviva.data._fields import SpatialCoordsField

        fields = [SpatialCoordsField(obsm_key=spatial_key)]
        manager = AnnDataManager(fields=fields)
        manager.register_fields(adata, **kwargs)
        cls.register_manager(manager)
        adata.uns["scviva_imaging"] = {"img_path_col": img_path_col}


def make_imaging_adata(
    n: int = 20,
    n_channels: int = 1,
    img_size: int = 8,
    tmp_path=None,
) -> AnnData:
    """AnnData with spatial coords and obs['img_path'] pointing to real PNG files."""
    adata = AnnData(X=np.zeros((n, 1)))
    adata.obsm["spatial"] = np.random.rand(n, 2).astype(np.float32)
    if tmp_path is not None:
        paths = []
        for i in range(n):
            arr = np.random.randint(0, 255, (img_size, img_size), dtype=np.uint8)
            p = tmp_path / f"cell_{i}.png"
            Image.fromarray(arr, mode="L").save(p)
            paths.append(str(p))
        adata.obs["img_path"] = paths
    return adata


@pytest.fixture(scope="module")
def minimal_checkpoint(tmp_path_factory):
    """Tiny checkpoint for _MinimalImagingModel: Sequential(Flatten, Linear(64, 8))."""
    tmp = tmp_path_factory.mktemp("ckpt")
    mod = nn.Sequential(nn.Flatten(), nn.Linear(64, 8))
    ckpt = {"model_type": "minimal", "state_dict": mod.state_dict()}
    path = tmp / "minimal.pt"
    torch.save(ckpt, path)
    return str(path)


@pytest.fixture(scope="module")
def imaging_adata(tmp_path_factory):
    """AnnData with 20 cells, 1-channel 8x8 PNG images."""
    tmp = tmp_path_factory.mktemp("imgs")
    return make_imaging_adata(n=20, n_channels=1, img_size=8, tmp_path=tmp)
```

- [ ] **Step 2: Write failing test**

In `tests/imaging/base/test_imaging_base.py`:

```python
"""Tests for ImagingBaseModel."""
from __future__ import annotations

import pytest

from tests.imaging.conftest import _MinimalImagingModel


def test_from_pretrained_local_sets_module(minimal_checkpoint):
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    assert model.module is not None


def test_from_pretrained_local_module_in_eval(minimal_checkpoint):
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    assert not model.module.training
```

- [ ] **Step 3: Run test — expect FAIL**

```bash
pytest tests/imaging/base/test_imaging_base.py -v
```

Expected: `ImportError` or `AttributeError` — `ImagingBaseModel` not implemented yet.

- [ ] **Step 4: Implement `ImagingBaseModel.__init__` and `from_pretrained` (local) in `_imaging_base.py`**

```python
"""ImagingBaseModel and image loading utilities."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scviva.model.base._spatial_base import SpatialBaseModel

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class ImagingBaseModel(SpatialBaseModel):
    """Base class for image-based spatial models.

    Provides pre-trained backbone loading, AnnData/SpatialData integration,
    and inference via get_latent_representation(). Subclasses implement
    _build_module() and setup_anndata().
    """

    _default_obsm_key: str = "X_imaging"

    def __init__(self, *args, **kwargs) -> None:
        if args or kwargs:
            raise TypeError(
                f"Use {type(self).__name__}.from_pretrained() to load a model, "
                "not the constructor directly."
            )
        super().__init__(adata=None)
        self.module: nn.Module | None = None

    @classmethod
    def _build_module(cls, checkpoint_data: dict) -> nn.Module:
        raise NotImplementedError(
            f"{cls.__name__} must implement _build_module(checkpoint_data)"
        )

    @classmethod
    def _resolve_path(cls, model_name_or_path: str) -> str:
        """Return local filesystem path; download from HuggingFace if needed."""
        if os.path.exists(model_name_or_path):
            return model_name_or_path
        # Heuristic: HF repo IDs look like "user/model-name" (a "/" but no os.sep)
        if "/" in model_name_or_path and os.sep not in model_name_or_path:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise ImportError(
                    "huggingface-hub is required to load from HuggingFace. "
                    "Install with: pip install 'scviva-tools[imaging]'"
                ) from e
            logger.info(f"Downloading {model_name_or_path} from HuggingFace Hub...")
            return hf_hub_download(repo_id=model_name_or_path, filename="model.pt")
        raise FileNotFoundError(
            f"Could not resolve model path: {model_name_or_path!r}. "
            "Provide a local path or a HuggingFace repo ID (e.g. 'YosefLab/sparl-imc-vitb16')."
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "ImagingBaseModel":
        """Load a pre-trained backbone from a local checkpoint or HuggingFace Hub.

        Parameters
        ----------
        model_name_or_path
            Local path to a ``.pt`` checkpoint file, or a HuggingFace repo ID
            (e.g. ``"YosefLab/sparl-imc-vitb16"``).

        Returns
        -------
        Loaded model instance with ``self.module`` set.
        """
        path = cls._resolve_path(model_name_or_path)
        checkpoint_data = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls()
        obj.module = cls._build_module(checkpoint_data)
        obj.module.eval()
        return obj
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/imaging/base/test_imaging_base.py::test_from_pretrained_local_sets_module \
       tests/imaging/base/test_imaging_base.py::test_from_pretrained_local_module_in_eval -v
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add src/scviva/imaging/base/_imaging_base.py tests/imaging/conftest.py \
        tests/imaging/base/test_imaging_base.py
git commit -m "feat(imaging): ImagingBaseModel __init__ and from_pretrained (local)"
```

---

## Task 3: `ImagingBaseModel` — `from_pretrained` HuggingFace path

**Files:**
- Modify: `tests/imaging/base/test_imaging_base.py`
- Modify: `src/scviva/imaging/base/_imaging_base.py` (already has HF logic — just tests needed)

- [ ] **Step 1: Write failing test**

Add to `tests/imaging/base/test_imaging_base.py`:

```python
import torch


def test_from_pretrained_hf_calls_hub_download(tmp_path, minimal_checkpoint, monkeypatch):
    """from_pretrained("user/repo") should call hf_hub_download and load the result."""
    import huggingface_hub

    call_log = {}

    def mock_download(repo_id, filename, **kwargs):
        call_log["repo_id"] = repo_id
        call_log["filename"] = filename
        # Return the path to our real local checkpoint
        return minimal_checkpoint

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", mock_download)

    model = _MinimalImagingModel.from_pretrained("testuser/testmodel")
    assert model.module is not None
    assert call_log["repo_id"] == "testuser/testmodel"
    assert call_log["filename"] == "model.pt"


def test_from_pretrained_bad_path_raises():
    with pytest.raises(FileNotFoundError, match="Could not resolve"):
        _MinimalImagingModel.from_pretrained("/nonexistent/path/model.pt")
```

- [ ] **Step 2: Run tests — expect PASS** (HF logic already implemented in Task 2)

```bash
pytest tests/imaging/base/test_imaging_base.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/imaging/base/test_imaging_base.py
git commit -m "test(imaging): HuggingFace path and bad path error in ImagingBaseModel"
```

---

## Task 4: `ImagingBaseModel` — `setup_anndata` + `from_spatialdata` override

**Files:**
- Modify: `src/scviva/imaging/base/_imaging_base.py`
- Modify: `tests/imaging/base/test_imaging_base.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/imaging/base/test_imaging_base.py`:

```python
from anndata import AnnData


def test_setup_anndata_stores_img_path_col(imaging_adata):
    _MinimalImagingModel.setup_anndata(
        imaging_adata, img_path_col="img_path", spatial_key="spatial"
    )
    assert "scviva_imaging" in imaging_adata.uns
    assert imaging_adata.uns["scviva_imaging"]["img_path_col"] == "img_path"


def test_from_spatialdata_returns_adata(tmp_path):
    """from_spatialdata should register fields and return the AnnData, not a model."""
    from unittest.mock import MagicMock

    # Build a minimal mock SpatialData object
    adata = AnnData(X=np.zeros((5, 1)))
    adata.obsm["spatial"] = np.random.rand(5, 2).astype(np.float32)
    adata.obs["img_path"] = [str(tmp_path / f"c{i}.png") for i in range(5)]

    sdata = MagicMock()
    sdata.__getitem__ = MagicMock(return_value=adata)
    sdata.__contains__ = MagicMock(return_value=True)

    result = _MinimalImagingModel.from_spatialdata(
        sdata, table_key="table", img_path_col="img_path"
    )
    assert isinstance(result, AnnData)
    assert "scviva_imaging" in result.uns
```

- [ ] **Step 2: Run tests — expect FAIL** (`from_spatialdata` not overridden yet)

```bash
pytest tests/imaging/base/test_imaging_base.py::test_setup_anndata_stores_img_path_col \
       tests/imaging/base/test_imaging_base.py::test_from_spatialdata_returns_adata -v
```

Expected: `test_setup_anndata_stores_img_path_col` passes (already in `_MinimalImagingModel`); `test_from_spatialdata_returns_adata` FAILS because `SpatialBaseModel.from_spatialdata` tries to call `cls(adata)`.

- [ ] **Step 3: Override `from_spatialdata` in `_imaging_base.py`**

Add to `ImagingBaseModel` class (after `from_pretrained`):

```python
    @classmethod
    def from_spatialdata(
        cls,
        sdata,
        table_key: str = "table",
        region: str | None = None,
        **kwargs,
    ) -> "AnnData":
        """Register SpatialData fields and return the extracted AnnData.

        Unlike other scviva models, imaging models are loaded via
        :meth:`from_pretrained`. This method only registers the data fields
        and returns the AnnData for passing to :meth:`get_latent_representation`.

        Parameters
        ----------
        sdata
            A :class:`spatialdata.SpatialData` object.
        table_key
            Key in ``sdata`` pointing to the AnnData table.
        region
            Region name to subset.
        **kwargs
            Passed to :meth:`setup_anndata`.

        Returns
        -------
        The extracted (and registered) AnnData object.
        """
        cls.setup_spatialdata(sdata, table_key=table_key, region=region, **kwargs)
        adata = sdata[table_key]
        if region is not None:
            region_key = adata.uns.get("spatialdata_attrs", {}).get("region_key", "region")
            adata = adata[adata.obs[region_key] == region].copy()
        return adata
```

- [ ] **Step 4: Run tests — expect all PASS**

```bash
pytest tests/imaging/base/test_imaging_base.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scviva/imaging/base/_imaging_base.py tests/imaging/base/test_imaging_base.py
git commit -m "feat(imaging): setup_anndata field registration and from_spatialdata override"
```

---

## Task 5: `ImagingBaseModel` — inference dataloader and `get_latent_representation`

**Files:**
- Modify: `src/scviva/imaging/base/_imaging_base.py`
- Modify: `tests/imaging/base/test_imaging_base.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/imaging/base/test_imaging_base.py`:

```python
import numpy as np


def test_get_latent_representation_shape(imaging_adata, minimal_checkpoint):
    _MinimalImagingModel.setup_anndata(
        imaging_adata, img_path_col="img_path", spatial_key="spatial"
    )
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    result = model.get_latent_representation(imaging_adata)
    assert result.shape == (len(imaging_adata), 8)


def test_get_latent_representation_writes_obsm(imaging_adata, minimal_checkpoint):
    _MinimalImagingModel.setup_anndata(
        imaging_adata, img_path_col="img_path", spatial_key="spatial"
    )
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    model.get_latent_representation(imaging_adata)
    assert "X_test_imaging" in imaging_adata.obsm
    assert imaging_adata.obsm["X_test_imaging"].shape == (len(imaging_adata), 8)


def test_get_latent_representation_no_nan(imaging_adata, minimal_checkpoint):
    _MinimalImagingModel.setup_anndata(
        imaging_adata, img_path_col="img_path", spatial_key="spatial"
    )
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    result = model.get_latent_representation(imaging_adata)
    assert not np.any(np.isnan(result))


def test_get_latent_representation_obsm_key_override(imaging_adata, minimal_checkpoint):
    _MinimalImagingModel.setup_anndata(
        imaging_adata, img_path_col="img_path", spatial_key="spatial"
    )
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    model.get_latent_representation(imaging_adata, obsm_key="X_custom")
    assert "X_custom" in imaging_adata.obsm
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/imaging/base/test_imaging_base.py::test_get_latent_representation_shape -v
```

Expected: `AttributeError` — `get_latent_representation` not implemented.

- [ ] **Step 3: Implement `_load_image_as_tensor`, `_ImagePathDataset`, `_build_inference_dataloader`, and `get_latent_representation` in `_imaging_base.py`**

Add after the `ImagingBaseModel` class definition (module-level helpers):

```python
def _load_image_as_tensor(path: str) -> torch.Tensor:
    """Load an image file to a float32 tensor of shape (C, H, W) in [0, 1]."""
    suffix = path.lower().rsplit(".", 1)[-1]
    if suffix in ("tif", "tiff"):
        try:
            import tifffile
        except ImportError as e:
            raise ImportError(
                "tifffile is required for TIFF images. "
                "Install with: pip install 'scviva-tools[imaging]'"
            ) from e
        arr = tifffile.imread(path).astype(np.float32)
    else:
        from PIL import Image as PILImage
        arr = np.array(PILImage.open(path), dtype=np.float32)

    # Ensure (C, H, W)
    if arr.ndim == 2:
        arr = arr[np.newaxis]           # (H, W) → (1, H, W)
    elif arr.ndim == 3:
        arr = arr.transpose(2, 0, 1)   # (H, W, C) → (C, H, W)

    # Normalize to [0, 1]
    max_val = arr.max()
    if max_val > 1.0:
        arr = arr / (255.0 if max_val <= 255.0 else max_val)

    return torch.from_numpy(arr)


class _ImagePathDataset(Dataset):
    """Minimal dataset loading per-cell image crops from disk paths."""

    def __init__(self, paths: list[str]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _load_image_as_tensor(self.paths[idx])
```

Add these methods inside `ImagingBaseModel`:

```python
    def _build_inference_dataloader(
        self, adata: "AnnData", batch_size: int
    ) -> DataLoader:
        img_path_col = adata.uns["scviva_imaging"]["img_path_col"]
        paths = adata.obs[img_path_col].tolist()
        dataset = _ImagePathDataset(paths)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    def get_latent_representation(
        self,
        adata: "AnnData",
        batch_size: int = 256,
        obsm_key: str | None = None,
        device: str = "cpu",
        backend: str = "cpu",
    ) -> np.ndarray:
        """Run inference and write CLS-token embeddings to ``adata.obsm[obsm_key]``.

        Parameters
        ----------
        adata
            AnnData registered via :meth:`setup_anndata` or :meth:`from_spatialdata`.
        batch_size
            Number of cells per forward pass.
        obsm_key
            Key to write embeddings into. Defaults to ``cls._default_obsm_key``.
        device
            Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        backend
            ``"cpu"`` returns numpy array. ``"rapids"`` returns cupy array
            (requires ``pip install 'scviva-tools[rapids]'``).

        Returns
        -------
        Embedding array of shape ``(n_cells, embed_dim)``.
        """
        if self.module is None:
            raise RuntimeError(
                "No backbone loaded. Call from_pretrained() before get_latent_representation()."
            )
        if obsm_key is None:
            obsm_key = self._default_obsm_key

        self.module.eval()
        self.module.to(device)

        loader = self._build_inference_dataloader(adata, batch_size)
        all_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                emb = self.module(batch)
                all_embeddings.append(emb.cpu().numpy())

        result = np.concatenate(all_embeddings, axis=0)
        adata.obsm[obsm_key] = result

        if backend == "rapids":
            try:
                import cupy as cp
                return cp.asarray(result)
            except ImportError as e:
                raise ImportError(
                    "backend='rapids' requires cupy. "
                    "Install with: pip install 'scviva-tools[rapids]'"
                ) from e

        return result
```

- [ ] **Step 4: Run all imaging base tests — expect all PASS**

```bash
pytest tests/imaging/base/test_imaging_base.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scviva/imaging/base/_imaging_base.py tests/imaging/base/test_imaging_base.py
git commit -m "feat(imaging): inference dataloader and get_latent_representation"
```

---

## Task 6: `SPARLModule` — ViT wrapper

**Files:**
- Create: `src/scviva/imaging/sparl/_module.py`
- Create: `tests/imaging/sparl/test_sparl.py` (initial skeleton)
- Create: `tests/imaging/conftest.py` additions (SPARL fixtures)

- [ ] **Step 1: Add SPARL fixtures to `tests/imaging/conftest.py`**

```python
import pytest
import torch


@pytest.fixture(scope="module")
def sparl_tiny_vit():
    """Tiny DinoVisionTransformer: 3 channels, 16x16 input, patch 8, embed_dim 32."""
    sparl = pytest.importorskip("sparl")
    from sparl.models.backbones.vision_transformer import DinoVisionTransformer

    vit = DinoVisionTransformer(
        channel_names=[0, 1, 2],
        img_size=16,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=2,
        ffn_ratio=4.0,
        qkv_bias=True,
        layerscale_init_values=None,
        norm_layer="layernorm",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        untie_cls_and_patch_norms=False,
        untie_global_and_local_cls_norm=False,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        pos_embed_type="learnable",
    )
    vit.eval()
    return vit


@pytest.fixture(scope="module")
def sparl_checkpoint(tmp_path_factory, sparl_tiny_vit):
    """SPARL checkpoint dict for the tiny ViT (3ch, 16x16, embed_dim=32)."""
    tmp = tmp_path_factory.mktemp("sparl_ckpt")
    ckpt = {
        "epoch": 1,
        "step": 10,
        "model": {"teacher": {"backbone": sparl_tiny_vit.state_dict()}},
        "model_config": {
            "arch": "vit",
            "channel_names": [0, 1, 2],
            "img_size": 16,
            "patch_size": 8,
            "embed_dim": 32,
            "depth": 2,
            "num_heads": 2,
            "ffn_ratio": 4.0,
            "qkv_bias": True,
            "layerscale_init_values": None,
            "norm_layer": "layernorm",
            "ffn_layer": "mlp",
            "ffn_bias": True,
            "proj_bias": True,
            "untie_cls_and_patch_norms": False,
            "untie_global_and_local_cls_norm": False,
            "interpolate_antialias": False,
            "interpolate_offset": 0.1,
            "pos_embed_type": "learnable",
        },
    }
    path = tmp / "sparl_tiny.pt"
    torch.save(ckpt, path)
    return str(path)


@pytest.fixture(scope="module")
def sparl_imaging_adata(tmp_path_factory):
    """AnnData for SPARL: 10 cells, 3-channel 16x16 PNGs."""
    import numpy as np
    from anndata import AnnData
    from PIL import Image

    tmp = tmp_path_factory.mktemp("sparl_imgs")
    n, n_channels, img_size = 10, 3, 16
    adata = AnnData(X=np.zeros((n, 1)))
    adata.obsm["spatial"] = np.random.rand(n, 2).astype(np.float32)
    paths = []
    for i in range(n):
        arr = np.random.randint(0, 255, (img_size, img_size, n_channels), dtype=np.uint8)
        p = tmp / f"cell_{i}.png"
        Image.fromarray(arr, mode="RGB").save(p)
        paths.append(str(p))
    adata.obs["img_path"] = paths
    return adata
```

- [ ] **Step 2: Write failing test for `SPARLModule`**

In `tests/imaging/sparl/test_sparl.py`:

```python
"""Tests for SPARL model. Requires sparl package."""
from __future__ import annotations

import numpy as np
import pytest
import torch

sparl = pytest.importorskip("sparl")

from tests.imaging.conftest import sparl_imaging_adata, sparl_checkpoint  # noqa: F401


def test_sparl_module_output_shape(sparl_tiny_vit):
    from scviva.imaging.sparl._module import SPARLModule

    module = SPARLModule(sparl_tiny_vit)
    module.eval()
    x = torch.randn(4, 3, 16, 16)  # batch=4, 3ch, 16x16
    with torch.no_grad():
        out = module(x)
    assert out.shape == (4, 32)  # embed_dim=32


def test_sparl_module_no_nan(sparl_tiny_vit):
    from scviva.imaging.sparl._module import SPARLModule

    module = SPARLModule(sparl_tiny_vit)
    module.eval()
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        out = module(x)
    assert not torch.any(torch.isnan(out))
```

- [ ] **Step 3: Run tests — expect FAIL**

```bash
pytest tests/imaging/sparl/test_sparl.py::test_sparl_module_output_shape -v
```

Expected: `ImportError` — `SPARLModule` not implemented.

- [ ] **Step 4: Implement `SPARLModule` in `src/scviva/imaging/sparl/_module.py`**

```python
"""SPARLModule: thin nn.Module wrapper around SPARL's DinoVisionTransformer."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from sparl.models.backbones.vision_transformer import DinoVisionTransformer


class SPARLModule(nn.Module):
    """Wraps a SPARL DinoVisionTransformer for scviva-tools inference.

    Handles the ``channels`` tensor required by DinoVisionTransformer internally,
    so callers only need to pass the image batch ``(B, C, H, W)``.
    """

    def __init__(self, backbone: "DinoVisionTransformer") -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning CLS-token embeddings.

        Parameters
        ----------
        x
            Image batch of shape ``(B, C, H, W)``, float32, values in [0, 1].

        Returns
        -------
        CLS-token embeddings of shape ``(B, embed_dim)``.
        """
        B = x.shape[0]
        # Expand the backbone's stored channel_names to match the batch
        channels = (
            self.backbone.channel_names.unsqueeze(0).expand(B, -1).to(x.device)
        )
        result = self.backbone(x, channels)
        return result["x_norm_clstoken"]
```

- [ ] **Step 5: Run SPARL module tests — expect PASS**

```bash
pytest tests/imaging/sparl/test_sparl.py::test_sparl_module_output_shape \
       tests/imaging/sparl/test_sparl.py::test_sparl_module_no_nan -v
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add src/scviva/imaging/sparl/_module.py tests/imaging/sparl/test_sparl.py \
        tests/imaging/conftest.py
git commit -m "feat(imaging/sparl): SPARLModule wrapping DinoVisionTransformer"
```

---

## Task 7: `SPARL` model — `from_pretrained` with SPARL checkpoint

**Files:**
- Create: `src/scviva/imaging/sparl/_model.py`
- Modify: `tests/imaging/sparl/test_sparl.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/imaging/sparl/test_sparl.py`:

```python
from scviva.imaging.sparl._model import SPARL
from scviva.imaging.sparl._module import SPARLModule


def test_sparl_from_pretrained_local(sparl_checkpoint):
    model = SPARL.from_pretrained(sparl_checkpoint)
    assert model.module is not None
    assert isinstance(model.module, SPARLModule)
    assert not model.module.training


def test_sparl_from_pretrained_embed_dim(sparl_checkpoint):
    model = SPARL.from_pretrained(sparl_checkpoint)
    assert model.module.backbone.embed_dim == 32


def test_sparl_from_pretrained_hf(sparl_checkpoint, monkeypatch):
    import huggingface_hub

    def mock_download(repo_id, filename, **kwargs):
        return sparl_checkpoint

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", mock_download)
    model = SPARL.from_pretrained("YosefLab/sparl-imc-vitb16")
    assert isinstance(model.module, SPARLModule)
    # TODO: add real hub test once YosefLab/sparl-* is published on HuggingFace
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/imaging/sparl/test_sparl.py::test_sparl_from_pretrained_local -v
```

Expected: `ImportError` — `SPARL` not implemented.

- [ ] **Step 3: Implement `SPARL` in `src/scviva/imaging/sparl/_model.py`**

```python
"""SPARL model for scviva-tools — inference-only wrapper."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch.nn as nn

from scviva.imaging.base._imaging_base import ImagingBaseModel
from scviva.imaging.sparl._module import SPARLModule

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class SPARL(ImagingBaseModel):
    """Spatial Proteomics Analysis with Representation Learning.

    Loads a pre-trained ViT backbone (trained with DINO/iBOT self-supervised
    learning on multi-channel microscopy images) and runs inference to produce
    per-cell CLS-token embeddings stored in ``adata.obsm["X_sparl"]``.

    Parameters
    ----------
    model_name_or_path
        Not used directly — use :meth:`from_pretrained` instead.

    Examples
    --------
    >>> model = SPARL.from_pretrained("YosefLab/sparl-imc-vitb16")
    >>> SPARL.setup_anndata(adata, img_path_col="crop_path",
    ...                     channel_names=["CD3", "CD8", "DAPI"])
    >>> model.get_latent_representation(adata)
    >>> # adata.obsm["X_sparl"] now contains per-cell embeddings
    """

    _default_obsm_key: str = "X_sparl"

    @classmethod
    def _build_module(cls, checkpoint_data: dict) -> SPARLModule:
        """Build SPARLModule from a SPARL training checkpoint."""
        try:
            from sparl.models.backbones.vision_transformer import DinoVisionTransformer
        except ImportError as e:
            raise ImportError(
                "The sparl package is required for SPARL inference. "
                "Install it from source: pip install /path/to/SPARL"
            ) from e

        config = checkpoint_data["model_config"]
        arch = config.pop("arch", "vit")
        logger.info(f"Building SPARL backbone: arch={arch}, embed_dim={config.get('embed_dim')}")

        backbone = DinoVisionTransformer(**config)
        backbone_state = checkpoint_data["model"]["teacher"]["backbone"]
        backbone.load_state_dict(backbone_state)

        return SPARLModule(backbone)

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        img_path_col: str,
        channel_names: list[str | int] | None = None,
        spatial_key: str = "spatial",
        **kwargs,
    ) -> None:
        """Register AnnData fields for SPARL inference.

        Parameters
        ----------
        adata
            AnnData with per-cell image paths in ``obs[img_path_col]``
            and spatial coordinates in ``obsm[spatial_key]``.
        img_path_col
            Column in ``adata.obs`` containing per-cell image file paths.
        channel_names
            Ordered list of channel names/IDs matching the backbone's training
            channels. Stored in ``adata.uns["sparl_channels"]``.
        spatial_key
            Key in ``adata.obsm`` for 2D spatial coordinates.
        """
        from scvi.data import AnnDataManager
        from scviva.data._fields import SpatialCoordsField

        fields = [SpatialCoordsField(obsm_key=spatial_key)]
        manager = AnnDataManager(fields=fields)
        manager.register_fields(adata, **kwargs)
        cls.register_manager(manager)

        adata.uns["scviva_imaging"] = {"img_path_col": img_path_col}
        if channel_names is not None:
            adata.uns["sparl_channels"] = list(channel_names)

    def get_latent_representation(
        self,
        adata: AnnData,
        batch_size: int = 256,
        obsm_key: str = "X_sparl",
        device: str = "cpu",
        backend: str = "cpu",
    ) -> "np.ndarray":
        """Run SPARL inference and write embeddings to ``adata.obsm["X_sparl"]``."""
        return super().get_latent_representation(
            adata,
            batch_size=batch_size,
            obsm_key=obsm_key,
            device=device,
            backend=backend,
        )
```

- [ ] **Step 4: Run SPARL model tests — expect PASS**

```bash
pytest tests/imaging/sparl/test_sparl.py::test_sparl_from_pretrained_local \
       tests/imaging/sparl/test_sparl.py::test_sparl_from_pretrained_embed_dim \
       tests/imaging/sparl/test_sparl.py::test_sparl_from_pretrained_hf -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/scviva/imaging/sparl/_model.py tests/imaging/sparl/test_sparl.py
git commit -m "feat(imaging/sparl): SPARL model class with from_pretrained"
```

---

## Task 8: `SPARL` — `setup_anndata` with channel names

**Files:**
- Modify: `tests/imaging/sparl/test_sparl.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/imaging/sparl/test_sparl.py`:

```python
def test_setup_anndata_registers_img_path_col(sparl_imaging_adata):
    SPARL.setup_anndata(
        sparl_imaging_adata,
        img_path_col="img_path",
        channel_names=["ch0", "ch1", "ch2"],
        spatial_key="spatial",
    )
    assert sparl_imaging_adata.uns["scviva_imaging"]["img_path_col"] == "img_path"


def test_setup_anndata_stores_channel_names(sparl_imaging_adata):
    SPARL.setup_anndata(
        sparl_imaging_adata,
        img_path_col="img_path",
        channel_names=["CD3", "CD8", "DAPI"],
        spatial_key="spatial",
    )
    assert sparl_imaging_adata.uns["sparl_channels"] == ["CD3", "CD8", "DAPI"]


def test_setup_anndata_no_channel_names(sparl_imaging_adata):
    SPARL.setup_anndata(sparl_imaging_adata, img_path_col="img_path")
    assert "sparl_channels" not in sparl_imaging_adata.uns
```

- [ ] **Step 2: Run tests — expect PASS** (already implemented in Task 7)

```bash
pytest tests/imaging/sparl/test_sparl.py::test_setup_anndata_registers_img_path_col \
       tests/imaging/sparl/test_sparl.py::test_setup_anndata_stores_channel_names \
       tests/imaging/sparl/test_sparl.py::test_setup_anndata_no_channel_names -v
```

Expected: all 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/imaging/sparl/test_sparl.py
git commit -m "test(imaging/sparl): setup_anndata channel names coverage"
```

---

## Task 9: `SPARL` — end-to-end inference

**Files:**
- Modify: `tests/imaging/sparl/test_sparl.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/imaging/sparl/test_sparl.py`:

```python
def test_sparl_get_latent_representation_shape(sparl_imaging_adata, sparl_checkpoint):
    SPARL.setup_anndata(
        sparl_imaging_adata,
        img_path_col="img_path",
        channel_names=[0, 1, 2],
        spatial_key="spatial",
    )
    model = SPARL.from_pretrained(sparl_checkpoint)
    result = model.get_latent_representation(sparl_imaging_adata)
    assert result.shape == (len(sparl_imaging_adata), 32)  # embed_dim=32


def test_sparl_get_latent_representation_writes_obsm(sparl_imaging_adata, sparl_checkpoint):
    SPARL.setup_anndata(
        sparl_imaging_adata, img_path_col="img_path", channel_names=[0, 1, 2]
    )
    model = SPARL.from_pretrained(sparl_checkpoint)
    model.get_latent_representation(sparl_imaging_adata)
    assert "X_sparl" in sparl_imaging_adata.obsm
    assert sparl_imaging_adata.obsm["X_sparl"].shape == (len(sparl_imaging_adata), 32)


def test_sparl_get_latent_representation_no_nan(sparl_imaging_adata, sparl_checkpoint):
    SPARL.setup_anndata(
        sparl_imaging_adata, img_path_col="img_path", channel_names=[0, 1, 2]
    )
    model = SPARL.from_pretrained(sparl_checkpoint)
    result = model.get_latent_representation(sparl_imaging_adata)
    assert not np.any(np.isnan(result))


def test_sparl_obsm_key_override(sparl_imaging_adata, sparl_checkpoint):
    SPARL.setup_anndata(
        sparl_imaging_adata, img_path_col="img_path", channel_names=[0, 1, 2]
    )
    model = SPARL.from_pretrained(sparl_checkpoint)
    model.get_latent_representation(sparl_imaging_adata, obsm_key="X_custom_sparl")
    assert "X_custom_sparl" in sparl_imaging_adata.obsm
```

- [ ] **Step 2: Run tests — expect FAIL** (dataloader reads 3ch PNG but channels tensor must match backbone's [0,1,2])

```bash
pytest tests/imaging/sparl/test_sparl.py::test_sparl_get_latent_representation_shape -v
```

Expected: FAIL or PASS depending on whether backbone channels match image channels.

> **Note on channels:** `DinoVisionTransformer` validates that the `channels` tensor passed to forward matches `self.channel_names`. `SPARLModule` generates the channels tensor from `self.backbone.channel_names` automatically (it does NOT read channel info from the image file). The user must ensure their images have the same number of channels as the backbone was trained with. If the PNG has 3 channels and `channel_names=[0,1,2]`, this matches.

- [ ] **Step 3: Run all SPARL tests**

```bash
pytest tests/imaging/sparl/test_sparl.py -v
```

Expected: all tests PASS. If the shape test fails due to a channels mismatch, verify:
- `sparl_imaging_adata` fixture creates 3-channel PNGs (RGB mode in Pillow)
- `sparl_checkpoint` has `channel_names=[0, 1, 2]` (3 channels)
- `_load_image_as_tensor` returns shape `(3, 16, 16)` for a 3-channel 16x16 PNG

- [ ] **Step 4: Run full test suite to check for regressions**

```bash
pytest tests/ -v --ignore=tests/imaging/sparl  # non-SPARL suite
pytest tests/imaging/ -v                        # imaging suite
```

Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add tests/imaging/sparl/test_sparl.py
git commit -m "test(imaging/sparl): end-to-end inference coverage"
```

---

## Task 10: Wire up imports and public API

**Files:**
- Modify: `src/scviva/imaging/__init__.py`
- Modify: `src/scviva/imaging/base/__init__.py`
- Modify: `src/scviva/imaging/sparl/__init__.py`
- Modify: `src/scviva/__init__.py`
- Modify: `tests/imaging/base/test_imaging_base.py`

- [ ] **Step 1: Write failing import tests**

Add to `tests/imaging/base/test_imaging_base.py`:

```python
def test_imaging_base_importable_from_imaging():
    from scviva.imaging import ImagingBaseModel
    assert ImagingBaseModel is not None


def test_sparl_importable_from_imaging():
    sparl_pkg = pytest.importorskip("sparl")  # noqa: F841
    from scviva.imaging import SPARL
    assert SPARL is not None


def test_sparl_importable_from_scviva_top_level():
    sparl_pkg = pytest.importorskip("sparl")  # noqa: F841
    import scviva
    SPARL = scviva.SPARL  # via lazy __getattr__
    assert SPARL is not None
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/imaging/base/test_imaging_base.py::test_imaging_base_importable_from_imaging -v
```

Expected: `ImportError` — `__init__.py` files are empty stubs.

- [ ] **Step 3: Fill in `__init__.py` files**

`src/scviva/imaging/base/__init__.py`:
```python
from __future__ import annotations

from scviva.imaging.base._imaging_base import ImagingBaseModel

__all__ = ["ImagingBaseModel"]
```

`src/scviva/imaging/sparl/__init__.py`:
```python
from __future__ import annotations

from scviva.imaging.sparl._model import SPARL

__all__ = ["SPARL"]
```

`src/scviva/imaging/__init__.py`:
```python
"""Image-based spatial models for scviva-tools."""
from __future__ import annotations

from scviva.imaging.base import ImagingBaseModel
from scviva.imaging.sparl import SPARL

__all__ = ["ImagingBaseModel", "SPARL"]
```

- [ ] **Step 4: Update `src/scviva/__init__.py` with lazy import**

Current file:
```python
from __future__ import annotations

from importlib import import_module

from scviva._settings import settings

__version__ = "0.1.0"

_MODEL_NAMES = {"SCVIVA", "DestVI", "ResolVI", "GIMVI"}


def __getattr__(name: str):
    if name in _MODEL_NAMES:
        mod = import_module("scviva.model")
        return getattr(mod, name)
    raise AttributeError(f"module 'scviva' has no attribute {name!r}")


__all__ = ["SCVIVA", "DestVI", "ResolVI", "GIMVI", "__version__", "settings"]
```

Replace with:
```python
from __future__ import annotations

from importlib import import_module

from scviva._settings import settings

__version__ = "0.1.0"

_MODEL_NAMES = {"SCVIVA", "DestVI", "ResolVI", "GIMVI"}
_IMAGING_NAMES = {"SPARL", "ImagingBaseModel"}


def __getattr__(name: str):
    if name in _MODEL_NAMES:
        mod = import_module("scviva.model")
        return getattr(mod, name)
    if name in _IMAGING_NAMES:
        mod = import_module("scviva.imaging")
        return getattr(mod, name)
    raise AttributeError(f"module 'scviva' has no attribute {name!r}")


__all__ = [
    "SCVIVA", "DestVI", "ResolVI", "GIMVI",
    "SPARL", "ImagingBaseModel",
    "__version__", "settings",
]
```

- [ ] **Step 5: Run import tests — expect PASS**

```bash
pytest tests/imaging/base/test_imaging_base.py::test_imaging_base_importable_from_imaging \
       tests/imaging/base/test_imaging_base.py::test_sparl_importable_from_imaging \
       tests/imaging/base/test_imaging_base.py::test_sparl_importable_from_scviva_top_level -v
```

Expected: all 3 PASS (SPARL tests skip if sparl not installed).

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all existing tests pass, all new tests pass (SPARL-dependent tests skip if sparl not installed).

- [ ] **Step 7: Commit**

```bash
git add src/scviva/imaging/__init__.py src/scviva/imaging/base/__init__.py \
        src/scviva/imaging/sparl/__init__.py src/scviva/__init__.py \
        tests/imaging/base/test_imaging_base.py
git commit -m "feat(imaging): wire up public API and lazy imports"
```

---

## Self-Review Checklist

### Spec coverage

| Spec requirement | Task that implements it |
|-----------------|------------------------|
| `imaging/` submodule as peer of `model/` and `external/` | Task 1 |
| `ImagingBaseModel(SpatialBaseModel)` | Task 2 |
| `from_pretrained(local_path)` | Task 2 |
| `from_pretrained(hf_repo_id)` | Task 3 |
| `setup_anndata(img_path_col, spatial_key)` | Task 4 |
| `from_spatialdata` returns AnnData (not model) | Task 4 |
| Inference dataloader (disk paths, no augmentation) | Task 5 |
| `get_latent_representation` writes to `obsm` | Task 5 |
| RAPIDS `backend` parameter | Task 5 (in `get_latent_representation`) |
| `SPARLModule` wrapping `DinoVisionTransformer` | Task 6 |
| `SPARL(ImagingBaseModel)` | Task 7 |
| `SPARL.setup_anndata` with `channel_names` in `uns` | Task 8 |
| End-to-end SPARL inference | Task 9 |
| `from scviva.imaging import SPARL` | Task 10 |
| `scviva.SPARL` top-level lazy import | Task 10 |
| TIFF image loading | Task 5 (in `_load_image_as_tensor`) |
| HF download mocked in CI | Tasks 3, 7 |
| `pytest.importorskip("sparl")` guard | Tasks 6–9 |
| `[imaging]` optional dep group | Task 1 |
| `sparl` NOT in core deps | Task 1 (not added to `[project.dependencies]`) |

All spec requirements covered. ✓

### Implementation notes from spec

The spec notes that `from_spatialdata` needs a light override because `SpatialBaseModel.from_spatialdata` calls `cls(adata)` which doesn't fit `ImagingBaseModel`. This is handled in Task 4 — the override returns `adata` instead of a model instance.
