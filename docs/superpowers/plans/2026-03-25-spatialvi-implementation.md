# spatialvi-tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `spatialvi-tools`, a clean scverse-compatible package consolidating DestVI, ResolVI, and scVIVA spatial transcriptomics models with shared base infrastructure, SpatialData/squidpy/RAPIDS integration.

**Architecture:** All three models inherit from `SpatialBaseModel` (wrapping scvi's `BaseModelClass`) which provides shared spatial setup, latent representation with RAPIDS dispatch, and spatial plotting. Model-specific capabilities (neighbor graphs, deconvolution) live in composable mixins. Every model and its module is ported from scvi-tools with import paths rewritten from `scvi.*` to `spatialvi.*` where applicable.

**Tech Stack:** Python 3.12–3.13, scvi-tools (base classes), torch, pyro-ppl, anndata, scanpy, squidpy, spatialdata, hatchling, ruff, pytest.

**Upstream source:** `/Users/orikr/PycharmProjects/scvi-tools-main-always/`
**Spec:** `docs/superpowers/specs/2026-03-24-spatialvi-design.md`

---

## Phase 0: Scope Note

Tasks 1–9 are sequential (each builds on the last). Tasks 10–15 (the three models + their tests) are **independent of each other** and can run in parallel after Task 9. Tasks 16–17 are post-model cleanup.

---

## Task 1: Package Scaffolding (root-level files)

**Files to create:**
- `pyproject.toml`
- `README.md`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `LICENSE`
- `.gitignore`
- `.pre-commit-config.yaml`
- `.readthedocs.yaml`
- `.editorconfig`
- `.markdownlint.yaml`
- `.codecov.yaml`
- `Dockerfile`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
build-backend = "hatchling.build"
requires = ["hatchling"]

[project]
name = "spatialvi-tools"
version = "0.1.0"
description = "Consolidated spatial transcriptomics analysis toolkit with variational inference methods"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.12"
authors = [{ name = "Ori Kronfeld", email = "ori.kronfeld@weizmann.ac.il" }]
keywords = ["spatial transcriptomics", "variational inference", "deep learning", "single-cell", "deconvolution"]
classifiers = [
  "Development Status :: 4 - Beta",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: BSD License",
  "Natural Language :: English",
  "Operating System :: OS Independent",
  "Programming Language :: Python :: 3 :: Only",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Programming Language :: Python :: 3.14",
  "Topic :: Scientific/Engineering :: Bio-Informatics",
]
dependencies = [
  "anndata>=0.11",
  "lightning>=2",
  "numpy>=1.21",
  "pandas",
  "pyro-ppl",
  "rich",
  "scanpy>=1.10",
  "scipy",
  "scvi-tools",
  "torch",
  "tqdm",
]

[project.optional-dependencies]
spatial = ["spatialdata>=0.2", "squidpy>=1.4"]
rapids   = ["cuml>=24.0", "cugraph>=24.0", "cupy-cuda12x"]
dev      = ["pre-commit", "twine>=4.0.2"]
test     = ["pytest>=7", "pytest-cov", "anndata", "scanpy"]
doc      = ["sphinx", "myst-nb", "sphinx-book-theme", "ipykernel"]
all      = ["spatialvi-tools[spatial,rapids,dev,doc,test]"]

[tool.hatch.build.targets.wheel]
packages = ["src/spatialvi"]

[tool.ruff]
src = ["src"]
line-length = 99
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
  - repo: https://github.com/codespell-project/codespell
    rev: v2.3.0
    hooks:
      - id: codespell
        args: ["--ignore-words=.codespell-ignore-words"]
```

- [ ] **Step 3: Create `.readthedocs.yaml`**

```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.13"
sphinx:
  configuration: docs/conf.py
python:
  install:
    - method: pip
      path: .
      extra_requirements: [doc]
```

- [ ] **Step 4: Create `.codecov.yaml`**

```yaml
coverage:
  status:
    project:
      default:
        target: 80%
        threshold: 2%
```

- [ ] **Step 5: Create `.editorconfig`**

```ini
root = true
[*]
indent_style = space
indent_size = 4
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true
[*.{yaml,yml,json,toml}]
indent_size = 2
```

- [ ] **Step 6: Create `LICENSE`**

Copy BSD 3-Clause license text (same as scvi-tools), replacing author with "Ori Kronfeld, Weizmann Institute".

- [ ] **Step 7: Create skeleton `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`**

`README.md` — one-paragraph description + install instructions:
```markdown
# spatialvi-tools

Consolidated spatial transcriptomics analysis toolkit based on scvi-tools.
Models: DestVI, ResolVI, scVIVA.

## Installation
```bash
pip install spatialvi-tools
pip install "spatialvi-tools[spatial]"  # + SpatialData/squidpy
```
```

`CHANGELOG.md` — standard format:
```markdown
# Changelog
## [0.1.0] - 2026-03-25
### Added
- Initial package with DestVI, ResolVI, scVIVA models
- SpatialBaseModel shared infrastructure
- SpatialData and squidpy integration
- RAPIDS acceleration backend
```

- [ ] **Step 8: Create `Dockerfile`** (copy from scvi-tools, update package name)

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml README.md CHANGELOG.md CONTRIBUTING.md LICENSE \
        .gitignore .pre-commit-config.yaml .readthedocs.yaml \
        .editorconfig .markdownlint.yaml .codecov.yaml Dockerfile
git commit -m "feat: add package scaffolding (pyproject, CI config, aux files)"
```

---

## Task 2: GitHub Workflows and Issue Templates

**Files to create:**
- `.github/workflows/build.yml`
- `.github/workflows/test_linux.yml`
- `.github/workflows/test_macos.yml`
- `.github/workflows/test_gpu.yml`
- `.github/workflows/release.yml`
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/ISSUE_TEMPLATE/config.yml`
- `.github/pull_request_template.md`

- [ ] **Step 1: Create `build.yml`**

```yaml
name: Build
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13" }
      - run: pip install hatchling
      - run: python -m hatchling build
```

- [ ] **Step 2: Create `test_linux.yml`**

```yaml
name: Test Linux
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    # continue-on-error must be at job level (not inside matrix.include) to take effect
    continue-on-error: ${{ matrix.python-version == '3.14' }}
    strategy:
      matrix:
        python-version: ["3.12", "3.13", "3.14"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[test,spatial]"
      - run: pytest tests/ -v --cov=spatialvi --cov-report=xml
      - uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.13'
```

- [ ] **Step 3: Create `test_macos.yml`**

```yaml
name: Test macOS
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13" }
      - run: pip install -e ".[test,spatial]"
      - run: pytest tests/ -v
```

- [ ] **Step 4: Create `test_gpu.yml`**

```yaml
name: Test GPU
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  test-gpu:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13" }
      - run: pip install -e ".[test,spatial,rapids]"
      - run: pytest tests/ -v -m "gpu"
```

- [ ] **Step 5: Create `release.yml`** (standard PyPI release on tag push `v*`)

Use the standard pattern: checkout → build → `twine upload`.

- [ ] **Step 6: Create issue templates** (copy from scvi-tools, replace `scvi` with `spatialvi`)

- [ ] **Step 7: Commit**

```bash
git add .github/
git commit -m "feat: add GitHub Actions workflows and issue templates"
```

---

## Task 3: Package Skeleton (`__init__` files, constants, settings)

**Files to create:**
- `src/spatialvi/__init__.py`
- `src/spatialvi/_constants.py`
- `src/spatialvi/_settings.py`
- `src/spatialvi/model/__init__.py`
- `src/spatialvi/model/base/__init__.py`
- `src/spatialvi/module/__init__.py`
- `src/spatialvi/external/__init__.py`
- `src/spatialvi/data/__init__.py`
- `src/spatialvi/utils/__init__.py`
- `src/spatialvi/train/__init__.py`
- `src/spatialvi/train/_config.py`
- `tests/__init__.py`

- [ ] **Step 1: Create `src/spatialvi/__init__.py`** with lazy imports

```python
from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

_lazy_map = {
    "SCVIVA": "spatialvi.model._scviva",
    "DestVI": "spatialvi.model._destvi",
    "ResolVI": "spatialvi.model._resolvi",
}


def __getattr__(name: str):
    if name in _lazy_map:
        mod = import_module(_lazy_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'spatialvi' has no attribute {name!r}")


__all__ = ["SCVIVA", "DestVI", "ResolVI", "__version__"]
```

- [ ] **Step 2: Create `src/spatialvi/_constants.py`**

Port from `scvi-tools-main-always/src/scvi/_constants.py`.
Keep `REGISTRY_KEYS` class. Add spatial-specific keys:

```python
from scvi import REGISTRY_KEYS as _SCVI_REGISTRY_KEYS

# Re-export all scvi registry keys so spatialvi code can import from one place
REGISTRY_KEYS = _SCVI_REGISTRY_KEYS

# Spatial-specific keys used across models
SPATIAL_COORDS_KEY = "spatial_coords"
NEIGHBOR_INDEX_KEY = "index_neighbor"
NEIGHBOR_DISTANCE_KEY = "distance_neighbor"
NICHE_COMPOSITION_KEY = "neighborhood_composition"
```

- [ ] **Step 3: Create `src/spatialvi/_settings.py`**

Port from `scvi-tools-main-always/src/scvi/_settings.py`. Replace `scvi` references with `spatialvi`.

- [ ] **Step 4: Create `src/spatialvi/model/__init__.py`** as a deferred stub

```python
# src/spatialvi/model/__init__.py
# Model imports are deferred to avoid ImportError while model files are being built.
# Populated fully in Task 20 (final integration).
from importlib import import_module

_lazy_model_map = {
    "SCVIVA": "spatialvi.model._scviva",
    "DestVI": "spatialvi.model._destvi",
    "ResolVI": "spatialvi.model._resolvi",
}


def __getattr__(name: str):
    if name in _lazy_model_map:
        mod = import_module(_lazy_model_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'spatialvi.model' has no attribute {name!r}")


__all__ = ["SCVIVA", "DestVI", "ResolVI"]
```

> **Why lazy here:** Eager imports (`from ._scviva import SCVIVA`) would break `import spatialvi.model` in Tasks 3–9 before the model files exist. Lazy imports allow the package to be installed and importable throughout the build process.

- [ ] **Step 5: Create all remaining `__init__.py` files as empty stubs**

```python
# src/spatialvi/model/base/__init__.py — to be filled in Task 5
# src/spatialvi/module/__init__.py     — to be filled in Tasks 10/13/16
# src/spatialvi/external/__init__.py   — intentionally empty
# src/spatialvi/data/__init__.py       — to be filled in Task 4
# src/spatialvi/utils/__init__.py      — to be filled here
# src/spatialvi/train/__init__.py      — to be filled here
```

`utils/__init__.py`:
```python
from ._spatial import get_spatial_coords, validate_spatial_coords

__all__ = ["get_spatial_coords", "validate_spatial_coords"]
```

`train/__init__.py` + `train/_config.py`:
```python
# train/_config.py — re-export scvi training configs for now
from scvi.train._config import TrainingPlanConfig, TrainerConfig

__all__ = ["TrainingPlanConfig", "TrainerConfig"]
```

- [ ] **Step 5b: Create `tests/base/__init__.py`, `tests/utils/__init__.py`, `tests/model/__init__.py`**

These empty files are required because `tests/__init__.py` exists — pytest uses package layout which requires `__init__.py` at each subdirectory to avoid import collisions and conftest scoping issues.

```bash
mkdir -p tests/base tests/utils tests/model
touch tests/base/__init__.py tests/utils/__init__.py tests/model/__init__.py
```

- [ ] **Step 5c: Create `tests/conftest.py`** with the shared `_MinimalSpatialModel` fixture

```python
# tests/conftest.py
"""Shared pytest fixtures for spatialvi test suite."""
from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from scvi.model.base import UnsupervisedTrainingMixin

from spatialvi.data._fields import SpatialCoordsField
from spatialvi.model.base._spatial_base import SpatialBaseModel


class _MinimalSpatialModel(SpatialBaseModel, UnsupervisedTrainingMixin):
    """Minimal concrete model for testing SpatialBaseModel methods in isolation.

    Cannot be trained (no real module.inference). Used only for testing
    field registration, setup_spatialdata, and plotting helpers.
    """

    @classmethod
    def setup_anndata(cls, adata, layer=None, spatial_key="spatial", **kwargs):
        fields = [
            LayerField("X", layer, is_count_data=True),
            SpatialCoordsField(obsm_key=spatial_key),
        ]
        manager = AnnDataManager(fields=fields)
        manager.register_fields(adata, **kwargs)
        cls.register_manager(manager)


def make_spatial_adata(n: int = 80, n_genes: int = 20) -> AnnData:
    """Create a minimal AnnData with counts layer and spatial coordinates."""
    adata = AnnData(X=np.abs(np.random.rand(n, n_genes)))
    adata.layers["counts"] = np.abs(np.random.poisson(3, size=(n, n_genes)))
    adata.obsm["spatial"] = np.random.rand(n, 2)
    return adata


@pytest.fixture(scope="module")
def minimal_spatial_adata():
    return make_spatial_adata()


@pytest.fixture(scope="module")
def minimal_model(minimal_spatial_adata):
    _MinimalSpatialModel.setup_anndata(
        minimal_spatial_adata, layer="counts", spatial_key="spatial"
    )
    return _MinimalSpatialModel(minimal_spatial_adata)
```

- [ ] **Step 6: Verify import works**

```bash
cd /Users/orikr/PycharmProjects/spatialvi-tools2
pip install -e ".[test]"
python -c "import spatialvi; print(spatialvi.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 7: Commit**

```bash
git add src/ tests/__init__.py
git commit -m "feat: add package skeleton with lazy imports, constants, settings"
```

---

## Task 4: Custom AnnData Fields (`data/_fields.py`)

**Files:**
- Create: `src/spatialvi/data/_fields.py`
- Modify: `src/spatialvi/data/__init__.py`
- Test: `tests/base/test_fields.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/base/test_fields.py
import numpy as np
import pytest
from anndata import AnnData

from spatialvi.data._fields import NeighborhoodGraphField, SpatialCoordsField


def _make_adata(coords_2d=True):
    n = 50
    adata = AnnData(X=np.random.rand(n, 20))
    if coords_2d:
        adata.obsm["spatial"] = np.random.rand(n, 2)
    else:
        adata.obsm["spatial"] = np.random.rand(n, 3)
    return adata


def test_spatial_coords_field_2d():
    adata = _make_adata(coords_2d=True)
    field = SpatialCoordsField(obsm_key="spatial")
    data = field.get_field_data(adata)
    assert data.shape == (50, 2)


def test_spatial_coords_field_3d():
    adata = _make_adata(coords_2d=False)
    field = SpatialCoordsField(obsm_key="spatial")
    data = field.get_field_data(adata)
    assert data.shape == (50, 3)


def test_spatial_coords_field_invalid_dim():
    adata = _make_adata()
    adata.obsm["spatial"] = np.random.rand(50, 5)  # invalid
    field = SpatialCoordsField(obsm_key="spatial")
    with pytest.raises(ValueError, match="2D or 3D"):
        field.get_field_data(adata)


def test_neighborhood_graph_field_converts_sparse():
    from scipy.sparse import csr_matrix
    adata = _make_adata()
    sparse_idx = csr_matrix(np.random.randint(0, 50, size=(50, 6)))
    adata.obsm["index_neighbor"] = sparse_idx
    field = NeighborhoodGraphField(obsm_key="index_neighbor")
    data = field.get_field_data(adata)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 6)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/base/test_fields.py -v
```

Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `src/spatialvi/data/_fields.py`**

```python
from __future__ import annotations

import numpy as np
from scvi.data.fields import ObsmField


class SpatialCoordsField(ObsmField):
    """AnnData obsm field for 2D/3D spatial coordinates.

    Wraps scvi's ObsmField with coordinate dimensionality validation.
    Stores coordinates as-is (no normalization) — callers may normalize
    before registration if needed.
    """

    def get_field_data(self, adata):
        data = super().get_field_data(adata)
        if data.shape[1] not in (2, 3):
            raise ValueError(
                f"Spatial coordinates must be 2D or 3D, got shape {data.shape}. "
                f"Expected obsm['{self.attr_key}'] with 2 or 3 columns."
            )
        return data


class NeighborhoodGraphField(ObsmField):
    """AnnData obsm field for neighbor index/distance arrays.

    Stores dense numpy arrays. Squidpy outputs sparse CSR matrices —
    this field converts them to dense on registration (documented behavior).
    Keys: ``index_neighbor`` and ``distance_neighbor`` (matching upstream ResolVI).
    """

    def get_field_data(self, adata):
        data = super().get_field_data(adata)
        # Squidpy returns CSR; convert to dense for AnnDataManager compatibility
        if hasattr(data, "toarray"):
            data = data.toarray()
        return np.asarray(data)
```

- [ ] **Step 4: Update `src/spatialvi/data/__init__.py`**

```python
from ._fields import NeighborhoodGraphField, SpatialCoordsField

__all__ = ["SpatialCoordsField", "NeighborhoodGraphField"]
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/base/test_fields.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/spatialvi/data/ tests/base/test_fields.py
git commit -m "feat: add SpatialCoordsField and NeighborhoodGraphField"
```

---

## Task 5: `SpatialBaseModel`

**Files:**
- Create: `src/spatialvi/model/base/_spatial_base.py`
- Modify: `src/spatialvi/model/base/__init__.py`
- Test: `tests/base/test_spatial_base.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/base/test_spatial_base.py
import numpy as np
import pytest
from anndata import AnnData
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from scvi.model.base import UnsupervisedTrainingMixin

from spatialvi.data._fields import SpatialCoordsField
from spatialvi.model.base._spatial_base import SpatialBaseModel


# Minimal concrete subclass — SpatialBaseModel cannot be instantiated directly
class _MinimalSpatialModel(SpatialBaseModel, UnsupervisedTrainingMixin):
    @classmethod
    def setup_anndata(cls, adata, layer=None, spatial_key="spatial", **kwargs):
        fields = [
            LayerField("X", layer, is_count_data=True),
            SpatialCoordsField(obsm_key=spatial_key),
        ]
        manager = AnnDataManager(fields=fields)
        manager.register_fields(adata, **kwargs)
        cls.register_manager(manager)


def _make_spatial_adata(n=80, n_genes=20):
    adata = AnnData(X=np.abs(np.random.rand(n, n_genes)))
    adata.layers["counts"] = np.abs(np.random.poisson(3, size=(n, n_genes)))
    adata.obsm["spatial"] = np.random.rand(n, 2)
    return adata


@pytest.fixture(scope="module")
def minimal_model():
    adata = _make_spatial_adata()
    _MinimalSpatialModel.setup_anndata(adata, layer="counts", spatial_key="spatial")
    return _MinimalSpatialModel(adata)


def test_setup_anndata_registers_spatial(minimal_model):
    mgr = minimal_model.adata_manager
    assert "spatial" in [f.attr_key for f in mgr.fields]


def test_setup_spatialdata_requires_spatialdata(minimal_model):
    """setup_spatialdata must raise ImportError if spatialdata is not installed
    or TypeError if given a non-SpatialData object."""
    with pytest.raises((ImportError, TypeError)):
        _MinimalSpatialModel.setup_spatialdata(object(), table_key="table", region="cells")


def test_get_latent_cpu_not_implemented_on_base(minimal_model):
    """SpatialBaseModel.get_latent_representation calls super() which raises
    NotImplementedError on the minimal model (no module.inference)."""
    with pytest.raises((NotImplementedError, AttributeError)):
        minimal_model.get_latent_representation()
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/base/test_spatial_base.py -v
```

- [ ] **Step 3: Implement `src/spatialvi/model/base/_spatial_base.py`**

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scvi.model.base import BaseModelClass

if TYPE_CHECKING:
    from typing import Literal
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SpatialBaseModel(BaseModelClass):
    """Base class for all spatialvi models.

    Extends scvi's BaseModelClass with:
    - SpatialData integration (setup_spatialdata / from_spatialdata)
    - RAPIDS-accelerated latent representation
    - Spatial embedding and prediction plots

    All spatialvi core models (SCVIVA, DestVI, ResolVI) inherit from this class.
    """

    # ------------------------------------------------------------------ #
    # SpatialData integration
    # ------------------------------------------------------------------ #

    @classmethod
    def setup_spatialdata(
        cls,
        sdata,
        table_key: str = "table",
        region: str | None = None,
        **kwargs,
    ) -> None:
        """Register fields from a SpatialData object.

        Extracts the AnnData table at ``sdata[table_key]`` and calls
        :meth:`setup_anndata`. Follows the same classmethod convention as
        :meth:`setup_anndata` — call this before constructing the model.

        Parameters
        ----------
        sdata
            A :class:`spatialdata.SpatialData` object.
        table_key
            Key in ``sdata`` pointing to the AnnData table.
        region
            Region name to subset (stored in ``sdata[table_key].obs``).
            If None, the full table is used.
        **kwargs
            Passed to :meth:`setup_anndata`.
        """
        try:
            import spatialdata  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "spatialdata is required for setup_spatialdata. "
                "Install with: pip install 'spatialvi-tools[spatial]'"
            ) from e

        if not hasattr(sdata, "__getitem__"):
            raise TypeError(
                f"Expected a SpatialData object, got {type(sdata).__name__}. "
                "Install spatialdata or pass an AnnData to setup_anndata."
            )

        adata = sdata[table_key]
        if region is not None:
            region_key = adata.uns.get("spatialdata_attrs", {}).get("region_key", "region")
            adata = adata[adata.obs[region_key] == region].copy()

        cls.setup_anndata(adata, **kwargs)

    @classmethod
    def from_spatialdata(
        cls,
        sdata,
        table_key: str = "table",
        region: str | None = None,
        **model_kwargs,
    ):
        """Convenience constructor from a SpatialData object.

        Calls :meth:`setup_spatialdata` then constructs and returns the model.

        Parameters
        ----------
        sdata
            A :class:`spatialdata.SpatialData` object.
        table_key
            Key in ``sdata`` pointing to the AnnData table.
        region
            Region name to subset.
        **model_kwargs
            Passed to the model constructor.

        Returns
        -------
        Instantiated model.
        """
        cls.setup_spatialdata(sdata, table_key=table_key, region=region)
        adata = sdata[table_key]
        if region is not None:
            region_key = adata.uns.get("spatialdata_attrs", {}).get("region_key", "region")
            adata = adata[adata.obs[region_key] == region].copy()
        return cls(adata, **model_kwargs)

    # ------------------------------------------------------------------ #
    # Latent representation with RAPIDS dispatch
    # ------------------------------------------------------------------ #

    def get_latent_representation(
        self,
        adata=None,
        indices=None,
        give_mean: bool = True,
        batch_size: int | None = None,
        backend: Literal["cpu", "rapids"] = "cpu",
        **kwargs,
    ) -> np.ndarray:
        """Return latent representation with optional RAPIDS acceleration.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the model's registered adata.
        indices
            Cell indices to use. If None, all cells are used.
        give_mean
            Return distribution mean rather than a sample.
        batch_size
            Mini-batch size.
        backend
            ``"cpu"`` (default) returns a numpy array as normal.
            ``"rapids"`` transfers the result to a cupy array for downstream
            GPU-accelerated UMAP/clustering (requires ``pip install cuml``).
        **kwargs
            Forwarded to the parent ``get_latent_representation``.

        Returns
        -------
        Latent representation as a numpy array (cpu) or cupy array (rapids).
        """
        latent = super().get_latent_representation(
            adata=adata,
            indices=indices,
            give_mean=give_mean,
            batch_size=batch_size,
            **kwargs,
        )
        if backend == "rapids":
            try:
                import cupy as cp
                return cp.asarray(latent)
            except ImportError as e:
                raise ImportError(
                    "backend='rapids' requires cupy. "
                    "Install with: pip install 'spatialvi-tools[rapids]'"
                ) from e
        return latent

    # ------------------------------------------------------------------ #
    # Spatial visualisation
    # ------------------------------------------------------------------ #

    def plot_spatial_embedding(
        self,
        adata=None,
        basis: str = "spatial",
        color: str | list[str] | None = None,
        **kwargs,
    ) -> plt.Figure | None:
        """Plot latent embedding overlaid on tissue spatial coordinates.

        A thin wrapper around :func:`scanpy.pl.embedding` that defaults
        ``basis`` to the spatial coordinate key so cells are displayed in
        tissue space.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the model's registered adata.
        basis
            Key in ``adata.obsm`` containing 2D spatial coordinates.
        color
            Keys to color cells by (obs columns, gene names, etc.).
        **kwargs
            Forwarded to :func:`scanpy.pl.embedding`.
        """
        import scanpy as sc

        adata = self._validate_anndata(adata)
        return sc.pl.embedding(adata, basis=basis, color=color, **kwargs)

    def plot_spatial_predictions(
        self,
        adata=None,
        key: str | None = None,
        basis: str = "spatial",
        **kwargs,
    ) -> plt.Figure | None:
        """Plot any obsm/obs key overlaid on tissue coordinates.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the model's registered adata.
        key
            obs or obsm key to visualize (e.g., latent cluster, cell type).
        basis
            Spatial coordinate key.
        **kwargs
            Forwarded to :func:`scanpy.pl.embedding`.
        """
        return self.plot_spatial_embedding(adata=adata, basis=basis, color=key, **kwargs)
```

- [ ] **Step 4: Update `src/spatialvi/model/base/__init__.py`**

Only export `SpatialBaseModel` now. Mixin exports are added in Tasks 6 and 7 once those files exist.

```python
from ._spatial_base import SpatialBaseModel

__all__ = ["SpatialBaseModel"]
```

> **Important:** Do NOT import `SpatialNeighborhoodMixin` or `SpatialDeconvolutionMixin` here yet — those files don't exist until Tasks 6 and 7. Add them after each mixin task completes to avoid broken intermediate imports.

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/base/test_spatial_base.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/spatialvi/model/base/ tests/base/test_spatial_base.py
git commit -m "feat: add SpatialBaseModel with SpatialData integration and RAPIDS dispatch"
```

---

## Task 6: `SpatialNeighborhoodMixin`

**Files:**
- Create: `src/spatialvi/model/base/_neighborhood_mixin.py`
- Test: `tests/base/test_neighborhood_mixin.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/base/test_neighborhood_mixin.py
import numpy as np
import pytest
from anndata import AnnData
from scvi.model.base import UnsupervisedTrainingMixin

from spatialvi.model.base._neighborhood_mixin import SpatialNeighborhoodMixin
from spatialvi.model.base._spatial_base import SpatialBaseModel


def _make_coords_adata(n=100, n_genes=20):
    adata = AnnData(X=np.abs(np.random.rand(n, n_genes)))
    adata.layers["counts"] = np.abs(np.random.poisson(3, size=(n, n_genes)))
    adata.obsm["spatial"] = np.random.rand(n, 2)
    return adata


def test_compute_neighbors_squidpy_adds_obsm():
    pytest.importorskip("squidpy")
    adata = _make_coords_adata()

    class _M(SpatialNeighborhoodMixin, SpatialBaseModel, UnsupervisedTrainingMixin):
        @classmethod
        def setup_anndata(cls, adata, **kwargs):
            from scvi.data import AnnDataManager
            from scvi.data.fields import LayerField
            mgr = AnnDataManager(fields=[LayerField("X", "counts", is_count_data=True)])
            mgr.register_fields(adata)
            cls.register_manager(mgr)

    _M.setup_anndata(adata)
    model = _M(adata)
    model.compute_neighbors(adata, coord_type="generic", n_neighs=6, backend="squidpy")
    assert "index_neighbor" in adata.obsm
    assert "distance_neighbor" in adata.obsm
    assert adata.obsm["index_neighbor"].shape == (100, 6)
    # Guard against silent all-zero failure (wrong squidpy obsp key suffix)
    assert adata.obsm["index_neighbor"].sum() > 0, (
        "index_neighbor is all zeros — squidpy obsp key suffix may have changed. "
        "Check sq.gr.spatial_neighbors key_added convention for installed squidpy version."
    )


def test_compute_neighbors_invalid_backend():
    adata = _make_coords_adata()
    mixin = SpatialNeighborhoodMixin()
    with pytest.raises(ValueError, match="backend"):
        mixin.compute_neighbors(adata, backend="unknown")
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/base/test_neighborhood_mixin.py -v
```

- [ ] **Step 3: Implement `src/spatialvi/model/base/_neighborhood_mixin.py`**

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)

_VALID_BACKENDS = ("squidpy", "rapids")


class SpatialNeighborhoodMixin:
    """Mixin for spatial neighbor graph computation.

    Applied to: SCVIVA, ResolVI.
    Provides a single entry point for neighbor graph computation with
    pluggable backends (squidpy on CPU, RAPIDS on GPU).

    The computed neighbor arrays are stored in:
    - ``adata.obsm["index_neighbor"]``   — dense int array, shape (n_cells, n_neighs)
    - ``adata.obsm["distance_neighbor"]``— dense float array, shape (n_cells, n_neighs)

    These keys match the upstream ResolVI module's expected input format.
    """

    def compute_neighbors(
        self,
        adata: AnnData,
        spatial_key: str = "spatial",
        coord_type: str = "generic",
        n_neighs: int = 6,
        backend: Literal["squidpy", "rapids"] = "squidpy",
    ) -> None:
        """Compute spatial neighbor graph and store in ``adata.obsm``.

        Parameters
        ----------
        adata
            AnnData object with spatial coordinates in ``adata.obsm[spatial_key]``.
        spatial_key
            Key in ``adata.obsm`` for spatial coordinates.
        coord_type
            Coordinate type passed to squidpy (``"generic"`` or ``"visium"``).
            Ignored when ``backend="rapids"``.
        n_neighs
            Number of nearest neighbors.
        backend
            ``"squidpy"`` (default): uses :func:`squidpy.gr.spatial_neighbors`.
            ``"rapids"``: uses cuGraph/cuML for GPU-accelerated computation.
        """
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {_VALID_BACKENDS}, got '{backend}'."
            )
        if backend == "squidpy":
            self._compute_neighbors_squidpy(adata, spatial_key, coord_type, n_neighs)
        else:
            self._compute_neighbors_rapids(adata, spatial_key, n_neighs)

    def _compute_neighbors_squidpy(
        self,
        adata: AnnData,
        spatial_key: str,
        coord_type: str,
        n_neighs: int,
    ) -> None:
        try:
            import squidpy as sq
        except ImportError as e:
            raise ImportError(
                "squidpy is required for backend='squidpy'. "
                "Install with: pip install 'spatialvi-tools[spatial]'"
            ) from e

        sq.gr.spatial_neighbors(
            adata,
            spatial_key=spatial_key,
            coord_type=coord_type,
            n_neighs=n_neighs,
            key_added="spatial_neighbors",
        )
        # squidpy stores distances in obsp; extract to obsm in ResolVI's expected format
        import scipy.sparse as sp

        conn = adata.obsp.get("spatial_neighbors_connectivities")
        dist = adata.obsp.get("spatial_neighbors_distances")

        n = adata.n_obs
        idx = np.zeros((n, n_neighs), dtype=np.int64)
        dst = np.zeros((n, n_neighs), dtype=np.float32)

        if conn is not None:
            cx = sp.csr_matrix(conn)
            for i in range(n):
                row = cx[i].indices
                d_row = (
                    sp.csr_matrix(dist)[i].data
                    if dist is not None
                    else np.ones(len(row), dtype=np.float32)
                )
                k = min(n_neighs, len(row))
                idx[i, :k] = row[:k]
                dst[i, :k] = d_row[:k]

        adata.obsm["index_neighbor"] = idx
        adata.obsm["distance_neighbor"] = dst
        logger.info("Computed %d spatial neighbors (squidpy backend).", n_neighs)

    def _compute_neighbors_rapids(
        self,
        adata: AnnData,
        spatial_key: str,
        n_neighs: int,
    ) -> None:
        try:
            import cuml
            import cupy as cp
        except ImportError as e:
            raise ImportError(
                "backend='rapids' requires cuml and cupy. "
                "Install with: pip install 'spatialvi-tools[rapids]'"
            ) from e

        coords = cp.asarray(adata.obsm[spatial_key].astype(np.float32))
        nn = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighs + 1)
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        # drop self (index 0)
        adata.obsm["index_neighbor"] = cp.asnumpy(indices[:, 1:]).astype(np.int64)
        adata.obsm["distance_neighbor"] = cp.asnumpy(distances[:, 1:]).astype(np.float32)
        logger.info("Computed %d spatial neighbors (RAPIDS backend).", n_neighs)

    def _setup_neighbor_field(self, adata: AnnData) -> None:
        """Register neighbor obsm fields in the AnnDataManager.

        Call this inside setup_anndata after computing neighbors, so
        the AnnDataManager tracks both ``index_neighbor`` and ``distance_neighbor``.
        """
        from spatialvi.data._fields import NeighborhoodGraphField

        for key in ("index_neighbor", "distance_neighbor"):
            if key not in adata.obsm:
                raise KeyError(
                    f"'{key}' not found in adata.obsm. "
                    "Call model.compute_neighbors(adata) before setup_anndata."
                )
        # Fields are registered by the model's own AnnDataManager setup.
        # This method exists as a hook for models to call in setup_anndata.
        return [
            NeighborhoodGraphField(obsm_key="index_neighbor"),
            NeighborhoodGraphField(obsm_key="distance_neighbor"),
        ]
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/base/test_neighborhood_mixin.py -v
```

- [ ] **Step 5: Update `src/spatialvi/model/base/__init__.py`** — add `SpatialNeighborhoodMixin`

```python
from ._neighborhood_mixin import SpatialNeighborhoodMixin
from ._spatial_base import SpatialBaseModel

__all__ = ["SpatialBaseModel", "SpatialNeighborhoodMixin"]
```

- [ ] **Step 6: Commit**

```bash
git add src/spatialvi/model/base/_neighborhood_mixin.py src/spatialvi/model/base/__init__.py \
        tests/base/test_neighborhood_mixin.py
git commit -m "feat: add SpatialNeighborhoodMixin with squidpy and RAPIDS backends"
```

---

## Task 7: `SpatialDeconvolutionMixin`

**Files:**
- Create: `src/spatialvi/model/base/_deconvolution_mixin.py`
- Test: `tests/base/test_deconvolution_mixin.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/base/test_deconvolution_mixin.py
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from spatialvi.model.base._deconvolution_mixin import SpatialDeconvolutionMixin


class _MockDeconvModel(SpatialDeconvolutionMixin):
    """Mock model that returns fake proportions (shape: n_spots x n_cell_types)."""
    cell_type_mapping = np.array(["CellA", "CellB", "CellC"])

    def get_proportions(self, adata=None):
        n = adata.n_obs if adata is not None else 20
        props = np.random.dirichlet(np.ones(3), size=n)
        return props


def test_get_proportions_df_shape():
    adata = AnnData(X=np.random.rand(20, 10))
    model = _MockDeconvModel()
    df = model.get_proportions_df(adata)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (20, 3)
    assert list(df.columns) == ["CellA", "CellB", "CellC"]


def test_get_proportions_df_sums_to_one():
    adata = AnnData(X=np.random.rand(20, 10))
    model = _MockDeconvModel()
    df = model.get_proportions_df(adata)
    np.testing.assert_allclose(df.sum(axis=1).values, 1.0, atol=1e-5)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/base/test_deconvolution_mixin.py -v
```

- [ ] **Step 3: Implement `src/spatialvi/model/base/_deconvolution_mixin.py`**

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SpatialDeconvolutionMixin:
    """Mixin for spatial deconvolution result formatting and visualization.

    Applied to: DestVI only.

    Requires the model to implement:
    - ``self.cell_type_mapping``: np.ndarray of cell type label strings
    - ``self.get_proportions(adata)``: returns np.ndarray of shape (n_spots, n_cell_types)
    """

    def get_proportions_df(self, adata: AnnData | None = None) -> pd.DataFrame:
        """Return cell type proportions as a tidy DataFrame.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the model's registered adata.

        Returns
        -------
        DataFrame of shape (n_spots, n_cell_types) with cell type names as columns.
        Rows sum to 1.
        """
        if adata is None and hasattr(self, "adata"):
            adata = self.adata

        proportions = self.get_proportions(adata)  # (n_spots, n_cell_types)
        cell_types = self.cell_type_mapping

        df = pd.DataFrame(proportions, columns=cell_types)
        if adata is not None and adata.obs_names is not None:
            df.index = adata.obs_names
        return df

    def plot_cell_type_map(
        self,
        adata: AnnData | None = None,
        cell_type: str | None = None,
        basis: str = "spatial",
        ax=None,
        **kwargs,
    ) -> plt.Axes | None:
        """Plot spatial map of a single cell type's proportion.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the model's registered adata.
        cell_type
            Name of the cell type to visualize. Must be in ``self.cell_type_mapping``.
        basis
            Key in ``adata.obsm`` for spatial coordinates.
        ax
            Matplotlib axes. If None, a new figure is created.
        **kwargs
            Forwarded to :func:`scanpy.pl.embedding`.
        """
        import scanpy as sc

        if adata is None and hasattr(self, "adata"):
            adata = self.adata

        df = self.get_proportions_df(adata)
        if cell_type is not None:
            if cell_type not in df.columns:
                raise ValueError(
                    f"cell_type '{cell_type}' not found. "
                    f"Available: {list(df.columns)}"
                )
            key = f"_spatialvi_prop_{cell_type}"
            adata.obs[key] = df[cell_type].values
            return sc.pl.embedding(adata, basis=basis, color=key, ax=ax, **kwargs)

        # No cell_type specified — plot all as a grid
        logger.info("No cell_type specified; plotting all %d cell types.", len(df.columns))
        return sc.pl.embedding(adata, basis=basis, color=list(df.columns), **kwargs)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/base/test_deconvolution_mixin.py -v
```

- [ ] **Step 5: Update `src/spatialvi/model/base/__init__.py`** — add `SpatialDeconvolutionMixin`

```python
from ._deconvolution_mixin import SpatialDeconvolutionMixin
from ._neighborhood_mixin import SpatialNeighborhoodMixin
from ._spatial_base import SpatialBaseModel

__all__ = ["SpatialBaseModel", "SpatialNeighborhoodMixin", "SpatialDeconvolutionMixin"]
```

- [ ] **Step 6: Commit**

```bash
git add src/spatialvi/model/base/_deconvolution_mixin.py src/spatialvi/model/base/__init__.py \
        tests/base/test_deconvolution_mixin.py
git commit -m "feat: add SpatialDeconvolutionMixin with get_proportions_df and plot_cell_type_map"
```

---

## Task 8: `utils/_spatial.py`

**Files:**
- Create: `src/spatialvi/utils/_spatial.py`
- Test: `tests/utils/test_spatial_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/utils/test_spatial_utils.py
import numpy as np
import pytest
from anndata import AnnData

from spatialvi.utils._spatial import get_spatial_coords, validate_spatial_coords


def test_get_spatial_coords_2d():
    adata = AnnData(X=np.random.rand(10, 5))
    adata.obsm["spatial"] = np.random.rand(10, 2)
    coords = get_spatial_coords(adata, key="spatial")
    assert coords.shape == (10, 2)


def test_get_spatial_coords_missing_key():
    adata = AnnData(X=np.random.rand(10, 5))
    with pytest.raises(KeyError):
        get_spatial_coords(adata, key="missing")


def test_validate_spatial_coords_valid():
    coords = np.random.rand(20, 2)
    validate_spatial_coords(coords)  # should not raise


def test_validate_spatial_coords_invalid():
    coords = np.random.rand(20, 5)
    with pytest.raises(ValueError):
        validate_spatial_coords(coords)
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `src/spatialvi/utils/_spatial.py`**

```python
from __future__ import annotations

import numpy as np
from anndata import AnnData


def get_spatial_coords(adata: AnnData, key: str = "spatial") -> np.ndarray:
    """Extract spatial coordinates from adata.obsm.

    Parameters
    ----------
    adata : AnnData
    key : str
        Key in ``adata.obsm`` containing coordinates.

    Returns
    -------
    np.ndarray of shape (n_cells, 2) or (n_cells, 3).
    """
    if key not in adata.obsm:
        raise KeyError(
            f"'{key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    return np.asarray(adata.obsm[key])


def validate_spatial_coords(coords: np.ndarray) -> None:
    """Assert that coords are 2D or 3D spatial coordinates.

    Parameters
    ----------
    coords : np.ndarray of shape (n_cells, 2) or (n_cells, 3)

    Raises
    ------
    ValueError if dimensionality is not 2 or 3.
    """
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError(
            f"Spatial coordinates must have shape (n_cells, 2) or (n_cells, 3), "
            f"got {coords.shape}."
        )
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/spatialvi/utils/ tests/utils/
git commit -m "feat: add spatial utility helpers (get_spatial_coords, validate_spatial_coords)"
```

---

## Task 9: Full base infrastructure smoke test

This task verifies the entire base layer works together before moving to model porting.

- [ ] **Step 1: Run all base tests together**

```bash
pytest tests/base/ tests/utils/ -v
```

Expected: all PASS.

- [ ] **Step 2: Verify the package imports cleanly**

```bash
python -c "
import spatialvi
from spatialvi.model.base import SpatialBaseModel, SpatialNeighborhoodMixin, SpatialDeconvolutionMixin
from spatialvi.data import SpatialCoordsField, NeighborhoodGraphField
from spatialvi.utils import get_spatial_coords
print('All base imports OK')
"
```

Expected: `All base imports OK` with no errors.

- [ ] **Step 3: Commit (if any fixes were needed)**

```bash
git commit -am "fix: resolve any import issues in base infrastructure"
```

---

## Task 10: scVIVA Module (`module/_nichevae.py`)

**Files:**
- Create: `src/spatialvi/module/_nichevae.py`
- Modify: `src/spatialvi/module/__init__.py`

**Source:** `scvi-tools-main-always/src/scvi/external/scviva/_module.py` + `_components.py` + `_log_likelihood.py`

- [ ] **Step 1: Copy and adapt module files**

Copy the following files from scvi-tools into `src/spatialvi/module/_nichevae.py`:
- `scvi.external.scviva._module` → the `nicheVAE` class
- `scvi.external.scviva._components` → `DirichletDecoder`, `NicheDecoder` classes
- `scvi.external.scviva._log_likelihood` → log likelihood helpers

Combine all three into `src/spatialvi/module/_nichevae.py` OR keep them as companion files:
- `src/spatialvi/module/_nichevae.py` (nicheVAE class)
- `src/spatialvi/module/_nichevae_components.py` (DirichletDecoder, NicheDecoder, Encoder)
- `src/spatialvi/module/_nichevae_log_likelihood.py` (log likelihood helpers)

**Import adjustments required:**
```python
# Change:
from scvi.external.scviva._components import DirichletDecoder, NicheDecoder
from scvi.external.scviva._constants import SCVIVA_MODULE_KEYS, SCVIVA_REGISTRY_KEYS
# To:
from spatialvi.module._nichevae_components import DirichletDecoder, NicheDecoder
from spatialvi._constants import SCVIVA_MODULE_KEYS, SCVIVA_REGISTRY_KEYS
```

All other imports (`from scvi.module import VAE`, `from scvi.module.base import LossOutput`) remain unchanged — these are base scvi imports we keep.

- [ ] **Step 2: Add `SCVIVA_MODULE_KEYS` and `SCVIVA_REGISTRY_KEYS` to `_constants.py`**

Port from `scvi-tools-main-always/src/scvi/external/scviva/_constants.py`:

```python
# src/spatialvi/_constants.py  (append to existing file)
from dataclasses import dataclass


@dataclass
class SCVIVA_REGISTRY_KEYS:
    CELL_COORDINATES_KEY: str = "cell_coordinates"
    NICHE_INDEXES_KEY: str = "niche_indexes"
    NICHE_DISTANCES_KEY: str = "niche_distances"
    NICHE_COMPOSITION_KEY: str = "niche_composition"
    EXPRESSION_EMBEDDING_KEY: str = "expression_embedding"
    EXPRESSION_EMBEDDING_NICHE_KEY: str = "expression_embedding_niche"


@dataclass
class SCVIVA_MODULE_KEYS:
    NICHE_MEAN_KEY: str = "niche_mean"
    NICHE_VAR_KEY: str = "niche_var"
    NICHE_LOGITS_KEY: str = "niche_logits"
```

- [ ] **Step 3: Update `src/spatialvi/module/__init__.py`**

```python
from ._mrdeconv import MRDeconv
from ._nichevae import nicheVAE
from ._nichevae_components import DirichletDecoder, NicheDecoder
from ._resolvae import RESOLVAE

__all__ = ["nicheVAE", "NicheDecoder", "DirichletDecoder", "MRDeconv", "RESOLVAE"]
```

(MRDeconv and RESOLVAE don't exist yet — add them in Tasks 13 and 16.)

- [ ] **Step 4: Smoke test the module**

```bash
python -c "from spatialvi.module._nichevae import nicheVAE; print('nicheVAE OK')"
```

- [ ] **Step 5: Commit**

```bash
git add src/spatialvi/module/
git commit -m "feat: port nicheVAE module from scvi-tools external/scviva"
```

---

## Task 11: scVIVA Model (`model/_scviva.py`) + Differential Expression

**Files:**
- Create: `src/spatialvi/model/_scviva.py`
- Create: `src/spatialvi/model/_scviva_de/__init__.py`
- Create: `src/spatialvi/model/_scviva_de/_niche_de_core.py`
- Create: `src/spatialvi/model/_scviva_de/_de_utils.py`
- Create: `src/spatialvi/model/_scviva_de/_marker_classifier.py`
- Create: `src/spatialvi/model/_scviva_de/_results_dataclass.py`

**Source:** `scvi-tools-main-always/src/scvi/external/scviva/_model.py` + `differential_expression/`

- [ ] **Step 1: Port `_scviva.py`**

Copy `scvi-tools-main-always/src/scvi/external/scviva/_model.py` to `src/spatialvi/model/_scviva.py`.

**Key import changes:**
```python
# Remove:
from scvi.external.scviva._module import nicheVAE
from scvi.external.scviva._constants import SCVIVA_REGISTRY_KEYS
from scvi.external.scviva.differential_expression import _niche_de_core
# Add:
from spatialvi.module._nichevae import nicheVAE
from spatialvi._constants import SCVIVA_REGISTRY_KEYS
from spatialvi.model._scviva_de import _niche_de_core
```

**Class declaration change** (add `SpatialNeighborhoodMixin` and `SpatialBaseModel`):
```python
from spatialvi.model.base import SpatialNeighborhoodMixin, SpatialBaseModel
from scvi.model.base import (
    ArchesMixin, EmbeddingMixin, RNASeqMixin, VAEMixin,
    UnsupervisedTrainingMixin, BaseMinifiedModeModelClass,
)

class SCVIVA(
    SpatialNeighborhoodMixin,
    EmbeddingMixin,
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    UnsupervisedTrainingMixin,
    SpatialBaseModel,           # inherits from BaseModelClass
    BaseMinifiedModeModelClass, # also inherits from BaseModelClass — diamond, valid C3
):
```

> **MRO note:** Both `SpatialBaseModel(BaseModelClass)` and `BaseMinifiedModeModelClass(BaseModelClass)` share the same ultimate base. Python's C3 linearization handles this diamond correctly — it does **not** raise `TypeError`. The resolved MRO is: `SCVIVA → SpatialNeighborhoodMixin → mixins... → SpatialBaseModel → BaseMinifiedModeModelClass → BaseModelClass`. Verify with `SCVIVA.__mro__` after class definition.

**`setup_anndata` change** — add `SpatialCoordsField` and neighbor fields:
```python
from spatialvi.data._fields import SpatialCoordsField, NeighborhoodGraphField

# Inside setup_anndata, add to the fields list:
SpatialCoordsField(obsm_key=cell_coordinates_key),
# Only add neighbor fields if they are already computed (i.e., after compute_neighbors):
*(
    [
        NeighborhoodGraphField(obsm_key=niche_indexes_key),
        NeighborhoodGraphField(obsm_key=niche_distances_key),
    ]
    if niche_indexes_key in adata.obsm
    else []
),
```

> **Note:** The upstream scVIVA's `preprocessing_anndata` step computes neighbor indexes. The `niche_indexes_key` / `niche_distances_key` fields are only registered if already present in `adata.obsm`. Users must call `SCVIVA.preprocessing_anndata(adata, ...)` or `model.compute_neighbors(adata, ...)` before `setup_anndata`.

Do **not** add `get_latent_representation` to this class — it is inherited from `SpatialBaseModel`.

- [ ] **Step 2: Port differential expression sub-module**

Create `src/spatialvi/model/_scviva_de/` and copy the four DE files from:
`scvi-tools-main-always/src/scvi/external/scviva/differential_expression/`

No import changes needed — these files import from `scvi` directly (fine, since scvi is a dependency).

- [ ] **Step 3: Verify import**

```bash
python -c "from spatialvi.model._scviva import SCVIVA; print('SCVIVA OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/spatialvi/model/_scviva.py src/spatialvi/model/_scviva_de/
git commit -m "feat: port SCVIVA model with SpatialBaseModel and SpatialNeighborhoodMixin"
```

---

## Task 12: scVIVA Tests

**Files:**
- Create: `tests/model/test_scviva.py`

**Source reference:** `scvi-tools-main-always/tests/external/scviva/test_scviva.py`

- [ ] **Step 1: Write tests** (adapt from upstream, change imports)

```python
# tests/model/test_scviva.py
import numpy as np
import pytest
from scvi.data import synthetic_iid

from spatialvi.model._scviva import SCVIVA

N_LATENT = 10
N_EPOCHS = 2
K_NN = 5

setup_kwargs = {
    "sample_key": "batch",
    "labels_key": "labels",
    "cell_coordinates_key": "coordinates",
    "expression_embedding_key": "qz1_m",
    "expression_embedding_niche_key": "qz1_m_niche_ct",
    "niche_composition_key": "neighborhood_composition",
    "niche_indexes_key": "niche_indexes",
    "niche_distances_key": "niche_distances",
}


@pytest.fixture(scope="session")
def scviva_adata():
    adata = synthetic_iid(
        batch_size=128, n_genes=50, n_proteins=0, n_regions=0,
        n_batches=2, n_labels=3, dropout_ratio=0.3,
        generate_coordinates=True, sparse_format=None, return_mudata=False,
    )
    adata.obsm["qz1_m"] = np.random.normal(size=(adata.shape[0], 20))
    adata.layers["counts"] = adata.X.copy()
    return adata


def test_scviva_train(scviva_adata):
    SCVIVA.preprocessing_anndata(scviva_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(scviva_adata, layer="counts", batch_key="batch", **setup_kwargs)
    model = SCVIVA(scviva_adata, prior_mixture=False)
    model.train(max_epochs=N_EPOCHS, accelerator="cpu")
    assert model.is_trained


def test_scviva_get_latent_cpu(scviva_adata):
    SCVIVA.preprocessing_anndata(scviva_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(scviva_adata, layer="counts", batch_key="batch", **setup_kwargs)
    model = SCVIVA(scviva_adata)
    model.train(max_epochs=N_EPOCHS, accelerator="cpu")
    latent = model.get_latent_representation(backend="cpu")
    assert latent.shape[0] == scviva_adata.n_obs


def test_scviva_plot_spatial_embedding(scviva_adata):
    SCVIVA.preprocessing_anndata(scviva_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(scviva_adata, layer="counts", batch_key="batch", **setup_kwargs)
    model = SCVIVA(scviva_adata)
    model.train(max_epochs=N_EPOCHS, accelerator="cpu")
    # Should not raise; returns None (shows figure inline) or Axes
    model.plot_spatial_embedding(scviva_adata, basis="coordinates")


def test_scviva_compute_neighbors(scviva_adata):
    pytest.importorskip("squidpy")
    SCVIVA.preprocessing_anndata(scviva_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(scviva_adata, layer="counts", **setup_kwargs)
    model = SCVIVA(scviva_adata)
    model.compute_neighbors(scviva_adata, spatial_key="coordinates", n_neighs=5)
    assert "index_neighbor" in scviva_adata.obsm
```

- [ ] **Step 2: Run**

```bash
pytest tests/model/test_scviva.py -v --accelerator=cpu
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/model/test_scviva.py
git commit -m "test: add scVIVA model tests"
```

---

## Task 13: DestVI Module (`module/_mrdeconv.py`)

**Source:** `scvi-tools-main-always/src/scvi/module/_mrdeconv.py`

- [ ] **Step 1: Copy and adapt**

Copy `src/scvi/module/_mrdeconv.py` → `src/spatialvi/module/_mrdeconv.py`.

No import changes needed — `MRDeconv` only imports from `scvi.module.base` and `torch`, all of which are dependencies.

- [ ] **Step 2: Smoke test**

```bash
python -c "from spatialvi.module._mrdeconv import MRDeconv; print('MRDeconv OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/spatialvi/module/_mrdeconv.py
git commit -m "feat: port MRDeconv module from scvi-tools"
```

---

## Task 14: DestVI Model (`model/_destvi.py`)

**Source:** `scvi-tools-main-always/src/scvi/model/_destvi.py`

- [ ] **Step 1: Port `_destvi.py`**

Copy and adapt:

```python
# Key import change:
from spatialvi.module._mrdeconv import MRDeconv
from spatialvi.model.base import SpatialDeconvolutionMixin, SpatialBaseModel
from scvi.model.base import UnsupervisedTrainingMixin, BaseModelClass

class DestVI(
    SpatialDeconvolutionMixin,
    UnsupervisedTrainingMixin,
    SpatialBaseModel,           # replaces BaseModelClass
):
    _module_cls = MRDeconv
    # rest of class unchanged from upstream
```

The `get_proportions()` method from upstream is preserved — `SpatialDeconvolutionMixin.get_proportions_df()` calls it.
`from_rna_model()` is preserved as-is (accepts `scvi.model.CondSCVI`).

- [ ] **Step 2: Verify import**

```bash
python -c "from spatialvi.model._destvi import DestVI; print('DestVI OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/spatialvi/model/_destvi.py
git commit -m "feat: port DestVI model with SpatialBaseModel and SpatialDeconvolutionMixin"
```

---

## Task 15: DestVI Tests

**Files:**
- Create: `tests/model/test_destvi.py`

**Source reference:** `scvi-tools-main-always/tests/model/test_destvi.py` (if exists) or write from scratch.

- [ ] **Step 1: Write tests**

```python
# tests/model/test_destvi.py
import numpy as np
import pytest
from scvi.data import synthetic_iid
from scvi.model import CondSCVI

from spatialvi.model._destvi import DestVI


@pytest.fixture(scope="session")
def destvi_data():
    sc_adata = synthetic_iid(n_labels=4, n_genes=50, sparse_format=None)
    sc_adata.layers["counts"] = sc_adata.X.copy()
    sc_adata.obsm["spatial"] = np.random.rand(sc_adata.n_obs, 2)

    st_adata = synthetic_iid(n_labels=4, n_genes=50, n_batches=1, sparse_format=None)
    st_adata.layers["counts"] = st_adata.X.copy()
    st_adata.obsm["spatial"] = np.random.rand(st_adata.n_obs, 2)
    return sc_adata, st_adata


def test_destvi_from_rna_model_train(destvi_data):
    sc_adata, st_adata = destvi_data

    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="labels")
    sc_model = CondSCVI(sc_adata, weight_obs=False)
    sc_model.train(max_epochs=2, accelerator="cpu")

    DestVI.setup_anndata(st_adata, layer="counts")
    st_model = DestVI.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=2, accelerator="cpu")

    assert st_model.is_trained


def test_destvi_get_proportions_df(destvi_data):
    sc_adata, st_adata = destvi_data

    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="labels")
    sc_model = CondSCVI(sc_adata, weight_obs=False)
    sc_model.train(max_epochs=2, accelerator="cpu")

    DestVI.setup_anndata(st_adata, layer="counts")
    st_model = DestVI.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=2, accelerator="cpu")

    df = st_model.get_proportions_df(st_adata)
    assert df.shape[1] == 4  # n_labels
    np.testing.assert_allclose(df.sum(axis=1).values, 1.0, atol=1e-4)


def test_condscvi_not_re_exported():
    """CondSCVI must NOT be exported from spatialvi — users import from scvi.model directly."""
    import spatialvi
    with pytest.raises(AttributeError):
        _ = spatialvi.CondSCVI


def test_destvi_plot_cell_type_map(destvi_data):
    sc_adata, st_adata = destvi_data
    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="labels")
    sc_model = CondSCVI(sc_adata, weight_obs=False)
    sc_model.train(max_epochs=2, accelerator="cpu")
    DestVI.setup_anndata(st_adata, layer="counts")
    st_model = DestVI.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=2, accelerator="cpu")
    # Should not raise
    import matplotlib
    matplotlib.use("Agg")
    st_model.plot_cell_type_map(st_adata, cell_type=st_model.cell_type_mapping[0])
```

- [ ] **Step 2: Run**

```bash
pytest tests/model/test_destvi.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/model/test_destvi.py
git commit -m "test: add DestVI model tests including get_proportions_df and plot_cell_type_map"
```

---

## Task 16: ResolVI Module (`module/_resolvae.py`)

**Source:** `scvi-tools-main-always/src/scvi/external/resolvi/_module.py`

- [ ] **Step 1: Copy and adapt**

Copy `_module.py` → `src/spatialvi/module/_resolvae.py`.

No import changes needed — RESOLVAE imports from `scvi.module.base`, `pyro`, `torch`. All are dependencies.

- [ ] **Step 2: Smoke test**

```bash
python -c "from spatialvi.module._resolvae import RESOLVAE; print('RESOLVAE OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/spatialvi/module/_resolvae.py
git commit -m "feat: port RESOLVAE Pyro module from scvi-tools external/resolvi"
```

---

## Task 17: ResolVI Model (`model/_resolvi.py`)

**Source:** `scvi-tools-main-always/src/scvi/external/resolvi/_model.py` + `_utils.py`

- [ ] **Step 1: Copy `_utils.py`** (ResolVIPredictiveMixin)

`_utils.py` is used as-is from `scvi.external.resolvi._utils`. Do **not** copy it — import directly:

```python
from scvi.external.resolvi._utils import ResolVIPredictiveMixin
```

- [ ] **Step 2: Port `_resolvi.py`**

```python
# Key import changes:
from spatialvi.module._resolvae import RESOLVAE
from spatialvi.model.base import SpatialNeighborhoodMixin, SpatialBaseModel
from scvi.model.base import PyroSviTrainMixin, PyroSampleMixin, ArchesMixin, BaseModelClass
from scvi.external.resolvi._utils import ResolVIPredictiveMixin

class ResolVI(
    SpatialNeighborhoodMixin,
    SpatialBaseModel,           # get_latent_representation defined here (takes MRO priority)
    PyroSviTrainMixin,
    PyroSampleMixin,
    ResolVIPredictiveMixin,     # retains get_neighbor_abundance, get_normalized_expression*
    ArchesMixin,
    BaseModelClass,
):
    _module_cls = RESOLVAE
```

**`setup_anndata` change** — suppress upstream neighbor computation:

```python
@classmethod
def setup_anndata(cls, adata, ..., **kwargs):
    # IMPORTANT: pass prepare_data=False to suppress upstream _prepare_data()
    # Neighbor computation is owned by SpatialNeighborhoodMixin.compute_neighbors()
    super().setup_anndata(adata, ..., prepare_data=False, **kwargs)
```

Also add `SpatialCoordsField` to the registered fields in `setup_anndata`.

- [ ] **Step 3: Verify import**

```bash
python -c "from spatialvi.model._resolvi import ResolVI; print('ResolVI OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/spatialvi/model/_resolvi.py
git commit -m "feat: port ResolVI model with SpatialBaseModel, SpatialNeighborhoodMixin, prepare_data=False"
```

---

## Task 18: ResolVI Tests

**Files:**
- Create: `tests/model/test_resolvi.py`

**Source reference:** `scvi-tools-main-always/tests/external/resolvi/test_resolvi.py`

- [ ] **Step 1: Write tests**

```python
# tests/model/test_resolvi.py
import numpy as np
import pytest
from scvi.data import synthetic_iid

from spatialvi.model._resolvi import ResolVI


@pytest.fixture(scope="session")
def resolvi_adata():
    adata = synthetic_iid(generate_coordinates=True, n_regions=0, n_proteins=0)
    adata.obsm["X_spatial"] = adata.obsm["coordinates"]
    adata.obs["cell_area"] = np.random.gamma(2.0, 1.0, size=adata.n_obs)
    return adata


def test_resolvi_train(resolvi_adata):
    ResolVI.setup_anndata(resolvi_adata)
    model = ResolVI(resolvi_adata)
    model.train(max_epochs=2)
    assert model.is_trained


def test_resolvi_get_latent_cpu(resolvi_adata):
    ResolVI.setup_anndata(resolvi_adata)
    model = ResolVI(resolvi_adata)
    model.train(max_epochs=2)
    latent = model.get_latent_representation(backend="cpu")
    assert latent.shape[0] == resolvi_adata.n_obs


def test_resolvi_compute_neighbors(resolvi_adata):
    pytest.importorskip("squidpy")
    ResolVI.setup_anndata(resolvi_adata)
    model = ResolVI(resolvi_adata)
    model.compute_neighbors(resolvi_adata, spatial_key="X_spatial", n_neighs=5)
    assert "index_neighbor" in resolvi_adata.obsm
    assert "distance_neighbor" in resolvi_adata.obsm


def test_resolvi_size_factor(resolvi_adata):
    ResolVI.setup_anndata(resolvi_adata, batch_key="batch", size_factor_key="cell_area")
    model = ResolVI(resolvi_adata, size_scaling=True)
    model.train(max_epochs=2)
    assert model.is_trained


def test_resolvi_get_neighbor_abundance(resolvi_adata):
    """Verify ResolVIPredictiveMixin methods are retained."""
    ResolVI.setup_anndata(resolvi_adata)
    model = ResolVI(resolvi_adata)
    model.train(max_epochs=2)
    model.compute_neighbors(resolvi_adata, spatial_key="X_spatial", n_neighs=5)
    abundance = model.get_neighbor_abundance(resolvi_adata)
    assert abundance is not None
```

- [ ] **Step 2: Run**

```bash
pytest tests/model/test_resolvi.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/model/test_resolvi.py
git commit -m "test: add ResolVI model tests including neighbor computation and ResolVIPredictiveMixin"
```

---

## Task 19: Documentation (`docs/user_guide/`)

**Files:**
- Create: `docs/user_guide/models/scviva.md`
- Create: `docs/user_guide/models/destvi.md`
- Create: `docs/user_guide/models/resolvi.md`
- Create: `docs/conf.py` (Sphinx config)

- [ ] **Step 1: Port user guide docs from scvi-tools**

Source files:
- `scvi-tools-main-always/docs/user_guide/models/scviva.md`
- `scvi-tools-main-always/docs/user_guide/models/destvi.md`
- `scvi-tools-main-always/docs/user_guide/models/resolvi.md`

Find/replace `scvi.external.SCVIVA` → `spatialvi.SCVIVA`, etc.
Update code examples to use `from spatialvi.model import ...` imports.

- [ ] **Step 2: Create minimal `docs/conf.py`** (Sphinx)

```python
project = "spatialvi-tools"
author = "Ori Kronfeld"
release = "0.1.0"
extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]
html_theme = "sphinx_book_theme"
bibtex_bibfiles = ["references.bib"]
```

- [ ] **Step 3: Commit**

```bash
git add docs/user_guide/ docs/conf.py
git commit -m "docs: add user guide for scVIVA, DestVI, ResolVI"
```

---

## Task 20: Final Integration & Smoke Test

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v --cov=spatialvi --cov-report=term-missing
```

Expected: all tests PASS, coverage reported.

- [ ] **Step 2: Verify top-level lazy imports**

```bash
python -c "
from spatialvi import SCVIVA, DestVI, ResolVI
print('SCVIVA:', SCVIVA)
print('DestVI:', DestVI)
print('ResolVI:', ResolVI)
"
```

- [ ] **Step 3: Verify pre-commit passes**

```bash
pre-commit run --all-files
```

- [ ] **Step 4: Build the package**

```bash
pip install hatchling
python -m hatchling build
```

Expected: a `.whl` and `.tar.gz` appear in `dist/`.

- [ ] **Step 5: Tag v0.1.0**

```bash
git tag v0.1.0
git push origin main --tags
```

- [ ] **Step 6: Final commit if any fixes were needed**

```bash
git commit -am "fix: final integration fixes for v0.1.0"
```

---

## File Map Summary

| File | Task | Status |
|------|------|--------|
| `pyproject.toml` | 1 | |
| `.github/workflows/*.yml` | 2 | |
| `src/spatialvi/__init__.py` | 3 | |
| `src/spatialvi/_constants.py` | 3, 10 | |
| `src/spatialvi/_settings.py` | 3 | |
| `src/spatialvi/train/_config.py` | 3 | |
| `src/spatialvi/data/_fields.py` | 4 | |
| `src/spatialvi/model/base/_spatial_base.py` | 5 | |
| `src/spatialvi/model/base/_neighborhood_mixin.py` | 6 | |
| `src/spatialvi/model/base/_deconvolution_mixin.py` | 7 | |
| `src/spatialvi/utils/_spatial.py` | 8 | |
| `src/spatialvi/module/_nichevae.py` | 10 | |
| `src/spatialvi/model/_scviva.py` | 11 | |
| `src/spatialvi/model/_scviva_de/` | 11 | |
| `tests/model/test_scviva.py` | 12 | |
| `src/spatialvi/module/_mrdeconv.py` | 13 | |
| `src/spatialvi/model/_destvi.py` | 14 | |
| `tests/model/test_destvi.py` | 15 | |
| `src/spatialvi/module/_resolvae.py` | 16 | |
| `src/spatialvi/model/_resolvi.py` | 17 | |
| `tests/model/test_resolvi.py` | 18 | |
| `docs/user_guide/models/*.md` | 19 | |

**Tasks 10–18 dependency graph:**
```
Task 9 (base smoke test)
    ├── Task 10 (nicheVAE module)
    │       └── Task 11 (SCVIVA model)
    │               └── Task 12 (SCVIVA tests)
    ├── Task 13 (MRDeconv module)
    │       └── Task 14 (DestVI model)
    │               └── Task 15 (DestVI tests)
    └── Task 16 (RESOLVAE module)
            └── Task 17 (ResolVI model)
                    └── Task 18 (ResolVI tests)
```
Tasks 10, 13, 16 can run in parallel. Tasks 11, 14, 17 can run in parallel after their module task.
