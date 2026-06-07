# spatialvi-tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `spatialvi-tools`, a clean scverse-compatible package consolidating DestVI, ResolVI, and scVIVA spatial transcriptomics models with shared base infrastructure, SpatialData/squidpy/RAPIDS integration.

**Architecture:** All three models inherit from `SpatialBaseModel` (wrapping scvi's `BaseModelClass`) which provides shared spatial setup, latent representation with RAPIDS dispatch, and spatial plotting. Model-specific capabilities (neighbor graphs, deconvolution) live in composable mixins. Every model and its module is ported from scvi-tools with import paths rewritten from `scvi.*` to `spatialvi.*` where applicable.

**Tech Stack:** Python 3.12–3.13, scvi-tools (base classes), torch, pyro-ppl, anndata, scanpy, squidpy, spatialdata, hatchling, ruff, pytest.

**Upstream source:** `/Users/orikr/PycharmProjects/scvi-tools-main-always/`
**Spec:** `docs/superpowers/specs/2026-03-24-spatialvi-design.md`

---

## Phase 0: Scope Note

Tasks 1–9 are sequential (each builds on the last). Tasks 10–18 (the three models + their tests) are **independent of each other** and can run in parallel after Task 9. Task 19 is docs. Task 20 is final integration. **Task 21 (code consolidation) runs last**, only after Task 20 passes — it requires a clean regression baseline before any refactoring.

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

---

## Task 21: Code Consolidation Across Models

**Prerequisite:** Task 20 (full test suite passing, all 3 models working).

**Goal:** Identify and extract duplicated logic between the three model implementations into shared utilities, without breaking any existing behaviour. The acceptance bar is strict: **all original scvi-tools tests for scVIVA, ResolVI, and DestVI must still pass against the spatialvi implementations.**

### Step 1: Run upstream scvi-tools tests against spatialvi models

Before touching any code, establish a regression baseline using the original scvi-tools test files, adapted to import from spatialvi:

- [ ] **Copy upstream tests as regression fixtures**

```bash
mkdir -p tests/regression
cp /Users/orikr/PycharmProjects/scvi-tools-main-always/tests/external/scviva/test_scviva.py \
   tests/regression/test_scviva_upstream.py
cp /Users/orikr/PycharmProjects/scvi-tools-main-always/tests/external/resolvi/test_resolvi.py \
   tests/regression/test_resolvi_upstream.py
```

Then do a single find-replace in each regression file:

```python
# In test_scviva_upstream.py:
# from scvi.external import SCVIVA  →  from spatialvi.model import SCVIVA
# from scvi.external.scviva.differential_expression import ...  →  from spatialvi.model._scviva_de import ...

# In test_resolvi_upstream.py:
# from scvi.external import RESOLVI  →  from spatialvi.model import ResolVI
# (and rename RESOLVI → ResolVI throughout)
```

- [ ] **Run regression suite — all must PASS before any consolidation**

```bash
pytest tests/regression/ -v --tb=short
```

Expected: all upstream tests pass with spatialvi imports. If any fail, fix the port (in Tasks 11/14/17) before proceeding.

- [ ] **Commit the regression test files**

```bash
git add tests/regression/
git commit -m "test: add upstream scvi-tools regression tests for scVIVA and ResolVI"
```

---

### Step 2: Audit for consolidation opportunities

Read the three model files and their modules side-by-side and record every duplication. Known candidates from the upstream source:

| Duplicated pattern | Found in | Candidate home |
|---|---|---|
| `_validate_anndata` + dataloader setup boilerplate | All 3 model `get_*` methods | Already in `SpatialBaseModel` via scvi |
| Spatial coordinate validation in `setup_anndata` | scVIVA + ResolVI | `SpatialBaseModel._register_spatial_coords()` |
| `scrna_raw_counts_properties` call pattern | scVIVA + ResolVI | Could share a `_get_gene_properties()` helper |
| NB / ZINB dispersion string validation | scVIVA + ResolVI modules | Could share a `_validate_dispersion()` utility |
| `get_normalized_expression` with size-factor scaling | scVIVA + ResolVI | Could share via a `SpatialExpressionMixin` |
| Neighbor index → one-hot encoding for module forward | scVIVA + ResolVI modules | Could share a `_neighbors_to_onehot()` helper |
| `save` / `load` + history assertions in tests | All 3 test files | Already handled by scvi's `BaseModelClass` |

- [ ] **For each candidate: decide consolidate or leave**

Use this rubric:
- **Consolidate** if: identical logic > 10 lines, no model-specific branching, change is purely extracting to a shared location.
- **Leave** if: logic differs in subtle ways between models, or the saving is < 5 lines, or extracting would require a new abstraction layer.

Write the decisions as a comment block at the top of `src/spatialvi/model/base/_spatial_base.py`:

```python
# Consolidation log (Task 21):
# - spatial coord validation → _register_spatial_coords() [DONE via SpatialCoordsField]
# - neighbor field registration → SpatialNeighborhoodMixin._setup_neighbor_field() [DONE]
# - get_latent_representation RAPIDS dispatch → SpatialBaseModel [DONE]
# - NB dispersion validation → LEFT in each module (logic differs, saving < 5 lines)
# - neighbor one-hot encoding → [DECISION: consolidate/leave + reason]
# - get_normalized_expression with size scaling → [DECISION]
```

---

### Step 3: Implement consolidations (one at a time, with regression check after each)

For each item marked CONSOLIDATE:

- [ ] **Extract to shared location**

Move the logic to its new home (`SpatialBaseModel`, a mixin, or `utils/_spatial.py`).
Update all imports in the three model files.

- [ ] **Run full test suite + regression suite**

```bash
pytest tests/ tests/regression/ -v --tb=short
```

Expected: all PASS. If anything breaks, revert this consolidation and mark it LEAVE.

- [ ] **Commit each consolidation separately**

```bash
git commit -m "refactor: consolidate <specific pattern> into <target location>"
```

> **Rule:** One consolidation per commit. Never batch multiple refactors. This makes it trivial to `git revert` a single change if a regression appears later.

---

### Step 4: Final regression check

- [ ] **Run the full suite one last time**

```bash
pytest tests/ tests/regression/ -v --cov=spatialvi --cov-report=term-missing
```

Expected: all PASS. Coverage should be >= 80%.

- [ ] **Compare line counts vs. original scvi-tools**

```bash
wc -l src/spatialvi/model/_scviva.py \
       /Users/orikr/PycharmProjects/scvi-tools-main-always/src/scvi/external/scviva/_model.py

wc -l src/spatialvi/model/_resolvi.py \
       /Users/orikr/PycharmProjects/scvi-tools-main-always/src/scvi/external/resolvi/_model.py

wc -l src/spatialvi/model/_destvi.py \
       /Users/orikr/PycharmProjects/scvi-tools-main-always/src/scvi/model/_destvi.py
```

The spatialvi versions should be **shorter** than the originals (due to shared base removing duplication). If any spatialvi file is longer, investigate why before tagging.

- [ ] **Update CHANGELOG.md** with the list of consolidations made

- [ ] **Final commit**

```bash
git commit -am "refactor: complete Task 21 code consolidation — all regression tests pass"
git tag v0.1.0
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
| `tests/regression/test_*_upstream.py` | 21 | |

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


---

# Cleanup, Restructure & Docs Expansion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three code-level redundancies in `_scviva.py`, add a missing unit test, restructure two internal packages to live under `utils/` sub-folders, and expand the docs tree to mirror scvi-tools structure.

**Architecture:** All changes stay on the existing branch `chore/remove-dead-code-and-fix-deps`. Code changes come first (tasks 1–5), then docs (tasks 6–10), then a single final commit push.

**Tech Stack:** Python 3.12, pytest, MyST Markdown (Sphinx), existing scvi-tools docs as reference template.

---

## File map

### Modified
- `src/spatialvi/model/_scviva.py` — H (shared decoder helper), I (lazy csr_matrix), J (anndata import)
- `src/spatialvi/module/_nichevae.py` — update import path after move
- `src/spatialvi/module/__init__.py` — update if it re-exports anything from components
- `docs/index.md` — add new top-level toctree entries
- `docs/user_guide/index.md` — add background + use_cases toctree
- `CHANGELOG.md` — add unreleased section

### Moved (rename)
- `src/spatialvi/module/_nichevae_components.py` → `src/spatialvi/module/utils/_nichevae_components.py`
- `src/spatialvi/model/_scviva_de/` → `src/spatialvi/model/utils/_scviva_de/`

### Created
- `src/spatialvi/module/utils/__init__.py`
- `src/spatialvi/model/utils/__init__.py`
- `tests/model/test_scviva_utils.py` — test for `get_niche_indexes`
- `docs/api/index.md`
- `docs/api/user.md`
- `docs/api/developer.md`
- `docs/developer/index.md`
- `docs/developer/code.md`
- `docs/developer/maintenance.md`
- `docs/user_guide/background/index.md`
- `docs/user_guide/background/variational_inference.md`
- `docs/user_guide/background/differential_expression.md`
- `docs/user_guide/background/spatial_methods.md`
- `docs/user_guide/use_cases/index.md`
- `docs/user_guide/use_cases/saving_and_loading_models.md`
- `docs/user_guide/use_cases/training_configuration.md`
- `docs/user_guide/use_cases/downstream_analysis.md`
- `docs/faq.md`
- `docs/installation.md`

---

## Task 1 — Fix H: extract shared helper for predict_neighborhood / predict_niche_activation

**Files:**
- Modify: `src/spatialvi/model/_scviva.py:388–491`

- [ ] **Step 1: Add private helper `_run_spatial_decoder` just before `predict_neighborhood`**

  Insert at line 387 (before `def predict_neighborhood`):

  ```python
  def _run_spatial_decoder(self, decoder_fn, adata, indices, batch_size):
      """Run a spatial decoder over batches and return concatenated numpy output.

      Parameters
      ----------
      decoder_fn
          Callable that takes ``(decoder_input, batch_index)`` and returns a
          tensor or a tuple whose first element is the tensor to collect.
      adata
          AnnData object or ``None`` (falls back to model adata).
      indices
          Cell indices or ``None`` for all cells.
      batch_size
          Mini-batch size.

      Returns
      -------
      numpy.ndarray of shape (n_cells, n_outputs)
      """
      self._check_if_trained(warn=False)
      adata = self._validate_anndata(adata)
      scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

      results = []
      for tensors in scdl:
          inference_inputs = self.module._get_inference_input(tensors)
          outputs = self.module.inference(**inference_inputs)
          batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
          decoder_input = outputs["qz"].loc
          batch_index = batch_index.to(decoder_input.device)
          out = decoder_fn(decoder_input, batch_index)
          # decoder may return a tuple (e.g. niche_decoder returns (p_m, p_v))
          if isinstance(out, tuple):
              out = out[0]
          results.append(out.detach().cpu())

      return torch.cat(results).numpy()
  ```

- [ ] **Step 2: Replace `predict_neighborhood` body**

  Replace the full method body (keep docstring and signature):

  ```python
  @torch.inference_mode()
  def predict_neighborhood(
      self,
      adata: AnnData | None = None,
      indices: np.ndarray | None = None,
      batch_size: int | None = 1024,
  ) -> np.ndarray:
      """
      Predict the cell type composition of each cell niche in the dataset.

      Parameters
      ----------
      adata
          AnnData object. If ``None``, the model's ``adata`` will be used.
      indices
          Indices of cells to use. If ``None``, all cells will be used.
      batch_size
          Minibatch size to use during inference.

      Returns
      -------
      ct_prediction
          Predicted cell type composition of each cell niche in the dataset.
          It is computed as the expectation of the Dirichlet distribution.
      """
      def _decoder(decoder_input, batch_index):
          dist = self.module.composition_decoder(decoder_input, batch_index)
          return dist.concentration / dist.concentration.sum(dim=1).unsqueeze(1)

      return self._run_spatial_decoder(_decoder, adata, indices, batch_size)
  ```

- [ ] **Step 3: Replace `predict_niche_activation` body**

  ```python
  @torch.inference_mode()
  def predict_niche_activation(
      self,
      adata: AnnData | None = None,
      indices: np.ndarray | None = None,
      batch_size: int | None = 1024,
  ) -> np.ndarray:
      """
      Predict the activation of each cell niche in the dataset.

      Parameters
      ----------
      adata
          AnnData object. If ``None``, the model's ``adata`` will be used.
      indices
          Indices of cells to use. If ``None``, all cells will be used.
      batch_size
          Minibatch size to use during inference.

      Returns
      -------
      niche_activation
          Predicted activation of each cell niche in the dataset.
      """
      return self._run_spatial_decoder(
          self.module.niche_decoder, adata, indices, batch_size
      )
  ```

- [ ] **Step 4: Run tests**

  ```bash
  python -m pytest tests/ -x -q
  ```
  Expected: all pass

---

## Task 2 — Fix I: make `csr_matrix` a local import in `_pad_and_sort_query_anndata`

**Files:**
- Modify: `src/spatialvi/model/_scviva.py:56` and `:1171`

- [ ] **Step 1: Remove the top-level import at line 56**

  Delete the line:
  ```python
  from scipy.sparse import csr_matrix
  ```

- [ ] **Step 2: Add local import at the usage site inside `_pad_and_sort_query_anndata`**

  Find the line `padding_mtx = csr_matrix(...)` (currently ~line 1171) and add the import
  immediately before it:
  ```python
  from scipy.sparse import csr_matrix
  padding_mtx = csr_matrix(np.zeros((adata.n_obs, len(genes_to_add))))
  ```

- [ ] **Step 3: Run tests**

  ```bash
  python -m pytest tests/ -x -q
  ```
  Expected: all pass

---

## Task 3 — Fix J: consolidate `anndata` imports

**Files:**
- Modify: `src/spatialvi/model/_scviva.py:8,13,1179`

- [ ] **Step 1: Remove `import anndata` (line 8) and merge into the existing `from anndata` line**

  Replace:
  ```python
  import anndata
  ...
  from anndata import AnnData
  ```
  With:
  ```python
  from anndata import AnnData, concat as anndata_concat
  ```

- [ ] **Step 2: Update the single call site (~line 1179)**

  Replace:
  ```python
  adata_out = anndata.concat(
  ```
  With:
  ```python
  adata_out = anndata_concat(
  ```

- [ ] **Step 3: Run tests**

  ```bash
  python -m pytest tests/ -x -q
  ```
  Expected: all pass

---

## Task 4 — Add test for `get_niche_indexes`

**Files:**
- Create: `tests/model/test_scviva_utils.py`

- [ ] **Step 1: Write the test file**

  ```python
  """Tests for module-level utility functions in _scviva.py."""
  from __future__ import annotations

  import numpy as np
  import pandas as pd
  import pytest
  from anndata import AnnData

  from spatialvi.model._scviva import get_niche_indexes


  def _make_adata(n_cells=20, n_genes=10, seed=0):
      rng = np.random.default_rng(seed)
      # Two samples of 10 cells each, laid out on a 2D grid
      coords = rng.uniform(0, 10, size=(n_cells, 2))
      obs = pd.DataFrame(
          {"sample": ["s1"] * 10 + ["s2"] * 10},
          index=[f"cell_{i}" for i in range(n_cells)],
      )
      adata = AnnData(
          X=rng.poisson(5, size=(n_cells, n_genes)).astype(float),
          obs=obs,
      )
      adata.obsm["spatial"] = coords
      return adata


  def test_get_niche_indexes_shapes():
      """Index and distance arrays must match (n_cells, k_nn)."""
      adata = _make_adata()
      k_nn = 5
      get_niche_indexes(
          adata,
          sample_key="sample",
          niche_indexes_key="niche_idx",
          cell_coordinates_key="spatial",
          k_nn=k_nn,
          niche_distances_key="niche_dist",
      )
      assert adata.obsm["niche_idx"].shape == (20, k_nn)
      assert adata.obsm["niche_dist"].shape == (20, k_nn)


  def test_get_niche_indexes_no_self_loops():
      """No cell should appear as its own neighbor."""
      adata = _make_adata()
      get_niche_indexes(
          adata,
          sample_key="sample",
          niche_indexes_key="niche_idx",
          cell_coordinates_key="spatial",
          k_nn=4,
          niche_distances_key="niche_dist",
      )
      idx = adata.obsm["niche_idx"]
      global_indices = np.arange(len(adata))
      for i, row in enumerate(idx):
          assert i not in row, f"cell {i} listed as its own neighbor"


  def test_get_niche_indexes_neighbors_within_sample():
      """All returned neighbor indices must belong to the same sample."""
      adata = _make_adata()
      get_niche_indexes(
          adata,
          sample_key="sample",
          niche_indexes_key="niche_idx",
          cell_coordinates_key="spatial",
          k_nn=4,
          niche_distances_key="niche_dist",
      )
      idx = adata.obsm["niche_idx"].astype(int)
      samples = adata.obs["sample"].values
      for i in range(len(adata)):
          for neighbor in idx[i]:
              assert samples[neighbor] == samples[i], (
                  f"cell {i} (sample {samples[i]}) has cross-sample neighbor "
                  f"{neighbor} (sample {samples[neighbor]})"
              )


  def test_get_niche_indexes_int_dtype():
      """Index array must be integer-typed."""
      adata = _make_adata()
      get_niche_indexes(
          adata,
          sample_key="sample",
          niche_indexes_key="niche_idx",
          cell_coordinates_key="spatial",
          k_nn=3,
          niche_distances_key="niche_dist",
      )
      assert np.issubdtype(adata.obsm["niche_idx"].dtype, np.integer)


  def test_get_niche_indexes_distances_nonnegative():
      """All distances must be ≥ 0."""
      adata = _make_adata()
      get_niche_indexes(
          adata,
          sample_key="sample",
          niche_indexes_key="niche_idx",
          cell_coordinates_key="spatial",
          k_nn=3,
          niche_distances_key="niche_dist",
      )
      assert (adata.obsm["niche_dist"] >= 0).all()
  ```

- [ ] **Step 2: Run the new tests**

  ```bash
  python -m pytest tests/model/test_scviva_utils.py -v
  ```
  Expected: 5 tests PASS

- [ ] **Step 3: Run full suite**

  ```bash
  python -m pytest tests/ -x -q
  ```
  Expected: all pass

---

## Task 5 — Move `_nichevae_components.py` → `module/utils/`

**Files:**
- Create: `src/spatialvi/module/utils/__init__.py`
- Move: `src/spatialvi/module/_nichevae_components.py` → `src/spatialvi/module/utils/_nichevae_components.py`
- Modify: `src/spatialvi/module/_nichevae.py:18`

- [ ] **Step 1: Create the utils sub-package**

  ```bash
  mkdir src/spatialvi/module/utils
  ```

  Create `src/spatialvi/module/utils/__init__.py`:
  ```python
  from __future__ import annotations

  from ._nichevae_components import DirichletDecoder, Encoder, NicheDecoder

  __all__ = ["DirichletDecoder", "Encoder", "NicheDecoder"]
  ```

- [ ] **Step 2: Move the file**

  ```bash
  git mv src/spatialvi/module/_nichevae_components.py src/spatialvi/module/utils/_nichevae_components.py
  ```

- [ ] **Step 3: Update the import in `_nichevae.py`**

  Replace:
  ```python
  from spatialvi.module._nichevae_components import DirichletDecoder, Encoder, NicheDecoder
  ```
  With:
  ```python
  from spatialvi.module.utils._nichevae_components import DirichletDecoder, Encoder, NicheDecoder
  ```

- [ ] **Step 4: Run tests**

  ```bash
  python -m pytest tests/ -x -q
  ```
  Expected: all pass

---

## Task 6 — Move `model/_scviva_de/` → `model/utils/_scviva_de/`

**Files:**
- Create: `src/spatialvi/model/utils/__init__.py`
- Move: `src/spatialvi/model/_scviva_de/` → `src/spatialvi/model/utils/_scviva_de/`
- Modify: `src/spatialvi/model/_scviva.py:41,54`

- [ ] **Step 1: Create the utils sub-package**

  ```bash
  mkdir -p src/spatialvi/model/utils
  ```

  Create `src/spatialvi/model/utils/__init__.py`:
  ```python
  from __future__ import annotations
  ```

- [ ] **Step 2: Move the folder**

  ```bash
  git mv src/spatialvi/model/_scviva_de src/spatialvi/model/utils/_scviva_de
  ```

- [ ] **Step 3: Update imports in `_scviva.py`**

  Replace:
  ```python
  from spatialvi.model._scviva_de import _niche_de_core
  ```
  With:
  ```python
  from spatialvi.model.utils._scviva_de import _niche_de_core
  ```

  And in the TYPE_CHECKING block replace:
  ```python
  from spatialvi.model._scviva_de import DifferentialExpressionResults
  ```
  With:
  ```python
  from spatialvi.model.utils._scviva_de import DifferentialExpressionResults
  ```

- [ ] **Step 4: Run tests**

  ```bash
  python -m pytest tests/ -x -q
  ```
  Expected: all pass

---

## Task 7 — Add `docs/faq.md` and `docs/installation.md`

**Files:**
- Create: `docs/faq.md`
- Create: `docs/installation.md`
- Modify: `docs/index.md`

- [ ] **Step 1: Create `docs/faq.md`**

  ```markdown
  # Frequently Asked Questions

  ## Installation

  ### Which Python versions are supported?

  spatialvi-tools requires Python 3.12 or later.

  ### Can I install spatialvi-tools without GPU support?

  Yes. The base package runs entirely on CPU. GPU acceleration is available via the optional
  `rapids` extra: `pip install "spatialvi-tools[rapids]"`.

  ## Data preparation

  ### What format does my data need to be in?

  All models expect raw count data stored in an `AnnData` object. Normalized or log-transformed
  data will cause numerical issues. Pass the correct `layer` argument to `setup_anndata` if your
  counts are not in `adata.X`.

  ### What is the difference between `batch_key` and `categorical_covariate_keys`?

  `batch_key` registers the primary technical covariate (e.g. sequencing run, donor slice) and
  unlocks model features such as per-batch dispersion and counterfactual decoding.
  `categorical_covariate_keys` accepts multiple secondary covariates but supports fewer downstream
  features. Prefer `batch_key` when you have a single dominant technical effect.

  ## Training

  ### My model produces NaN losses — how do I fix this?

  Common causes:

  - **Unnormalized input**: ensure `adata.X` (or the specified `layer`) contains raw integer counts.
  - **Low-count cells/spots**: filter cells with very few total counts before training.
  - **Learning rate too high**: try reducing `lr` by an order of magnitude.
  - **Batch size too small**: small batches increase gradient variance; try `batch_size=256` or
    larger.

  ### How long should I train?

  Rule of thumb: monitor ELBO convergence in the training progress bar. Most models converge
  within 200–500 epochs on typical datasets. ResolVI requires more epochs (default 50 is a
  starting point — increase for larger datasets).

  ## Spatial analysis

  ### Do I need squidpy installed?

  squidpy is required for the default `backend="squidpy"` in `compute_neighbors`. Install it with
  `pip install "spatialvi-tools[spatial]"`. A RAPIDS GPU backend is also available.

  ### Can I use SpatialData objects directly?

  Yes. All models expose `setup_spatialdata()` and `from_spatialdata()` class methods that accept
  a `SpatialData` object and an optional `region` filter.

  ## Saving and loading

  ### How do I save and reload a trained model?

  ```python
  model.save("my_model_dir/")
  model = MyModel.load("my_model_dir/", adata=adata)
  ```

  See {doc}`user_guide/use_cases/saving_and_loading_models` for full details.
  ```

- [ ] **Step 2: Create `docs/installation.md`**

  ```markdown
  # Installation

  ## Quick install

  spatialvi-tools requires Python ≥ 3.12. Install into a fresh virtual environment:

  ```bash
  pip install spatialvi-tools
  ```

  ## Optional extras

  | Extra | Command | Installs |
  |-------|---------|---------|
  | Spatial backends | `pip install "spatialvi-tools[spatial]"` | squidpy, spatialdata |
  | GPU acceleration | `pip install "spatialvi-tools[rapids]"` | cuML, cuPy, cuGraph |
  | Tutorials | `pip install "spatialvi-tools[tutorials]"` | jupyter, matplotlib, seaborn |
  | All | `pip install "spatialvi-tools[all]"` | everything above |

  ## Development install

  ```bash
  git clone https://github.com/YosefLab/spatialvi-tools.git
  cd spatialvi-tools
  pip install -e ".[dev,test]"
  pre-commit install
  ```

  ## Verifying the installation

  ```python
  import spatialvi
  print(spatialvi.__version__)
  ```

  ## GPU support

  RAPIDS-based acceleration (neighbor computation and latent representation) requires a CUDA-capable
  GPU and the `rapids` extra. The base package runs on CPU without any GPU dependencies.
  ```

- [ ] **Step 3: Update `docs/index.md` toctree**

  Replace the current toctree block with:

  ```markdown
  ```{toctree}
  :maxdepth: 1
  :hidden:

  installation
  user_guide/index
  api/index
  developer/index
  tutorials/index
  faq
  references
  ```
  ```

---

## Task 8 — Add `docs/api/` folder

**Files:**
- Create: `docs/api/index.md`
- Create: `docs/api/user.md`
- Create: `docs/api/developer.md`

- [ ] **Step 1: Create `docs/api/index.md`**

  ```markdown
  # API Reference

  Import spatialvi-tools as:

  ```python
  import spatialvi
  ```

  ```{toctree}
  :maxdepth: 2

  user
  developer
  ```
  ```

- [ ] **Step 2: Create `docs/api/user.md`**

  ```markdown
  # User API

  Import spatialvi-tools as:

  ```python
  import spatialvi
  ```

  ## Models

  ```{eval-rst}
  .. currentmodule:: spatialvi

  .. autosummary::
     :toctree: reference/
     :nosignatures:

     model.SCVIVA
     model.DestVI
     model.ResolVI
     model.GIMVI
  ```

  ## External models

  ```{eval-rst}
  .. currentmodule:: spatialvi

  .. autosummary::
     :toctree: reference/
     :nosignatures:

     external.RNAStereoscope
     external.SpatialStereoscope
  ```

  ## Settings

  ```{eval-rst}
  .. currentmodule:: spatialvi

  .. autosummary::
     :toctree: reference/
     :nosignatures:

     settings
  ```
  ```

- [ ] **Step 3: Create `docs/api/developer.md`**

  ```markdown
  # Developer API

  Internal base classes and mixins used to build new spatialvi models.

  ## Base classes

  ```{eval-rst}
  .. currentmodule:: spatialvi.model.base

  .. autosummary::
     :toctree: reference/
     :nosignatures:

     SpatialBaseModel
     SpatialNeighborhoodMixin
     SpatialDeconvolutionMixin
     ResolVIPredictiveMixin
  ```

  ## Data fields

  ```{eval-rst}
  .. currentmodule:: spatialvi.data

  .. autosummary::
     :toctree: reference/
     :nosignatures:

     SpatialCoordsField
     NeighborhoodGraphField
  ```

  ## Modules (neural networks)

  ```{eval-rst}
  .. currentmodule:: spatialvi.module

  .. autosummary::
     :toctree: reference/
     :nosignatures:

     nicheVAE
     RESOLVAE
     MRDeconv
     JVAE
  ```
  ```

---

## Task 9 — Add `docs/developer/` folder

**Files:**
- Create: `docs/developer/index.md`
- Create: `docs/developer/code.md`
- Create: `docs/developer/maintenance.md`

- [ ] **Step 1: Create `docs/developer/index.md`**

  ```markdown
  # Developer Documentation

  Contributions are welcome and greatly appreciated. You can contribute by:

  - Reporting bugs or requesting features via [GitHub Issues](https://github.com/YosefLab/spatialvi-tools/issues)
  - Improving or expanding the [documentation]
  - Contributing code via pull requests

  ```{toctree}
  :maxdepth: 2
  :hidden:

  code
  maintenance
  ```
  ```

- [ ] **Step 2: Create `docs/developer/code.md`**

  ```markdown
  # Contributing Code

  ## Setting up a development environment

  1. Fork the [repository](https://github.com/YosefLab/spatialvi-tools) on GitHub.

  2. Clone your fork:

     ```bash
     git clone https://github.com/{your-username}/spatialvi-tools.git
     cd spatialvi-tools
     ```

  3. Install the development dependencies in editable mode:

     ```bash
     pip install -e ".[dev,test]"
     ```

  4. Install pre-commit hooks:

     ```bash
     pre-commit install
     ```

  ## Running tests

  ```bash
  python -m pytest tests/ -v
  ```

  Run a subset:

  ```bash
  python -m pytest tests/model/test_scviva.py -v
  ```

  ## Code style

  We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run it with:

  ```bash
  ruff check src/
  ruff format src/
  ```

  Pre-commit hooks run these automatically on every commit.

  ## Adding a new model

  1. Add the neural-network module under `src/spatialvi/module/`.
  2. Add the model class under `src/spatialvi/model/`, inheriting from `SpatialBaseModel`.
  3. Register it in `src/spatialvi/model/__init__.py` and `src/spatialvi/__init__.py`.
  4. Write tests under `tests/model/`.

  ## Package layout

  ```
  src/spatialvi/
  ├── model/           # High-level model classes
  │   ├── base/        # Shared mixins and base class
  │   └── utils/       # Model-level utilities and DE sub-packages
  ├── module/          # PyTorch neural-network modules
  │   └── utils/       # Module-level component libraries
  ├── data/            # Custom AnnData fields
  ├── external/        # External/ported model adaptations
  ├── train/           # Training plan overrides
  └── utils/           # Shared spatial utilities
  ```
  ```

- [ ] **Step 3: Create `docs/developer/maintenance.md`**

  ```markdown
  # Maintenance Guide

  ## Releases

  We follow [Semantic Versioning](https://semver.org). Each release is tagged `MAJOR.MINOR.PATCH`.

  ### Creating a release

  1. Update `CHANGELOG.md` — move items from `[Unreleased]` to a new versioned section.
  2. Bump the version in `pyproject.toml` and `src/spatialvi/__init__.py`.
  3. Open a PR, merge, then create a GitHub release tag.

  ## Updating dependencies

  Dependency bounds are declared in `pyproject.toml`. When a new major version of a dependency
  is released:

  1. Test against the new version locally.
  2. Update the lower bound if new features are required, or upper-bound if breaking changes exist.
  3. Update `CHANGELOG.md`.

  ## Pre-commit hooks

  The project uses [pre-commit](https://pre-commit.com) hooks for code quality. To update hook
  versions:

  ```bash
  pre-commit autoupdate
  git add .pre-commit-config.yaml
  git commit -m "chore: update pre-commit hooks"
  ```
  ```

---

## Task 10 — Add `docs/user_guide/background/` and `docs/user_guide/use_cases/`

**Files:**
- Create: `docs/user_guide/background/index.md`
- Create: `docs/user_guide/background/variational_inference.md`
- Create: `docs/user_guide/background/differential_expression.md`
- Create: `docs/user_guide/background/spatial_methods.md`
- Create: `docs/user_guide/use_cases/index.md`
- Create: `docs/user_guide/use_cases/saving_and_loading_models.md`
- Create: `docs/user_guide/use_cases/training_configuration.md`
- Create: `docs/user_guide/use_cases/downstream_analysis.md`
- Modify: `docs/user_guide/index.md`

- [ ] **Step 1: Create background index**

  `docs/user_guide/background/index.md`:
  ```markdown
  # Background

  Conceptual guides explaining the statistical and algorithmic foundations of spatialvi-tools models.

  ```{toctree}
  :maxdepth: 2

  variational_inference
  differential_expression
  spatial_methods
  ```
  ```

- [ ] **Step 2: Create `variational_inference.md`**

  ```markdown
  # Variational Inference

  ## The generative model

  All spatialvi-tools models are based on **amortized variational inference** (AVI). The core idea
  is to learn a probabilistic generative model $p_\theta(x \mid z)$ of observed gene expression $x$
  conditioned on a low-dimensional latent variable $z$, alongside an approximate posterior
  $q_\phi(z \mid x)$ parameterized by an encoder neural network.

  Training maximizes the **Evidence Lower Bound (ELBO)**:

  $$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x \mid z)] - \mathrm{KL}(q_\phi(z \mid x) \| p(z))$$

  ## Gene likelihood

  Most models support two gene likelihood distributions:

  - **Negative Binomial (NB)**: default; models overdispersed count data.
  - **Poisson**: simpler; suitable for very low-count spatial data.

  ## Spatial priors (scVIVA, ResolVI)

  Spatial models extend the standard VAE with a **niche-aware prior** that conditions the latent
  distribution on the cellular neighbourhood, encoding microenvironment structure directly into the
  latent space.

  ## References

  - Lopez et al. (2018) *Deep generative modeling for single-cell transcriptomics*. Nature Methods.
  - Levy et al. (2025) *scVIVA*. bioRxiv.
  ```

- [ ] **Step 3: Create `differential_expression.md`**

  ```markdown
  # Differential Expression

  ## Overview

  spatialvi-tools inherits scvi-tools' Bayes-factor differential expression framework and extends
  it with **niche-aware DE** (scVIVA) and **niche abundance DE** (ResolVI).

  ## Standard DE (vanilla / change mode)

  All models expose `differential_expression()`. Two modes are supported:

  - **vanilla**: computes log-fold-change between groups using posterior samples.
  - **change**: computes the posterior probability that the log-fold-change exceeds a threshold
    $\delta$ (default 0.25).

  ```python
  de_df = model.differential_expression(
      adata,
      groupby="cell_type",
      group1=["T cells"],
      group2=["B cells"],
      mode="change",
      delta=0.25,
  )
  ```

  ## Niche DE (scVIVA)

  scVIVA's `differential_niche_expression()` tests for gene expression differences that are
  explained by the **cellular microenvironment** (niche composition) rather than intrinsic
  cell-type identity.

  ## Niche abundance DE (ResolVI)

  ResolVI's `differential_niche_abundance()` tests for differences in the **composition of the
  spatial neighbourhood** between conditions.

  ## References

  - Boyeau et al. (2019) *Deep generative models for detecting differential expression*. bioRxiv.
  ```

- [ ] **Step 4: Create `spatial_methods.md`**

  ```markdown
  # Spatial Transcriptomics Methods

  ## Technology overview

  Spatial transcriptomics (ST) measures gene expression while preserving the spatial position of
  cells or spots within the tissue. Key platforms include:

  | Platform | Resolution | Typical use case |
  |----------|-----------|-----------------|
  | Visium (10x) | Multi-cell spots (~55 µm) | Whole-tissue profiling |
  | Xenium / MERSCOPE | Single-cell resolved | High-plex FISH |
  | Slide-seq | Near single-cell | Broad coverage |

  ## Key challenges

  - **Spot deconvolution** (Visium): multiple cell types per spot → DestVI.
  - **Segmentation noise** (resolved ST): transcript assignment errors → ResolVI.
  - **Niche modelling**: capturing cellular microenvironment effects → scVIVA.

  ## Neighbour graphs

  Spatial neighbour graphs encode tissue topology. spatialvi-tools computes these via
  `model.compute_neighbors()` using squidpy (CPU) or RAPIDS (GPU). The resulting
  `index_neighbor` and `distance_neighbor` arrays in `adata.obsm` are consumed by ResolVI
  and scVIVA during training.

  ## SpatialData integration

  All models support [SpatialData](https://spatialdata.scverse.org) objects via
  `setup_spatialdata()` and `from_spatialdata()`.
  ```

- [ ] **Step 5: Create use_cases index**

  `docs/user_guide/use_cases/index.md`:
  ```markdown
  # Use Cases

  Practical guides for common analysis workflows with spatialvi-tools.

  ```{toctree}
  :maxdepth: 2

  saving_and_loading_models
  training_configuration
  downstream_analysis
  ```
  ```

- [ ] **Step 6: Create `saving_and_loading_models.md`**

  ```markdown
  # Saving and Loading Models

  ## Saving

  After training, save the full model state to a directory:

  ```python
  model.save("my_model/", overwrite=True)
  ```

  This saves model weights, the AnnData manager registry, and `init_params_` so the model
  can be reconstructed exactly.

  ## Loading

  ```python
  from spatialvi import SCVIVA

  model = SCVIVA.load("my_model/", adata=adata)
  ```

  The `adata` argument must have the same variables (genes) as the training data.

  ## Transferring to a new dataset (scArches / ARCHES)

  SCVIVA and ResolVI inherit scvi-tools' `ArchesMixin`, enabling query-to-reference mapping:

  ```python
  query_model = SCVIVA.load_query_data(
      adata_query,
      reference_model="my_model/",
  )
  query_model.train(max_epochs=100)
  ```
  ```

- [ ] **Step 7: Create `training_configuration.md`**

  ```markdown
  # Training Configuration

  ## Basic training

  ```python
  model.train(max_epochs=300)
  ```

  ## Adjusting learning rate and batch size

  ```python
  model.train(max_epochs=300, lr=1e-3, batch_size=256)
  ```

  ## KL annealing

  KL divergence weight is annealed from 0 to 1 over the first `n_epochs_kl_warmup` epochs
  (default varies by model). Increase this for more stable early training:

  ```python
  model.train(max_epochs=400, n_epochs_kl_warmup=100)
  ```

  ## GPU training

  Training automatically uses a GPU when one is available via PyTorch Lightning's
  `accelerator="auto"` default. To force CPU:

  ```python
  model.train(max_epochs=300, accelerator="cpu")
  ```

  ## Monitoring training

  The training progress bar reports ELBO loss. For programmatic monitoring, pass a Lightning
  callback:

  ```python
  from lightning.pytorch.callbacks import EarlyStopping

  model.train(
      max_epochs=500,
      callbacks=[EarlyStopping(monitor="elbo_train", patience=20)],
  )
  ```
  ```

- [ ] **Step 8: Create `downstream_analysis.md`**

  ```markdown
  # Downstream Analysis

  ## Latent space

  Extract the latent representation and store it in `adata.obsm`:

  ```python
  adata.obsm["X_spatialvi"] = model.get_latent_representation()
  ```

  Use RAPIDS for GPU-accelerated UMAP:

  ```python
  z = model.get_latent_representation(backend="rapids")  # returns cupy array
  ```

  ## Clustering

  Apply standard scanpy clustering to the latent space:

  ```python
  import scanpy as sc

  sc.pp.neighbors(adata, use_rep="X_spatialvi")
  sc.tl.leiden(adata)
  sc.pl.embedding(adata, basis="spatial", color="leiden")
  ```

  ## Deconvolution (DestVI)

  ```python
  proportions = model.get_proportions()   # DataFrame (n_spots × n_celltypes)
  model.plot_cell_type_map(cell_type="Neuron")
  ```

  ## Niche prediction (scVIVA)

  ```python
  ct_composition = model.predict_neighborhood()    # (n_cells, n_celltypes)
  niche_activation = model.predict_niche_activation()  # (n_cells, n_niches)
  ```

  ## Differential expression

  ```python
  de = model.differential_expression(groupby="cell_type", mode="change")
  de[de["is_de_fdr_0.05"]].head(20)
  ```
  ```

- [ ] **Step 9: Update `docs/user_guide/index.md`**

  Replace with:
  ```markdown
  # User Guide

  ```{toctree}
  :maxdepth: 2

  models/index
  background/index
  use_cases/index
  ```
  ```

---

## Task 11 — Update `CHANGELOG.md`

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add unreleased section**

  Prepend after the title and intro (before `## [0.1.0]`):

  ```markdown
  ## [Unreleased]

  ### Changed

  - **Dead code removal**: dropped MLflow settings (`mlflow_set_tracking_uri`,
    `mlflow_set_experiment`) and all JAX/XLA environment-variable logic from
    `SpatialviConfig`; removed the now-unused `import os` from `_settings.py`
  - **Removed** stale in-source development comments (Task 20/21 consolidation notes)
  - **Removed** `train/_config.py` dead re-export of scvi `TrainerConfig`/`TrainingPlanConfig`
  - **Removed** `SpatialBaseModel.plot_spatial_predictions()` trivial alias
  - **Simplified** lazy model map: `spatialvi.__init__` now delegates to `spatialvi.model`
    instead of duplicating the same four-entry mapping

  ### Fixed

  - `scikit-learn>=1.4` added to core dependencies (was incorrectly listed only in test extras
    despite being required by SCVIVA DE machinery and ResolVI `_prepare_data`)
  - Removed `anndata` and `scanpy` from test extras (both are already core dependencies)

  ### Refactored

  - `SCVIVA.predict_neighborhood` and `predict_niche_activation` now share a private
    `_run_spatial_decoder` helper, eliminating ~30 lines of duplicated boilerplate
  - `from scipy.sparse import csr_matrix` moved from module-level to a local import inside
    the one function that uses it (`_pad_and_sort_query_anndata`)
  - `import anndata` replaced with `from anndata import AnnData, concat as anndata_concat`
    to eliminate the redundant module-level alias
  - `src/spatialvi/module/_nichevae_components.py` moved to
    `src/spatialvi/module/utils/_nichevae_components.py`
  - `src/spatialvi/model/_scviva_de/` moved to `src/spatialvi/model/utils/_scviva_de/`

  ### Documentation

  - Added `docs/installation.md` and `docs/faq.md`
  - Added `docs/api/` with user and developer API reference stubs
  - Added `docs/developer/` with contributing-code and maintenance guides
  - Added `docs/user_guide/background/` with variational inference, DE, and spatial methods guides
  - Added `docs/user_guide/use_cases/` with saving/loading, training config, and downstream
    analysis guides
  ```

---

## Task 12 — Final: run all tests, pre-commit, commit, push

- [ ] **Step 1: Run full test suite**

  ```bash
  python -m pytest tests/ -q
  ```
  Expected: all pass (83 tests: original 78 + 5 new)

- [ ] **Step 2: Run pre-commit**

  ```bash
  pre-commit run --all-files
  ```
  Fix any ruff or formatting issues, then re-stage.

- [ ] **Step 3: Stage and commit**

  ```bash
  git add -A
  git commit -m "$(cat <<'EOF'
  refactor+docs: code dedup, utils restructure, and docs expansion

  Code changes:
  - Extract _run_spatial_decoder helper to deduplicate predict_neighborhood
    and predict_niche_activation (H)
  - Move csr_matrix import to local scope in _pad_and_sort_query_anndata (I)
  - Replace `import anndata` with `from anndata import AnnData, concat` (J)
  - Add 5 unit tests for get_niche_indexes

  Structural changes:
  - Move module/_nichevae_components.py -> module/utils/_nichevae_components.py
  - Move model/_scviva_de/ -> model/utils/_scviva_de/; update all imports

  Docs:
  - Add docs/installation.md, docs/faq.md
  - Add docs/api/ (index, user, developer)
  - Add docs/developer/ (index, code, maintenance)
  - Add docs/user_guide/background/ (variational_inference, differential_expression,
    spatial_methods)
  - Add docs/user_guide/use_cases/ (saving_loading, training_config, downstream_analysis)
  - Update docs/index.md and docs/user_guide/index.md toctrees
  - Update CHANGELOG.md with unreleased section

  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
  EOF
  )"
  ```

- [ ] **Step 4: Push**

  ```bash
  git push
  ```


---

# External AMICI and Starfysh Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the smallest useful `scviva.external.amici` and `scviva.external.starfysh` model surfaces with tests and an architecture diagram update.

**Architecture:** Both integrations live under `scviva.external` and reuse `SpatialBaseModel`, scvi field registration, and existing external-package export patterns. AMICI keeps its attention-neighbor model/module logic in a compact Phase 1 form; Starfysh keeps the expression AVAE deconvolution logic in a compact Phase 1 form. Larger upstream capabilities remain later phases.

**Tech Stack:** Python, torch, scvi-tools, AnnDataManager, pytest, existing scviva base classes and mixins.

**Spec:** `docs/superpowers/specs/2026-06-01-external-amici-starfysh-design.md`

**User constraints:** Do not commit. Keep simple. Tests go in `tests/external`. Update `docs/architecture/scviva-tools-block-diagram.html` after the phase. Reuse scviva code and keep the model/module logic from AMICI and Starfysh.

**Phase 1.5 hardening note:** After the initial Phase 1 surface, add regression coverage for AMICI residual retrieval, AMICI same-label neighbor setup when requested, Starfysh singleton-batch avoidance, Starfysh two-observation training, and Starfysh's clear one-observation training error. No new architecture object is introduced in this hardening step.

---

## File Structure

- Create `src/scviva/external/amici/__init__.py`: exports `AMICI`, `AMICIModule`, and `AMICI_REGISTRY_KEYS`.
- Create `src/scviva/external/amici/_constants.py`: AMICI neighbor registry keys.
- Create `src/scviva/external/amici/_module.py`: compact attention-based AMICI module preserving the upstream input/output contract.
- Create `src/scviva/external/amici/_model.py`: scviva wrapper using `SpatialBaseModel`, scvi registration, simple neighbor computation, training, and prediction.
- Create `src/scviva/external/starfysh/__init__.py`: exports `Starfysh` and `StarfyshModule`.
- Create `src/scviva/external/starfysh/_module.py`: expression-only Starfysh AVAE-like module preserving the deconvolution logic.
- Create `src/scviva/external/starfysh/_model.py`: scviva wrapper using `SpatialBaseModel` and `SpatialDeconvolutionMixin`.
- Modify `src/scviva/external/__init__.py`: export `AMICI` and `Starfysh`.
- Create `tests/external/test_amici.py`: Phase 1 AMICI tests.
- Create `tests/external/test_starfysh.py`: Phase 1 Starfysh tests.
- Modify `docs/architecture/scviva-tools-block-diagram.html`: add AMICI and Starfysh as external spatial-track models.

---

## Task 1: AMICI Phase 1 Tests

**Files:**
- Create: `tests/external/test_amici.py`

- [ ] **Step 1: Write failing AMICI tests**

Create tests that define a small labeled spatial AnnData, call `AMICI.setup_anndata`, construct the model, run a tiny training pass, and check prediction shape.

- [ ] **Step 2: Run AMICI tests and verify failure**

Run: `pytest tests/external/test_amici.py -v`

Expected: fail because `scviva.external.amici` does not exist.

---

## Task 2: AMICI Minimal Implementation

**Files:**
- Create: `src/scviva/external/amici/__init__.py`
- Create: `src/scviva/external/amici/_constants.py`
- Create: `src/scviva/external/amici/_module.py`
- Create: `src/scviva/external/amici/_model.py`
- Modify: `src/scviva/external/__init__.py`
- Test: `tests/external/test_amici.py`

- [ ] **Step 1: Add AMICI constants**

Define registry keys for coordinates, neighbor indices, neighbor distances, neighbor expression, and neighbor labels.

- [ ] **Step 2: Add `AMICIModule`**

Implement a compact attention-neighbor module that takes target labels, neighbor expression, and neighbor distances; returns `prediction`, `residual`, `attention_patterns`, and `nn_embed`; and computes a Gaussian reconstruction loss.

- [ ] **Step 3: Add `AMICI` wrapper**

Use `SpatialBaseModel` and `UnsupervisedTrainingMixin`. Implement `setup_anndata` with `LayerField`, `CategoricalObsField`, `ObsmField`, and neighbor computation. Implement `train` by delegating to scvi's training mixin with the custom module. Implement `get_predictions`.

- [ ] **Step 4: Export AMICI**

Expose `AMICI` and `AMICIModule` from `scviva.external.amici` and `scviva.external`.

- [ ] **Step 5: Run AMICI tests**

Run: `pytest tests/external/test_amici.py -v`

Expected: pass.

---

## Task 3: Starfysh Phase 1 Tests

**Files:**
- Create: `tests/external/test_starfysh.py`

- [ ] **Step 1: Write failing Starfysh tests**

Create tests that define a small spatial count AnnData and a signature matrix, call `Starfysh.setup_anndata`, construct the model, run a tiny training pass, and check proportions shape and import/export behavior.

- [ ] **Step 2: Run Starfysh tests and verify failure**

Run: `pytest tests/external/test_starfysh.py -v`

Expected: fail because `scviva.external.starfysh` does not exist.

---

## Task 4: Starfysh Minimal Implementation

**Files:**
- Create: `src/scviva/external/starfysh/__init__.py`
- Create: `src/scviva/external/starfysh/_module.py`
- Create: `src/scviva/external/starfysh/_model.py`
- Modify: `src/scviva/external/__init__.py`
- Test: `tests/external/test_starfysh.py`

- [ ] **Step 1: Add expression-only `StarfyshModule`**

Implement a compact AVAE-like expression model with cell-type proportions, latent expression, library-size scaling, and a negative-binomial-inspired reconstruction objective.

- [ ] **Step 2: Add `Starfysh` wrapper**

Use `SpatialDeconvolutionMixin`, `UnsupervisedTrainingMixin`, and `SpatialBaseModel`. Register counts and spatial coordinates. Store signature matrix and cell-type names on the model. Add `get_proportions` and a small training override.

- [ ] **Step 3: Export Starfysh**

Expose `Starfysh` and `StarfyshModule` from `scviva.external.starfysh` and `scviva.external`.

- [ ] **Step 4: Run Starfysh tests**

Run: `pytest tests/external/test_starfysh.py -v`

Expected: pass.

---

## Task 5: Architecture Diagram Phase 1 Update

**Files:**
- Modify: `docs/architecture/scviva-tools-block-diagram.html`

- [ ] **Step 1: Add external model cards**

Add `AMICI` and `Starfysh` cards to the external model group in the spatial transcriptomics track.

- [ ] **Step 2: Add Phase 1 chips**

Mark AMICI with `SpatialBaseModel`, neighbor-aware attention, and Phase 1 core. Mark Starfysh with `SpatialBaseModel`, `DeconvolutionMixin`, expression AVAE, and Phase 1 core.

- [ ] **Step 3: Update data-flow outputs**

Add AMICI predictions/attention and Starfysh proportions to the transcriptomics output node.

---

## Task 6: Verification

**Files:**
- All files changed in Tasks 1-5.

- [ ] **Step 1: Run focused external tests**

Run: `pytest tests/external/test_amici.py tests/external/test_starfysh.py -v`

Expected: pass.

- [ ] **Step 2: Run existing external regression surface**

Run: `pytest tests/external -v`

Expected: pass or fail only for pre-existing unrelated dependency issues that are documented in the final report.

- [ ] **Step 3: Inspect changed files**

Run: `git status --short`

Expected: new AMICI/Starfysh files, tests, docs updates, the design spec, and this plan. No commit.

---

## Self-Review Notes

This plan covers Phase 1 from the design spec: minimal AMICI wrapper/module/setup/test surface, minimal expression-only Starfysh wrapper/module/setup/test surface, tests in `tests/external`, architecture alignment, and no commits. It intentionally excludes Starfysh PoE, Starfysh preprocessing/archetypes/plots, AMICI interpretation modules, W&B callbacks, and upstream regression equivalence.


---

# External AMICI and Starfysh Phase 2A Tutorial Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the smallest AMICI and Starfysh output APIs needed to adapt the upstream tutorials to the `scviva.external` ecosystem.

**Architecture:** Keep AMICI and Starfysh as external wrappers under `scviva.external`; do not introduce new large objects. Add small getter methods that reuse the existing module forward outputs and optionally write results back to the registered `AnnData` under predictable keys.

**Tech Stack:** Python, torch, NumPy, pandas, AnnData, pytest, scviva external model wrappers.

**User constraints:** Do not commit. Keep this phase simple. Tests stay in `tests/external`. Update `docs/architecture/scviva-tools-block-diagram.html` after the phase.

---

## File Structure

- Modify `tests/external/test_amici.py`: add tests for attention patterns, neighbor embeddings, and optional AnnData writes.
- Modify `src/scviva/external/amici/_model.py`: add shared output collection helper, `get_attention_patterns`, `get_nn_embed`, and `store_key` support for `get_predictions`.
- Modify `tests/external/test_starfysh.py`: add tests for latent representation, structured model outputs, and optional AnnData writes.
- Modify `src/scviva/external/starfysh/_model.py`: add shared output collection helper, `get_latent_representation`, `get_model_outputs`, and `store_key` support for `get_proportions`.
- Modify `docs/architecture/scviva-tools-block-diagram.html`: add the Phase 2A output names to the external model cards and output rail.

---

## Task 1: AMICI Output Tests

**Files:**
- Modify: `tests/external/test_amici.py`

- [ ] **Step 1: Write failing tests**

Add tests that train the tiny AMICI model and assert:
- `get_attention_patterns(batch_size=8, store_key="amici_attention")` returns shape `(adata.n_obs, n_neighbors)` and writes `adata.obsm["amici_attention"]`.
- `get_nn_embed(batch_size=8, store_key="X_amici_nn")` returns first dimension `adata.n_obs` and writes `adata.obsm["X_amici_nn"]`.
- `get_predictions(batch_size=8, store_key="amici_prediction")` writes predictions to `adata.obsm["amici_prediction"]`.

- [ ] **Step 2: Run AMICI tests and verify failure**

Run: `pytest tests/external/test_amici.py -v`

Expected: fail because the new AMICI methods or `store_key` argument do not exist.

---

## Task 2: AMICI Output Implementation

**Files:**
- Modify: `src/scviva/external/amici/_model.py`
- Test: `tests/external/test_amici.py`

- [ ] **Step 1: Add a private output collector**

Add `_collect_outputs(self, keys: tuple[str, ...], batch_size: int, prog_bar: bool) -> dict[str, np.ndarray]` that iterates over `_tensor_dataset()`, calls `self.module(..., compute_loss=False)`, and concatenates requested outputs along observation axis.

- [ ] **Step 2: Add public getters and storage**

Update `get_predictions(..., store_key: str | None = None)` to optionally write to `self.adata.obsm[store_key]`.

Add:

```python
def get_attention_patterns(
    self,
    batch_size: int = 128,
    store_key: str | None = None,
    prog_bar: bool = False,
) -> np.ndarray:
    ...
```

Add:

```python
def get_nn_embed(
    self,
    batch_size: int = 128,
    store_key: str | None = None,
    prog_bar: bool = False,
) -> np.ndarray:
    ...
```

- [ ] **Step 3: Run AMICI tests and verify pass**

Run: `pytest tests/external/test_amici.py -v`

Expected: pass.

---

## Task 3: Starfysh Output Tests

**Files:**
- Modify: `tests/external/test_starfysh.py`

- [ ] **Step 1: Write failing tests**

Add tests that train the tiny Starfysh model and assert:
- `get_latent_representation(batch_size=8, store_key="X_starfysh")` returns shape `(adata.n_obs, n_latent)` and writes `adata.obsm["X_starfysh"]`.
- `get_model_outputs(batch_size=8, store=True)` returns keys `qc_m`, `qz_m`, `qz_logv`, `px_rate`, and `px_scale`.
- `get_model_outputs(store=True)` writes `adata.obsm["starfysh_proportions"]`, `adata.obsm["X_starfysh"]`, and `adata.layers["starfysh_px_rate"]`.
- `get_proportions(store_key="starfysh_proportions")` writes to `adata.obsm`.

- [ ] **Step 2: Run Starfysh tests and verify failure**

Run: `pytest tests/external/test_starfysh.py -v`

Expected: fail because the new Starfysh methods or `store_key` argument do not exist.

---

## Task 4: Starfysh Output Implementation

**Files:**
- Modify: `src/scviva/external/starfysh/_model.py`
- Test: `tests/external/test_starfysh.py`

- [ ] **Step 1: Add a private output collector**

Add `_collect_outputs(self, keys: tuple[str, ...], batch_size: int) -> dict[str, np.ndarray]` that iterates over `_tensor_dataset()`, calls `self.module(..., compute_loss=False)`, and concatenates requested outputs along observation axis.

- [ ] **Step 2: Add public getters and storage**

Update `get_proportions(..., store_key: str | None = None, batch_size: int = 128)` to optionally write the values to `self.adata.obsm[store_key]`.

Add:

```python
def get_latent_representation(
    self,
    batch_size: int = 128,
    store_key: str | None = None,
) -> np.ndarray:
    ...
```

Add:

```python
def get_model_outputs(
    self,
    batch_size: int = 128,
    store: bool = False,
) -> dict[str, np.ndarray]:
    ...
```

Use default storage keys `starfysh_proportions`, `X_starfysh`, and `starfysh_px_rate`.

- [ ] **Step 3: Run Starfysh tests and verify pass**

Run: `pytest tests/external/test_starfysh.py -v`

Expected: pass.

---

## Task 5: Architecture Diagram Update

**Files:**
- Modify: `docs/architecture/scviva-tools-block-diagram.html`

- [ ] **Step 1: Update external model cards**

Add AMICI chips for `attention patterns` and `neighbor embeddings`. Add Starfysh chips for `latent representation` and `model outputs`.

- [ ] **Step 2: Update output rail**

Update the output sentence to include AMICI attention/neighbor embeddings and Starfysh latent/model-output arrays.

---

## Task 6: Verification

**Files:**
- All files changed in Tasks 1-5.

- [ ] **Step 1: Run focused lint**

Run: `python -m ruff check src/scviva/external/amici src/scviva/external/starfysh tests/external/test_amici.py tests/external/test_starfysh.py`

Expected: pass.

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/external/test_amici.py tests/external/test_starfysh.py -v`

Expected: pass.

- [ ] **Step 3: Run full external tests**

Run: `pytest tests/external -v`

Expected: pass.

- [ ] **Step 4: Inspect diff hygiene**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: only planned uncommitted files. No commit.

---

## Self-Review Notes

This plan implements the Phase 2 tutorial-enabling output contract from `docs/superpowers/specs/2026-06-01-external-amici-starfysh-design.md`. It intentionally leaves Starfysh PoE, Starfysh preprocessing/archetypes/plotting, Starfysh cell-type-specific expression, and AMICI ablation/counterfactual/explained-variance modules for later phases.


---

# AMICI and Starfysh Tutorial Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create four tutorial notebooks in `docs/tutorials/` that teach users the scviva AMICI and Starfysh APIs, porting the upstream notebooks with Phase 1–2A active cells and clearly marked stubs for unimplemented phases.

**Architecture:** Each notebook replaces raw upstream imports (`from amici import …`, `from starfysh import …`) with `from scviva.external import AMICI` / `Starfysh`. Phase 1–2A cells are fully runnable. Cells requiring unimplemented features are `# TODO: Phase N` stubs. Data paths are kept as-is from the upstream notebooks. No notebooks are executed.

**Tech Stack:** Python, nbformat JSON (v4.4), scviva, scanpy, torch, pandas, numpy, matplotlib.

**Spec:** `docs/superpowers/specs/2026-06-01-tutorial-amici-starfysh-design.md`

**User constraints:** Do not commit. Do not execute notebooks.

---

## API Reference (use these exact signatures)

```python
# AMICI
from scviva.external import AMICI

AMICI.setup_anndata(adata, labels_key="subclass", spatial_key="spatial", n_neighbors=30)
model = AMICI(adata, n_label_embed=16, n_nn_embed=64, n_hidden=128)
model.train(max_epochs=50, batch_size=512, lr=1e-3, prog_bar=True)

predictions  = model.get_predictions(store_key="amici_prediction")          # np.ndarray (n_obs, n_genes)
residuals    = model.get_predictions(get_residuals=True, store_key="amici_residual")
attention    = model.get_attention_patterns(store_key="amici_attention")     # np.ndarray (n_obs, n_neighbors)
nn_embed     = model.get_nn_embed(store_key="X_amici_nn")                   # np.ndarray (n_obs, n_nn_embed)

# Starfysh
from scviva.external import Starfysh

Starfysh.setup_anndata(adata, layer=None, spatial_key="spatial")
model = Starfysh(adata, signature_scores=signature_scores)   # signature_scores: DataFrame (n_obs, n_cell_types)
model.train(max_epochs=100, batch_size=128, lr=1e-3, prog_bar=True)

proportions = model.get_proportions(store_key="starfysh_proportions")       # DataFrame (n_obs, n_cell_types)
latent      = model.get_latent_representation(store_key="X_starfysh")       # np.ndarray (n_obs, n_latent)
outputs     = model.get_model_outputs(store=True)                           # dict of np.ndarray
```

**`compute_signature_scores` helper** (needed by all Starfysh tutorials; define once per notebook):

```python
def compute_signature_scores(adata, gene_sig):
    """Compute per-spot mean log-normalized expression for each cell-type marker set."""
    import scipy.sparse as sp
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    scores = pd.DataFrame(index=adata.obs_names)
    for ct in gene_sig.columns:
        markers = [g for g in gene_sig[ct].dropna().tolist() if g in adata.var_names]
        if markers:
            X_sub = adata_norm[:, markers].X
            if sp.issparse(X_sub):
                X_sub = X_sub.toarray()
            scores[ct] = np.asarray(X_sub).mean(axis=1)
        else:
            scores[ct] = 0.0
    # clip to [0, inf] and normalise rows to sum to 1
    scores = scores.clip(lower=0)
    row_sums = scores.sum(axis=1).replace(0, np.nan)
    scores = scores.div(row_sums, axis=0).fillna(1.0 / scores.shape[1])
    return scores
```

---

## File Structure

- Create: `docs/tutorials/amici_tutorial.ipynb`
- Create: `docs/tutorials/starfysh_tutorial_simulation.ipynb`
- Create: `docs/tutorials/starfysh_slideseq_tutorial.ipynb`
- Create: `docs/tutorials/starfysh_tutorial_integration.ipynb`
- Modify: `docs/tutorials/index.md`

---

## Task 1: AMICI Tutorial Notebook

**Files:**
- Create: `docs/tutorials/amici_tutorial.ipynb`

- [ ] **Step 1: Create `amici_tutorial.ipynb`**

Write the file at `docs/tutorials/amici_tutorial.ipynb` with the following cells (use the Write tool with this exact JSON structure):

```python
import json, os

cells = []

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells.append(md(
    "# AMICI Tutorial: Neighborhood-Aware Spatial Expression Prediction\n\n"
    "AMICI models spatial transcriptomics by attending over the expression of neighbouring cells "
    "to predict target-cell gene expression. This tutorial uses the `scviva.external.AMICI` wrapper.\n\n"
    "**Data:** Mouse visual cortex spatial transcriptomics "
    "([figshare, 58303438](https://figshare.com/ndownloader/files/58303438)). "
    "Download runs automatically on first use via `sc.read`."
))

cells.append(code(
    "!pip install scviva-tools"
))

cells.append(code(
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "import os\n"
    "import torch\n"
    "import scanpy as sc\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import seaborn as sns\n"
    "import matplotlib.pyplot as plt\n\n"
    "import scviva\n"
    "from scviva.external import AMICI"
))

cells.append(code(
    "scviva.settings.seed = 0\n"
    "sc.set_figure_params(figsize=(6, 6), frameon=False)\n"
    "print('scviva version:', scviva.__version__)"
))

cells.append(md("## Load the Data\n\nLoad the AnnData and store spatial coordinates in `adata.obsm['spatial']`."))

cells.append(code(
    "adata_path = './data/mouse_cortex_tutorial.h5ad'\n"
    "adata = sc.read(adata_path, backup_url='https://figshare.com/ndownloader/files/58303438')\n\n"
    "adata.obsm['spatial'] = adata.obs[['centroid_x', 'centroid_y']].values\n\n"
    "adata_train = adata[adata.obs['in_test'] == False].copy()\n"
    "adata_test  = adata[adata.obs['in_test'] == True].copy()\n\n"
    "labels_key = 'subclass'\n"
    "print('Train:', adata_train.shape, '| Test:', adata_test.shape)"
))

cells.append(code(
    "CELL_TYPE_PALETTE = {\n"
    "    'L2/3 IT': '#e41a1c', 'L4/5 IT': '#ff7f00', 'L5 IT': '#fdbf6f',\n"
    "    'L5 ET': '#e31a1c',   'L6 IT': '#6a3d9a',   'L6 IT Car3': '#cab2d6',\n"
    "    'L6 CT': '#fb9a99',   'L5/6 NP': '#a6cee3', 'L6b': '#1f78b4',\n"
    "    'Pvalb': '#8dd3c7',   'Sst': '#80b1d3',     'Lamp5': '#33a02c',\n"
    "    'Vip': '#b2df8a',     'Sncg': '#bc80bd',    'Astro': '#bebada',\n"
    "    'Oligo': '#fb8072',   'OPC': '#b3de69',     'Micro': '#fccde5',\n"
    "    'VLMC': '#d9d9d9',    'Endo': '#ffff33',    'Peri': '#ffffb3',\n"
    "    'PVM': '#fdb462',     'SMC': '#8dd3c7',     'other': '#999999',\n"
    "}\n\n"
    "def visualize_spatial_distribution(adata, labels_key='subclass'):\n"
    "    plot_df = pd.DataFrame(adata.obsm['spatial'].copy(), columns=['X', 'Y'])\n"
    "    plot_df[labels_key] = adata.obs[labels_key].values\n"
    "    plot_df['in_test'] = adata.obs['in_test'].values\n"
    "    plt.figure(figsize=(8, 6))\n"
    "    sns.scatterplot(plot_df, x='X', y='Y', hue=labels_key, alpha=0.7, s=8, palette=CELL_TYPE_PALETTE)\n"
    "    test_df = plot_df[plot_df['in_test']]\n"
    "    if len(test_df) > 0:\n"
    "        mn_x, mx_x = test_df['X'].min(), test_df['X'].max()\n"
    "        mn_y, mx_y = test_df['Y'].min(), test_df['Y'].max()\n"
    "        pad = 20\n"
    "        plt.gca().add_patch(plt.Rectangle((mn_x-pad, mn_y-pad), mx_x-mn_x+2*pad, mx_y-mn_y+2*pad,\n"
    "            fill=False, color='black', linestyle='--', linewidth=2, label='Test Region'))\n"
    "    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)\n"
    "    plt.tight_layout(); plt.show()\n\n"
    "visualize_spatial_distribution(adata)"
))

cells.append(md(
    "## Setup and Train AMICI\n\n"
    "`AMICI.setup_anndata` computes spatial neighbours and registers all required fields.\n"
    "The model constructor takes simplified architecture params: `n_label_embed`, `n_nn_embed`, `n_hidden`."
))

cells.append(code(
    "torch.manual_seed(18)\n\n"
    "model_params = dict(n_label_embed=16, n_nn_embed=64, n_hidden=128)\n"
    "n_neighbors = 30"
))

cells.append(code(
    "AMICI.setup_anndata(\n"
    "    adata_train,\n"
    "    labels_key=labels_key,\n"
    "    spatial_key='spatial',\n"
    "    n_neighbors=n_neighbors,\n"
    ")\n"
    "model = AMICI(adata_train, **model_params)\n"
    "print(model)"
))

cells.append(code(
    "model.train(\n"
    "    max_epochs=50,\n"
    "    batch_size=512,\n"
    "    lr=1e-3,\n"
    "    prog_bar=True,\n"
    ")"
))

cells.append(md(
    "## Get Predictions\n\n"
    "Run `setup_anndata` on the full `adata` (train + test) before calling inference methods.\n"
    "All outputs are stored in `adata.obsm` under the provided `store_key`."
))

cells.append(code(
    "AMICI.setup_anndata(\n"
    "    adata,\n"
    "    labels_key=labels_key,\n"
    "    spatial_key='spatial',\n"
    "    n_neighbors=n_neighbors,\n"
    ")\n\n"
    "predictions = model.get_predictions(store_key='amici_prediction')\n"
    "residuals   = model.get_predictions(get_residuals=True, store_key='amici_residual')\n"
    "attention   = model.get_attention_patterns(store_key='amici_attention')\n"
    "nn_embed    = model.get_nn_embed(store_key='X_amici_nn')\n\n"
    "print('Predictions shape:', predictions.shape)\n"
    "print('Residuals shape:  ', residuals.shape)\n"
    "print('Attention shape:  ', attention.shape)\n"
    "print('NN Embed shape:   ', nn_embed.shape)"
))

cells.append(md("## Save / Load Model"))

cells.append(code(
    "# TODO: Phase 2 — model.save() / AMICI.load()\n"
    "# model.save('./saved_models/amici_cortex', overwrite=True)\n"
    "# model = AMICI.load('./saved_models/amici_cortex', adata=adata)"
))

cells.append(md(
    "## Downstream Interpretation\n\n"
    "The sections below show the full interpretation workflow from the upstream AMICI package.\n"
    "These capabilities will be available in scviva once Phase 5 lands."
))

cells.append(code(
    "# TODO: Phase 5 — AMICIAblationModule (high-level interaction scores, per-gene ablation)\n"
    "# from scviva.external.amici import AMICIAblationModule\n"
    "#\n"
    "# ablation = model.get_neighbor_ablation_scores(adata=adata, compute_z_value=True)\n"
    "# interaction_df = ablation._get_interaction_weight_matrix()\n"
    "# ablation.plot_interaction_directed_graph(significance_threshold=0.05)\n"
    "# ablation.plot_featurewise_contributions_dotplot(cell_type='Astro', n_top_genes=5)"
))

cells.append(code(
    "# TODO: Phase 5 — AMICICounterfactualAttentionModule\n"
    "# from scviva.external.amici import AMICICounterfactualAttentionModule\n"
    "#\n"
    "# cf = model.get_counterfactual_attention_patterns(cell_type='Astro', adata=adata)\n"
    "# cf.plot_length_scale_distribution(head_idxs=range(model.module.n_heads),\n"
    "#                                    sender_types=['L4/5 IT', 'L2/3 IT', 'Oligo'])"
))

cells.append(code(
    "# TODO: Phase 5 — AMICIAttentionModule (empirical attention summary)\n"
    "# from scviva.external.amici import AMICIAttentionModule\n"
    "#\n"
    "# attn_module = AMICIAttentionModule.from_model(model, adata)\n"
    "# attn_module.plot_attention_summary(cell_type_sub=['Astro'])"
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = '/Users/orikr/PycharmProjects/spatialvi-tools-chatgpt/docs/tutorials/amici_tutorial.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print('Written:', out)
```

Run this script:
```bash
python3 -c "exec(open('/dev/stdin').read())" << 'PYEOF'
# ... paste the script above
PYEOF
```

Or use the Write tool directly (preferred) to create the file from the JSON that the script above generates. The Write tool approach: compose the full `nb` dict with all cells and write it as JSON.

- [ ] **Step 2: Verify AMICI notebook**

```bash
python3 -c "
import json
with open('docs/tutorials/amici_tutorial.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']
sources = [(''.join(c['source']), c['cell_type']) for c in cells]

# Must have scviva import, not amici raw import
assert any('from scviva.external import AMICI' in s for s,_ in sources), 'missing scviva AMICI import'
assert not any('from amici import AMICI' in s for s,_ in sources), 'raw amici import found'
assert any('AMICI.setup_anndata' in s for s,_ in sources), 'missing setup_anndata'
assert any('model.train' in s for s,_ in sources), 'missing train'
assert any('get_predictions' in s for s,_ in sources), 'missing get_predictions'
assert any('TODO: Phase 5' in s for s,_ in sources), 'missing Phase 5 stubs'
assert any('TODO: Phase 2' in s for s,_ in sources), 'missing Phase 2 stub'
print('AMICI notebook OK:', len(cells), 'cells')
"
```

Expected output: `AMICI notebook OK: 18 cells`

---

## Task 2: Starfysh Simulation Tutorial Notebook

**Files:**
- Create: `docs/tutorials/starfysh_tutorial_simulation.ipynb`

- [ ] **Step 1: Create `starfysh_tutorial_simulation.ipynb`**

Write the file with the following cells:

```
Cell 0 [markdown]:
# Starfysh Tutorial: Spatial Deconvolution on Simulated ST Data

Starfysh jointly embeds spatial gene expression and cell-type signatures to perform
spot-level deconvolution. This tutorial uses `scviva.external.Starfysh` on a simulated
spatial transcriptomics dataset.

**Data:** Simulated ST data (`simulated_ST_data_1`) generated from scRNA-seq.
Download from [Google Drive](https://drive.google.com/drive/folders/1bLV37YzJle7Wq-q0HFIi6s8XG5sX9EVa).
Place data under `../data/simulated_ST_data_1/`.

Cell 1 [code]:
!pip install scviva-tools

Cell 2 [code]:
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import rcParams

import scviva
from scviva.external import Starfysh

rcParams.update({'font.size': 10, 'figure.dpi': 100, 'figure.figsize': (4, 4)})
print('scviva version:', scviva.__version__)

Cell 3 [markdown]:
## Load Data and Marker Genes

Cell 4 [code]:
data_path = '../data/'
sample_id = 'simulated_ST_data_1'

# Load expression count matrix.
# The original tutorial uses starfysh.utils.load_adata; here we use scanpy directly.
adata = sc.read_h5ad(os.path.join(data_path, sample_id, 'adata.h5ad'))
sc.pp.filter_genes(adata, min_counts=1)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat', subset=True)
print('adata:', adata.shape)

# Load marker gene signatures (genes × cell-types)
gene_sig = pd.read_csv(os.path.join(data_path, 'tnbc_signature.csv'), index_col=0)
print('Signatures:', gene_sig.shape, '| Cell types:', list(gene_sig.columns))
gene_sig.head()

Cell 5 [code]:
# Set spatial coordinates.
# For Visium: adata.obsm['spatial'] is already set by sc.read_visium.
# For this simulation dataset, spatial coordinates are in a CSV next to the expression file.
map_info = pd.read_csv(
    os.path.join(data_path, sample_id, 'spot_list.csv'), index_col=0
)
adata.obsm['spatial'] = map_info[['array_row', 'array_col']].loc[adata.obs_names].values
print('Spatial coords shape:', adata.obsm['spatial'].shape)

Cell 6 [markdown]:
## Compute Per-Spot Signature Scores

Starfysh requires per-spot prior scores (one value per cell type per spot).
`compute_signature_scores` derives these as the mean log-normalised expression
of marker genes in each cell type.

Cell 7 [code]:
def compute_signature_scores(adata, gene_sig):
    """Compute per-spot mean log-normalised expression for each cell-type marker set."""
    import scipy.sparse as sp
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    scores = pd.DataFrame(index=adata.obs_names)
    for ct in gene_sig.columns:
        markers = [g for g in gene_sig[ct].dropna().tolist() if g in adata.var_names]
        if markers:
            X_sub = adata_norm[:, markers].X
            if sp.issparse(X_sub):
                X_sub = X_sub.toarray()
            scores[ct] = np.asarray(X_sub).mean(axis=1)
        else:
            scores[ct] = 0.0
    scores = scores.clip(lower=0)
    row_sums = scores.sum(axis=1).replace(0, np.nan)
    return scores.div(row_sums, axis=0).fillna(1.0 / scores.shape[1])

signature_scores = compute_signature_scores(adata, gene_sig)
print('Signature scores:', signature_scores.shape)
signature_scores.head()

Cell 8 [markdown]:
## Preprocessing (Phase 6)

The upstream Starfysh workflow includes `VisiumArguments` for windowed library-size
smoothing, anchor spot detection, and refined signature loading.

Cell 9 [code]:
# TODO: Phase 6 — VisiumArguments preprocessing, windowed library smoothing, anchor detection
# from scviva.external.starfysh import VisiumArguments
#
# visium_args = VisiumArguments(adata, adata_normed, gene_sig, img_metadata, window_size=1)
# adata, adata_normed = visium_args.get_adata()
# anchors_df = visium_args.get_anchors()

Cell 10 [markdown]:
## Archetypal Analysis (Phase 6)

Archetypal analysis can be used to refine marker genes and anchor spots.

Cell 11 [code]:
# TODO: Phase 6 — ArchetypalAnalysis, anchor refinement
# from scviva.external.starfysh import ArchetypalAnalysis
#
# aa_model = ArchetypalAnalysis(adata_orig=adata)
# archetype, arche_dict, major_idx, evs = aa_model.compute_archetypes(converge=1e-2)
# arche_df = aa_model.find_archetypal_spots(major=True)
# markers_df = aa_model.find_markers(display=False)
# visium_args = refine_anchors(visium_args, aa_model)

Cell 12 [markdown]:
## Setup and Train Starfysh

Cell 13 [code]:
Starfysh.setup_anndata(adata, spatial_key='spatial')
model = Starfysh(adata, signature_scores=signature_scores)
print(model)

Cell 14 [code]:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.train(max_epochs=100, batch_size=128, lr=1e-3, device=device, prog_bar=True)

Cell 15 [markdown]:
## Downstream Analysis

Cell 16 [code]:
proportions = model.get_proportions(store_key='starfysh_proportions')
print('Proportions shape:', proportions.shape)
proportions.head()

Cell 17 [code]:
latent = model.get_latent_representation(store_key='X_starfysh')
outputs = model.get_model_outputs(store=True)
print('Latent shape:', latent.shape)
print('Output keys:', list(outputs.keys()))

Cell 18 [markdown]:
## Histology Integration / PoE (Phase 4)

Starfysh can optionally incorporate a paired H&E image via a Product-of-Experts (PoE) module.

Cell 19 [code]:
# TODO: Phase 4 — StarfyshPoEModule (histology + PoE integration)
# from scviva.external.starfysh import StarfyshPoEModule
#
# model_poe = Starfysh(adata, signature_scores=signature_scores, use_poe=True,
#                      image_patches=img_patches)
# model_poe.train(max_epochs=100, batch_size=128)

Cell 20 [markdown]:
## Spatial Visualizations (Phase 6)

Cell 21 [code]:
# TODO: Phase 6 — spatial visualizations and cell-type expression plots
# from scviva.external.starfysh import plot_spatial_inf_feature, plot_spatial_inf_gene
#
# plot_spatial_inf_feature(adata, feature='qc_m', spot_size=30, vmax=1)
# plot_spatial_inf_gene(adata, factor='T-cells', feature='CD69', spot_size=30)

Cell 22 [markdown]:
## Cell-Type-Specific Expression (Phase 2)

Cell 23 [code]:
# TODO: Phase 2 — cell-type-specific inferred expression
# pred_exprs = model.get_cell_type_expression()
# # pred_exprs is a dict {cell_type: np.ndarray (n_obs, n_genes)}

Cell 24 [markdown]:
## Save Model

Cell 25 [code]:
# Save model state and annotated data
outdir = './results/'
os.makedirs(outdir, exist_ok=True)
torch.save(model.module.state_dict(), os.path.join(outdir, 'starfysh_model.pt'))
adata.write(os.path.join(outdir, 'st.h5ad'))
print('Saved model and adata to', outdir)
```

- [ ] **Step 2: Verify Starfysh simulation notebook**

```bash
python3 -c "
import json
with open('docs/tutorials/starfysh_tutorial_simulation.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']
sources = [''.join(c['source']) for c in cells]
all_src = '\n'.join(sources)

assert 'from scviva.external import Starfysh' in all_src, 'missing scviva Starfysh import'
assert 'from starfysh import' not in all_src, 'raw starfysh import found'
assert 'Starfysh.setup_anndata' in all_src, 'missing setup_anndata'
assert 'model.train' in all_src, 'missing train'
assert 'get_proportions' in all_src, 'missing get_proportions'
assert 'compute_signature_scores' in all_src, 'missing compute_signature_scores helper'
assert 'TODO: Phase 6' in all_src, 'missing Phase 6 stubs'
assert 'TODO: Phase 4' in all_src, 'missing Phase 4 stub'
print('Starfysh simulation notebook OK:', len(cells), 'cells')
"
```

Expected output: `Starfysh simulation notebook OK: 26 cells`

---

## Task 3: Starfysh Slideseq Tutorial Notebook

**Files:**
- Create: `docs/tutorials/starfysh_slideseq_tutorial.ipynb`

- [ ] **Step 1: Create `starfysh_slideseq_tutorial.ipynb`**

Write the file with the following cells:

```
Cell 0 [markdown]:
# Starfysh Tutorial: Spatial Deconvolution on Slide-seq Data

This tutorial applies `scviva.external.Starfysh` to Slide-seq data loaded from raw CSV
coordinates and count files. The preprocessing helper below recreates the `preprocess_slideseq`
function from the upstream Starfysh package using scanpy directly.

**Data:** MPM08 Slide-seq mesothelioma sample.
Place `MPM08_on_later_coords.csv` and `MPM08_on_later_counts.csv` in a `spatial/` subdirectory.

Cell 1 [code]:
!pip install scviva-tools

Cell 2 [code]:
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib import rcParams

import scviva
from scviva.external import Starfysh

rcParams.update({'font.size': 10, 'figure.dpi': 100, 'figure.figsize': (4, 4)})
print('scviva version:', scviva.__version__)

Cell 3 [markdown]:
## Load Slide-seq Data

The original `preprocess_slideseq` function reads a counts CSV and a coordinates CSV
into an AnnData with `obsm['spatial']` set. We replicate it here with standard libraries.

Cell 4 [code]:
def preprocess_slideseq(path_to_coords, path_to_counts):
    """Load Slide-seq counts + spatial coords into AnnData."""
    counts = pd.read_csv(path_to_counts, index_col=0)
    coords = pd.read_csv(path_to_coords, index_col=0)

    # Keep spots present in both files
    shared = counts.index.intersection(coords.index)
    counts = counts.loc[shared]
    coords = coords.loc[shared]

    adata = ad.AnnData(X=counts.values.astype('float32'),
                       obs=pd.DataFrame(index=counts.index),
                       var=pd.DataFrame(index=counts.columns))
    adata.obsm['spatial'] = coords.values.astype('float32')
    sc.pp.filter_genes(adata, min_counts=1)
    return adata

adata = preprocess_slideseq(
    'spatial/MPM08_on_later_coords.csv',
    'spatial/MPM08_on_later_counts.csv',
)
print('adata:', adata.shape)

Cell 5 [code]:
# Select highly variable genes for modelling
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat', subset=True)
print('After HVG selection:', adata.shape)

Cell 6 [markdown]:
## Load Marker Gene Signatures

Cell 7 [code]:
gene_sig = pd.read_csv('full_sigs.csv', index_col=0)
print('Signatures:', gene_sig.shape)
gene_sig.head()

Cell 8 [markdown]:
## Compute Per-Spot Signature Scores

Cell 9 [code]:
def compute_signature_scores(adata, gene_sig):
    """Compute per-spot mean log-normalised expression for each cell-type marker set."""
    import scipy.sparse as sp
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    scores = pd.DataFrame(index=adata.obs_names)
    for ct in gene_sig.columns:
        markers = [g for g in gene_sig[ct].dropna().tolist() if g in adata.var_names]
        if markers:
            X_sub = adata_norm[:, markers].X
            if sp.issparse(X_sub):
                X_sub = X_sub.toarray()
            scores[ct] = np.asarray(X_sub).mean(axis=1)
        else:
            scores[ct] = 0.0
    scores = scores.clip(lower=0)
    row_sums = scores.sum(axis=1).replace(0, np.nan)
    return scores.div(row_sums, axis=0).fillna(1.0 / scores.shape[1])

signature_scores = compute_signature_scores(adata, gene_sig)
print('Signature scores:', signature_scores.shape)

Cell 10 [markdown]:
## Preprocessing Stubs (Phase 6)

Cell 11 [code]:
# TODO: Phase 6 — windowed library-size smoothing and anchor detection
# from scviva.external.starfysh import get_windowed_library, get_anchor_spots
#
# win_loglib = get_windowed_library(adata, map_info, window_size=3)
# pure_spots, pure_dict, pure_idx = get_anchor_spots(adata, sig_mean, n_anchor=60)

Cell 12 [markdown]:
## Setup and Train Starfysh

Cell 13 [code]:
Starfysh.setup_anndata(adata, spatial_key='spatial')
model = Starfysh(adata, signature_scores=signature_scores)
print(model)

Cell 14 [code]:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.train(max_epochs=100, batch_size=128, lr=1e-3, device=device, prog_bar=True)

Cell 15 [markdown]:
## Downstream Analysis

Cell 16 [code]:
proportions = model.get_proportions(store_key='starfysh_proportions')
print('Proportions shape:', proportions.shape)
proportions.head()

Cell 17 [code]:
latent  = model.get_latent_representation(store_key='X_starfysh')
outputs = model.get_model_outputs(store=True)
print('Latent shape:', latent.shape)
print('Output keys:', list(outputs.keys()))

Cell 18 [markdown]:
## Spatial Visualizations (Phase 6)

Cell 19 [code]:
# TODO: Phase 6 — spatial density, proportions, and gene expression plots
# from scviva.external.starfysh import plot_spatial_inf_feature
#
# plot_spatial_inf_feature(adata, feature='qc_m', spot_size=30, vmax=1)

Cell 20 [markdown]:
## Save Model

Cell 21 [code]:
outdir = './results/'
os.makedirs(outdir, exist_ok=True)
torch.save(model.module.state_dict(), os.path.join(outdir, 'starfysh_slideseq_model.pt'))
adata.write(os.path.join(outdir, 'slideseq_st.h5ad'))
print('Saved to', outdir)
```

- [ ] **Step 2: Verify slideseq notebook**

```bash
python3 -c "
import json
with open('docs/tutorials/starfysh_slideseq_tutorial.ipynb') as f:
    nb = json.load(f)
sources = '\n'.join(''.join(c['source']) for c in nb['cells'])
assert 'from scviva.external import Starfysh' in sources
assert 'from starfysh import' not in sources
assert 'Starfysh.setup_anndata' in sources
assert 'preprocess_slideseq' in sources
assert 'compute_signature_scores' in sources
assert 'TODO: Phase 6' in sources
print('Slideseq notebook OK:', len(nb[\"cells\"]), 'cells')
"
```

Expected output: `Slideseq notebook OK: 22 cells`

---

## Task 4: Starfysh Integration Tutorial Notebook

**Files:**
- Create: `docs/tutorials/starfysh_tutorial_integration.ipynb`

- [ ] **Step 1: Create `starfysh_tutorial_integration.ipynb`**

Write the file with the following cells:

```
Cell 0 [markdown]:
# Starfysh Tutorial: Multi-Sample Integration

This tutorial shows how to run `scviva.external.Starfysh` independently on each sample,
then combine the resulting latent representations and proportions for joint downstream analysis.

The upstream Starfysh package offers a `utils_integrate.run_starfysh` convenience wrapper
for joint training; that capability is deferred to Phase 7 in scviva. The per-sample workflow
shown here produces equivalent single-sample outputs ready for concatenation.

**Data:** ER and TNBC breast cancer Visium samples (`P1A_ER`, `CID44971_TNBC`).
Place each sample's `adata.h5ad` in `../data/<sample_id>/`.

Cell 1 [code]:
!pip install scviva-tools

Cell 2 [code]:
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib import rcParams

import scviva
from scviva.external import Starfysh

rcParams.update({'font.size': 10, 'figure.dpi': 100, 'figure.figsize': (4, 4)})
print('scviva version:', scviva.__version__)

Cell 3 [markdown]:
## Load Datasets and Marker Genes

Cell 4 [code]:
data_path = '../data/'
sig_file_name = 'tnbc_signature.csv'

# Sample metadata: [sample_id, file_stem, subtype]
meta_info = pd.DataFrame([
    ['P1A_ER',      'P1_ER',         'ER'],
    ['CID44971',    'CID44971_TNBC', 'TNBC'],
], columns=['sample', 'file_stem', 'subtype'])

# Load marker gene signatures
gene_sig = pd.read_csv(os.path.join(data_path, sig_file_name), index_col=0)
print('Signatures:', gene_sig.shape)

Cell 5 [code]:
# Load per-sample expression data
adatas = {}
for _, row in meta_info.iterrows():
    adata = sc.read_h5ad(os.path.join(data_path, row['sample'], 'adata.h5ad'))
    adata.obs['sample'] = row['sample']
    adata.obs['subtype'] = row['subtype']
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat', subset=True)
    adatas[row['sample']] = adata
    print(row['sample'], adata.shape)

Cell 6 [markdown]:
## Compute Per-Spot Signature Scores

Cell 7 [code]:
def compute_signature_scores(adata, gene_sig):
    """Compute per-spot mean log-normalised expression for each cell-type marker set."""
    import scipy.sparse as sp
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    scores = pd.DataFrame(index=adata.obs_names)
    for ct in gene_sig.columns:
        markers = [g for g in gene_sig[ct].dropna().tolist() if g in adata.var_names]
        if markers:
            X_sub = adata_norm[:, markers].X
            if sp.issparse(X_sub):
                X_sub = X_sub.toarray()
            scores[ct] = np.asarray(X_sub).mean(axis=1)
        else:
            scores[ct] = 0.0
    scores = scores.clip(lower=0)
    row_sums = scores.sum(axis=1).replace(0, np.nan)
    return scores.div(row_sums, axis=0).fillna(1.0 / scores.shape[1])

sig_scores = {sid: compute_signature_scores(ad, gene_sig) for sid, ad in adatas.items()}

Cell 8 [markdown]:
## Setup and Train Starfysh Per Sample

Cell 9 [code]:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = {}

for sample_id, adata in adatas.items():
    print(f'--- {sample_id} ---')
    Starfysh.setup_anndata(adata, spatial_key='spatial')
    model = Starfysh(adata, signature_scores=sig_scores[sample_id])
    model.train(max_epochs=100, batch_size=128, lr=1e-3, device=device, prog_bar=True)
    model.get_proportions(store_key='starfysh_proportions')
    model.get_latent_representation(store_key='X_starfysh')
    model.get_model_outputs(store=True)
    models[sample_id] = model
    print(f'  proportions: {adata.obsm[\"starfysh_proportions\"].shape}')

Cell 10 [markdown]:
## Concatenate and Explore Joint Embeddings

Cell 11 [code]:
# Concatenate across samples for joint downstream analysis
adata_all = ad.concat(list(adatas.values()), label='sample', keys=list(adatas.keys()))

# Compute UMAP on Starfysh latent space
sc.pp.neighbors(adata_all, use_rep='X_starfysh', n_neighbors=15)
sc.tl.umap(adata_all)
sc.pl.umap(adata_all, color='sample', title='Starfysh latent — by sample')

Cell 12 [code]:
# Visualise deconvolution on UMAP
for ct in gene_sig.columns[:4]:
    if ct in adata_all.obsm.get('starfysh_proportions', pd.DataFrame()).columns:
        adata_all.obs[ct] = adata_all.obsm['starfysh_proportions'][ct].values

sc.pl.umap(adata_all, color=list(gene_sig.columns[:4]), ncols=2)

Cell 13 [markdown]:
## Joint Integration via utils_integrate (Phase 7)

The upstream Starfysh package provides `utils_integrate.run_starfysh` for joint training
across samples with optional PoE histology integration. This will be available in scviva Phase 7.

Cell 14 [code]:
# TODO: Phase 7 — utils_integrate.run_starfysh (joint multi-sample training)
# from scviva.external.starfysh import IntegratedStarfysh
#
# model_integrated = IntegratedStarfysh(adatas, signature_scores=sig_scores, poe_on=False)
# model_integrated.train(max_epochs=100, n_repeats=1, device=device)
# adata_integrated = model_integrated.get_integrated_adata()

Cell 15 [markdown]:
## Spatial Hub Calculation (Phase 6)

Cell 16 [code]:
# TODO: Phase 6 — phenograph clustering on Starfysh embeddings, spatial hub plots
# import scanpy.external as sce
# sce.tl.phenograph(adata_all, clustering_algo='louvain', k=50)
# plot_integrated_spatial_feature(adata_all, ...)

Cell 17 [markdown]:
## Save Results

Cell 18 [code]:
outdir = './results/integration/'
os.makedirs(outdir, exist_ok=True)
for sample_id, adata in adatas.items():
    adata.write(os.path.join(outdir, f'{sample_id}.h5ad'))
    torch.save(models[sample_id].module.state_dict(),
               os.path.join(outdir, f'{sample_id}_model.pt'))
print('Saved to', outdir)
```

- [ ] **Step 2: Verify integration notebook**

```bash
python3 -c "
import json
with open('docs/tutorials/starfysh_tutorial_integration.ipynb') as f:
    nb = json.load(f)
sources = '\n'.join(''.join(c['source']) for c in nb['cells'])
assert 'from scviva.external import Starfysh' in sources
assert 'from starfysh import' not in sources
assert 'Starfysh.setup_anndata' in sources
assert 'compute_signature_scores' in sources
assert 'TODO: Phase 7' in sources
print('Integration notebook OK:', len(nb[\"cells\"]), 'cells')
"
```

Expected output: `Integration notebook OK: 19 cells`

---

## Task 5: Update `docs/tutorials/index.md`

**Files:**
- Modify: `docs/tutorials/index.md`

- [ ] **Step 1: Append four new entries**

Current `index.md` ends with `tangram_scvi_tools`. Append the four new entries so the file reads:

```markdown
# Tutorials

```{toctree}
:maxdepth: 1

resolVI_tutorial
DestVI_tutorial
scVIVA_tutorial
gimvi_tutorial
stereoscope_heart_LV_tutorial
cell2location_lymph_node_spatial_tutorial
tangram_scvi_tools
amici_tutorial
starfysh_tutorial_simulation
starfysh_slideseq_tutorial
starfysh_tutorial_integration
```
```

- [ ] **Step 2: Verify index.md**

```bash
python3 -c "
txt = open('docs/tutorials/index.md').read()
for name in ['amici_tutorial', 'starfysh_tutorial_simulation',
             'starfysh_slideseq_tutorial', 'starfysh_tutorial_integration']:
    assert name in txt, f'missing {name}'
print('index.md OK')
"
```

Expected output: `index.md OK`

---

## Task 6: Final Verification

**Files:**
- All files created in Tasks 1–5.

- [ ] **Step 1: Validate all four notebooks are valid JSON**

```bash
python3 -c "
import json, pathlib
for p in pathlib.Path('docs/tutorials').glob('*.ipynb'):
    nb = json.loads(p.read_text())
    assert nb['nbformat'] == 4
    print('OK', p.name, len(nb['cells']), 'cells')
"
```

Expected: all five existing tutorials plus the four new ones print `OK`.

- [ ] **Step 2: Check no raw upstream imports leak into active (non-stub) cells**

```bash
python3 -c "
import json, pathlib, re

BAD = ['from amici import', 'from starfysh import', 'from amici.', 'from starfysh.']
for p in ['docs/tutorials/amici_tutorial.ipynb',
          'docs/tutorials/starfysh_tutorial_simulation.ipynb',
          'docs/tutorials/starfysh_slideseq_tutorial.ipynb',
          'docs/tutorials/starfysh_tutorial_integration.ipynb']:
    nb = json.loads(pathlib.Path(p).read_text())
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell['source'])
        # Active lines: not starting with #
        active = '\n'.join(l for l in src.splitlines() if not l.strip().startswith('#'))
        for bad in BAD:
            if bad in active:
                raise AssertionError(f'{p} cell {i}: found raw import [{bad}] in active code')
    print('Clean:', pathlib.Path(p).name)
"
```

Expected: `Clean:` for all four notebooks.

- [ ] **Step 3: Confirm git status shows only new files, no commits**

```bash
git status --short docs/tutorials/
```

Expected: four `??` entries for the new notebooks (untracked), one `M` for `index.md`.

---

## Self-Review Notes

- All four notebooks use `from scviva.external import AMICI` / `Starfysh` exclusively in active cells.
- `AMICI.setup_anndata` uses `spatial_key` (not `coord_obsm_key` from the upstream notebook).
- `AMICI.train` uses the Phase 1 signature: `max_epochs`, `batch_size`, `lr`, `prog_bar` — no `callbacks`, `plan_kwargs`, or `early_stopping`.
- `Starfysh.get_proportions()` raises `ValueError` if called with a different `adata` than the registered one; tutorials call it without arguments.
- All Starfysh tutorials include the same `compute_signature_scores` helper verbatim (no shared import — each notebook is self-contained).
- save/load for AMICI is a Phase 2 stub (not in Phase 1 SpatialBaseModel).
- Phase 5 AMICI interpretation stubs name `AMICIAblationModule`, `AMICICounterfactualAttentionModule`, `AMICIAttentionModule` explicitly.
- Integration tutorial defers `utils_integrate` to Phase 7 and demonstrates the equivalent per-sample loop instead.
