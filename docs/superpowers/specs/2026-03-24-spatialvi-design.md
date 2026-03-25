# spatialvi-tools: Package Design Specification

**Date:** 2026-03-24
**Status:** Approved
**Authors:** Ori Kronfeld + Claude

---

## Overview

`spatialvi-tools` is a consolidated spatial transcriptomics analysis toolkit built on top of
`scvi-tools`. It brings together three core spatial VI models — DestVI, ResolVI, and scVIVA —
into a single, clean, scverse-compatible package, with shared base classes, spatial integrations,
and a clear extension point for future external models.

The package is intended to become part of the scverse ecosystem, following the conventions of
`scvi-tools` (anndata, scanpy, AnnDataManager, mixins) while adding spatial-specific
infrastructure (SpatialData, squidpy, RAPIDS).

---

## Decision Log

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Dependency on scvi-tools | Thin wrapper (import base classes from scvi) | Avoids reinventing VAE machinery; free upstream updates; ecosystem alignment |
| 2 | Model placement | All 3 are **core models** in `spatialvi.model` | scVIVA/ResolVI/DestVI are the primary spatial VI models, equivalent to SCVI/TOTALVI in scvi-tools |
| 3 | Shared code strategy | Hybrid A+B: `SpatialBaseModel` for universal logic + optional mixins for model-specific capabilities | Base class for all; mixins only where capabilities diverge |
| 4 | RAPIDS acceleration | `backend` parameter on relevant functions (not a mixin) | Cleaner API; no inheritance complexity; consistent with scikit-learn convention |
| 5 | Python versions | 3.12, 3.13, 3.14 (aspirational); default 3.13 | Modern Python; 3.14 is aspirational — PyTorch/scvi-tools 3.14 wheels may not exist at v1 release. CI 3.14 job uses `continue-on-error: true` |
| 6 | CI strategy | Linux CPU (multi-Python) + macOS CPU + GPU (CUDA) + pre-commit + ReadTheDocs | GPU is essential for spatial transcriptomics development |
| 7 | External folder | Reserved for future models (amici, harreman, sparl, starfysh, vivs, etc.) | Not used in v1; kept for extensibility |

---

## Architecture

### Inheritance Hierarchy

Concrete model classes inherit from both `SpatialBaseModel` and relevant mixins.
Mixins do **not** inherit from `SpatialBaseModel` — they are composed in the model class declaration.

```
scvi.model.base.BaseModelClass          (from scvi-tools, not owned)
        |
SpatialBaseModel                        (owned: spatialvi.model.base._spatial_base)

Mixins (independent, composed into model classes):
  SpatialNeighborhoodMixin              (scVIVA + ResolVI — see note below)
  SpatialDeconvolutionMixin             (DestVI only)

Model class composition (MRO left to right):
  SCVIVA   = SpatialNeighborhoodMixin + scvi mixins + SpatialBaseModel
  DestVI   = SpatialDeconvolutionMixin + SpatialBaseModel
  ResolVI  = SpatialBaseModel + ResolVIPredictiveMixin + scvi pyro mixins
```

### ResolVI and Neighbor Computation (C1 fix)

The upstream `RESOLVI._prepare_data()` (called via `setup_anndata(prepare_data=True)`) already
computes and stores neighbors in `.obsm` as `index_neighbor` and `distance_neighbor`. It also
contains its own RAPIDS branch (`rapids_singlecell` fallback).

**Decision:** `spatialvi.ResolVI` suppresses the upstream `prepare_data` step and takes
**full ownership** of neighbor computation via `SpatialNeighborhoodMixin.compute_neighbors()`.
This means:
- `spatialvi.ResolVI.setup_anndata()` calls `super().setup_anndata(prepare_data=False)`
- `SpatialNeighborhoodMixin.compute_neighbors()` is the single entry point for neighbor graphs
- The `_setup_neighbor_field()` registers the same obsm keys the module expects
- The `backend` parameter ("squidpy" | "rapids") replaces the upstream's implicit RAPIDS fallback

`SpatialNeighborhoodMixin` is applied to both `scVIVA` and `ResolVI`.

### `get_latent_representation` Override Strategy (C2 fix)

`SpatialBaseModel` defines `get_latent_representation(self, ..., backend="cpu")`.
Each concrete model class (`SCVIVA`, `DestVI`, `ResolVI`) does **not** define its own version.
Instead, MRO resolution means `SpatialBaseModel.get_latent_representation` runs for all three.

Internally, the method:
1. Calls `super().get_latent_representation(...)` (resolves to scvi's implementation via MRO)
2. If `backend="rapids"`: passes the resulting numpy array through `cuml.manifold.UMAP` or
   returns raw latent and delegates UMAP to the caller
3. If `backend="cpu"`: returns as-is

This is explicitly documented so implementers do not define `get_latent_representation` in the
model classes — it lives only in `SpatialBaseModel`.

### `ResolVIPredictiveMixin` Retention (I2 fix)

The upstream `RESOLVI` MRO includes `ResolVIPredictiveMixin` which provides:
- `get_latent_representation` (overridden by `SpatialBaseModel`)
- `get_neighbor_abundance`
- `get_normalized_expression`
- `get_normalized_expression_importance`

`spatialvi.ResolVI` retains `ResolVIPredictiveMixin` in its MRO to preserve these methods.
`SpatialBaseModel.get_latent_representation` takes precedence by appearing earlier in the MRO.

```python
# from scvi.external.resolvi._utils import ResolVIPredictiveMixin
class ResolVI(
    SpatialNeighborhoodMixin,
    SpatialBaseModel,            # get_latent_representation defined here
    PyroSviTrainMixin,           # from scvi.model.base
    PyroSampleMixin,             # from scvi.model.base
    ResolVIPredictiveMixin,      # from scvi.external.resolvi._utils — retained
    ArchesMixin,                 # from scvi.model.base
    BaseModelClass,              # from scvi.model.base
):
```

### `SpatialBaseModel` (all models inherit)

```python
# spatialvi.model.base._spatial_base
class SpatialBaseModel(BaseModelClass):  # scvi.model.base.BaseModelClass
    @classmethod
    def setup_spatialdata(cls, sdata, table_key, region, **kwargs): ...
    # extracts AnnData table from SpatialData, then calls cls.setup_anndata()

    @classmethod
    def from_spatialdata(cls, sdata, table_key, region, **model_kwargs): ...
    # calls cls.setup_spatialdata() then returns cls(sdata[table_key].to_adata(), **model_kwargs)

    def get_latent_representation(self, adata=None, ..., backend="cpu"): ...
    # wraps super() result with optional RAPIDS dispatch

    def plot_spatial_embedding(self, adata=None, basis="spatial", color=None, **kwargs): ...
    def plot_spatial_predictions(self, adata=None, key=None, **kwargs): ...
```

`setup_spatialdata` and `from_spatialdata` are both classmethods, following scvi convention.
`setup_spatialdata` = field registration (like `setup_anndata`).
`from_spatialdata` = convenience constructor that calls `setup_spatialdata` + `__init__`.

### `SpatialNeighborhoodMixin` (scVIVA + ResolVI)

```python
class SpatialNeighborhoodMixin:
    def compute_neighbors(self, adata, coord_type="generic", n_neighs=6,
                          backend="squidpy"): ...
    # backend="squidpy": uses sq.gr.spatial_neighbors
    # backend="rapids": uses cugraph.structure.graph with cupy arrays
    def _setup_neighbor_field(self, adata): ...
    # registers index_neighbor + distance_neighbor via AnnDataManager
```

### `SpatialDeconvolutionMixin` (DestVI only)

```python
class SpatialDeconvolutionMixin:
    def plot_cell_type_map(self, adata, cell_type, ax=None, **kwargs): ...
    def get_proportions_df(self, adata=None) -> pd.DataFrame: ...
```

### Import Paths for scvi Classes (I1 fix)

All scvi classes used in model composition are imported from:

```python
from scvi.model.base import (
    BaseModelClass,
    ArchesMixin,
    EmbeddingMixin,
    RNASeqMixin,
    VAEMixin,
    UnsupervisedTrainingMixin,
    BaseMinifiedModeModelClass,
    PyroSviTrainMixin,       # <-- lives in scvi.model.base, NOT scvi.train
    PyroSampleMixin,         # <-- same
)
from scvi.external.resolvi._utils import ResolVIPredictiveMixin
```

### CondSCVI / DestVI Workflow (S3 fix)

`DestVI.from_rna_model(condscvi_model, ...)` is the primary user-facing constructor for DestVI.
It requires a fitted `CondSCVI` model. `CondSCVI` is **not** wrapped or re-exported by
`spatialvi` — users import it directly from `scvi.model`:

```python
from scvi.model import CondSCVI
from spatialvi.model import DestVI

condscvi = CondSCVI(sc_adata, ...)
condscvi.train()
DestVI.from_rna_model(condscvi, st_adata)  # standard workflow preserved
```

This is documented in the DestVI tutorial and user guide. The `from_rna_model` classmethod is
inherited from the upstream implementation with no changes.

### Lazy Imports in `__init__.py` (S4 fix)

`src/spatialvi/__init__.py` uses `__getattr__` lazy loading to avoid import-time overhead from
torch/pyro, matching scvi-tools convention:

```python
# src/spatialvi/__init__.py
from importlib import import_module

_lazy_map = {
    "SCVIVA": "spatialvi.model._scviva",
    "DestVI": "spatialvi.model._destvi",
    "ResolVI": "spatialvi.model._resolvi",
}

def __getattr__(name):
    if name in _lazy_map:
        mod = import_module(_lazy_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'spatialvi' has no attribute {name!r}")
```

---

## Package Structure

```
spatialvi-tools2/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── config.yml
│   ├── pull_request_template.md
│   └── workflows/
│       ├── build.yml                    # package build check (Python 3.13)
│       ├── test_linux.yml               # Linux CPU, Python 3.12 + 3.13 + 3.14*
│       ├── test_macos.yml               # macOS CPU, Python 3.13
│       ├── test_gpu.yml                 # CUDA GPU tests, Python 3.13
│       └── release.yml                  # PyPI release on tag push v*
│                                        # * 3.14 job uses continue-on-error: true
│
├── docs/
│   ├── superpowers/specs/               # design documents (this file)
│   ├── user_guide/models/
│   │   ├── scviva.md
│   │   ├── destvi.md
│   │   └── resolvi.md
│   ├── references.md                    # model and package references
│   ├── references.bib                   # BibTeX citations
│   └── tutorials/
│       ├── scVIVA_tutorial.ipynb
│       ├── DestVI_tutorial.ipynb
│       └── resolVI_tutorial.ipynb
│
├── src/spatialvi/
│   ├── __init__.py                      # lazy exports: SCVIVA, DestVI, ResolVI
│   ├── _constants.py                    # package-wide REGISTRY_KEYS and constants
│   ├── _settings.py                     # spatialvi settings object
│   │
│   ├── model/
│   │   ├── __init__.py                  # exports SCVIVA, DestVI, ResolVI
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   ├── _spatial_base.py         # SpatialBaseModel
│   │   │   ├── _neighborhood_mixin.py   # SpatialNeighborhoodMixin
│   │   │   └── _deconvolution_mixin.py  # SpatialDeconvolutionMixin
│   │   ├── _scviva.py                   # SCVIVA model class
│   │   ├── _destvi.py                   # DestVI model class
│   │   └── _resolvi.py                  # ResolVI model class
│   │
│   ├── module/
│   │   ├── __init__.py
│   │   ├── _nichevae.py                 # nicheVAE (scVIVA PyTorch module)
│   │   ├── _mrdeconv.py                 # MRDeconv (DestVI PyTorch module)
│   │   └── _resolvae.py                 # RESOLVAE (ResolVI Pyro module)
│   │
│   ├── external/
│   │   └── __init__.py                  # empty; reserved for future models
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── _fields.py                   # SpatialCoordsField, NeighborhoodGraphField
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── _spatial.py                  # spatialdata/squidpy helpers
│   │
│   └── train/
│       ├── __init__.py
│       └── _config.py                   # spatial-specific training plan configs (if needed)
│                                        # otherwise re-exports scvi.train._config
│
├── tests/
│   ├── conftest.py                      # shared fixtures incl. minimal SpatialBaseModel subclass
│   ├── model/
│   │   ├── test_scviva.py
│   │   ├── test_destvi.py
│   │   └── test_resolvi.py
│   └── base/
│       └── test_spatial_base.py         # uses minimal concrete subclass fixture
│
├── .codecov.yaml
├── .editorconfig
├── .gitignore
├── .markdownlint.yaml
├── .pre-commit-config.yaml
├── .readthedocs.yaml
├── CHANGELOG.md
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── README.md
└── pyproject.toml
```

### `data/_fields.py` Notes (I4 fix)

`NeighborhoodGraphField` stores connectivity as a **dense numpy array** (matching how the upstream
ResolVI `_prepare_data` stores `index_neighbor` and `distance_neighbor` in `.obsm`).
Squidpy outputs a sparse CSR matrix — the field applies `.toarray()` on registration.
This is documented behavior, not silent conversion.

`SpatialCoordsField` wraps `ObsmField` with:
- Validation that coordinates are 2D or 3D
- Optional L2 normalization of coordinates before registration

### `train/_config.py` Notes (S2 fix)

For v1, this file re-exports from `scvi.train._config`. It exists as a placeholder for future
spatial-specific training plan configs (e.g., coordinate-regularized loss terms for scVIVA).
If no spatial-specific configs are needed, the file is removed before v1 release.

### Test Fixture for `SpatialBaseModel` (S5 fix)

`tests/conftest.py` defines a minimal concrete subclass for testing `SpatialBaseModel` in
isolation:

```python
class _MinimalSpatialModel(SpatialBaseModel, UnsupervisedTrainingMixin):
    """Minimal concrete model for testing SpatialBaseModel methods."""
    def __init__(self, adata):
        super().__init__(adata)
        self.module = nn.Linear(10, 5)  # dummy module

    @classmethod
    def setup_anndata(cls, adata, **kwargs):
        manager = AnnDataManager(fields=[LayerField(...), SpatialCoordsField(...)])
        cls.register_manager(manager)
```

---

## Data Flow

### Path 1: Standard AnnData (backward compatible)

```python
import scanpy as sc
from spatialvi.model import SCVIVA

adata = sc.read_h5ad("visium.h5ad")
SCVIVA.setup_anndata(adata, layer="counts", spatial_key="spatial")
model = SCVIVA(adata)
model.train()
latent = model.get_latent_representation()                     # CPU
latent = model.get_latent_representation(backend="rapids")     # GPU
```

### Path 2: SpatialData (scverse-native)

```python
import spatialdata
from spatialvi.model import SCVIVA

sdata = spatialdata.read_zarr("xenium.zarr")
# Two-step (explicit, mirrors setup_anndata convention):
SCVIVA.setup_spatialdata(sdata, table_key="table", region="cells")
model = SCVIVA(sdata["table"].to_adata())

# One-step convenience constructor:
model = SCVIVA.from_spatialdata(sdata, table_key="table", region="cells")
```

### Path 3: Neighbor computation

```python
# squidpy backend (default)
model.compute_neighbors(adata, coord_type="generic", n_neighs=6, backend="squidpy")

# RAPIDS GPU backend
model.compute_neighbors(adata, backend="rapids")
```

---

## Dependency Stack

```toml
[project]
requires-python = ">=3.12"

[project.dependencies]
anndata = ">=0.11"
scanpy = ">=1.10"
scvi-tools = "*"           # base classes, mixins, AnnDataManager
torch = "*"
lightning = ">=2"
pyro-ppl = "*"             # ResolVI
numpy = ">=1.21"
pandas = "*"
scipy = "*"
rich = "*"
tqdm = "*"

[project.optional-dependencies]
spatial = ["spatialdata>=0.2", "squidpy>=1.4"]
rapids  = ["cuml>=24.0", "cugraph>=24.0", "cupy-cuda12x"]
dev     = ["pre-commit", "twine>=4.0.2"]
test    = ["pytest", "pytest-cov", "anndata", "scanpy"]
doc     = ["sphinx", "myst-nb", "sphinx-book-theme", "ipykernel"]
all     = ["spatialvi-tools[spatial,rapids,dev,doc,test]"]
```

---

## CI/CD

| Workflow | Trigger | Runner | Python |
|---|---|---|---|
| `build.yml` | PR + push main | ubuntu-latest | 3.13 |
| `test_linux.yml` | PR + push main | ubuntu-latest | 3.12, 3.13, 3.14* |
| `test_macos.yml` | PR + push main | macos-latest | 3.13 |
| `test_gpu.yml` | PR + push main | self-hosted CUDA | 3.13 |
| `release.yml` | tag push `v*` | ubuntu-latest | 3.13 |

\* Python 3.14 CI job uses `continue-on-error: true` — aspirational target; wheel availability
from PyTorch/scvi-tools is not guaranteed at v1 release time.

**Tooling:**
- Build: `hatchling`
- Lint/format: `ruff`
- Pre-commit: `ruff`, `codespell`, trailing whitespace, end-of-file fixer
- Docs: `sphinx` + `myst-nb` + ReadTheDocs
- Tests: `pytest` + `pytest-cov`

---

## Shared Code Consolidation (vs. scvi-tools originals)

| Duplicated concern | Where it was | Consolidated into |
|---|---|---|
| Spatial coord registration | Each model's `setup_anndata` | `SpatialBaseModel._register_spatial_coords()` |
| Neighbor graph computation | scVIVA (squidpy), ResolVI (`_prepare_data`) | `SpatialNeighborhoodMixin.compute_neighbors(backend=)` |
| Neighbor graph registration | scVIVA + ResolVI `setup_anndata` | `SpatialNeighborhoodMixin._setup_neighbor_field()` |
| Spatial plotting | Scattered / absent in originals | `SpatialBaseModel.plot_spatial_embedding/predictions()` |
| `get_latent_representation` | Each model separately | `SpatialBaseModel.get_latent_representation(backend=)` |
| Deconvolution result formatting | DestVI only | `SpatialDeconvolutionMixin.get_proportions_df()` |

---

## Future External Models (not in v1)

The `spatialvi.external` namespace is reserved for models to be ported in subsequent iterations:
amici, harreman, lambda_model, nolan, ppi, sparl, starfysh, vivs.

These are tracked in `/Users/orikr/PycharmProjects/spatialvi-tools`.

---

## Out of Scope for v1

- JAX/MLX backends
- Hub model integration (scvi-hub)
- Multi-GPU training
- Minified model support (beyond what scvi provides)
- Any model from `spatialvi.external`
- `CondSCVI` wrapper (imported directly from `scvi.model` by users)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         spatialvi-tools                                 │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      scvi-tools (upstream)                        │  │
│  │  BaseModelClass · UnsupervisedTrainingMixin · VAEMixin            │  │
│  │  RNASeqMixin · EmbeddingMixin · ArchesMixin                      │  │
│  │  BaseMinifiedModeModelClass · PyroSviTrainMixin · PyroSampleMixin │  │
│  └───────────────────────────┬──────────────────────────────────────┘  │
│                               │ inherits                                │
│  ┌────────────────────────────▼───────────────────────────────────┐    │
│  │                    SpatialBaseModel                             │    │
│  │  setup_spatialdata()  ·  from_spatialdata()                    │    │
│  │  get_latent_representation(backend="cpu"|"rapids")             │    │
│  │  plot_spatial_embedding()  ·  plot_spatial_predictions()       │    │
│  └──────────────┬─────────────────────────┬────────────────────┘      │
│                 │ composed                 │ composed                   │
│  ┌──────────────▼──────────┐  ┌───────────▼────────────┐              │
│  │  SpatialNeighborhoodMixin│  │ SpatialDeconvolutionMixin│            │
│  │  compute_neighbors(      │  │ get_proportions_df()    │             │
│  │    backend="squidpy"     │  │ plot_cell_type_map()    │             │
│  │           |"rapids")     │  └───────────┬────────────┘              │
│  └──────┬────────┬──────────┘             │                            │
│         │        │                        │                            │
│  ┌──────▼──┐  ┌──▼──────┐  ┌─────────────▼──┐                        │
│  │ SCVIVA  │  │ ResolVI │  │     DestVI      │                        │
│  │         │  │         │  │                 │                        │
│  │ nicheVAE│  │RESOLVAE │  │  MRDeconv       │                        │
│  │ (module)│  │ (module)│  │  (module)       │                        │
│  └─────────┘  └─────────┘  └─────────────────┘                        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Custom AnnData Fields                                           │   │
│  │  SpatialCoordsField (2D/3D validation)                          │   │
│  │  NeighborhoodGraphField (CSR → dense conversion)                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Key:
  SCVIVA  = SpatialNeighborhoodMixin + scvi mixins + SpatialBaseModel + BaseMinifiedModeModelClass
  ResolVI = SpatialNeighborhoodMixin + SpatialBaseModel + PyroSviTrainMixin + PyroSampleMixin
            + ResolVIPredictiveMixin (vendored) + ArchesMixin + BaseModelClass
  DestVI  = SpatialDeconvolutionMixin + UnsupervisedTrainingMixin + SpatialBaseModel
```

---

## Post-Implementation Changes (2026-03-25)

Changes applied after initial implementation, tracked here for spec completeness:

| # | Change | Rationale |
|---|--------|-----------|
| P1 | `compute_composition_error` / `compute_niche_error` moved from `module/_nichevae_log_likelihood.py` into `module/_nichevae.py` | Eliminated standalone single-purpose file; functions logically belong with the module they operate on |
| P2 | Regression tests rewritten to compare spatialvi vs scvi outputs directly | Original tests only ran spatialvi in isolation; true regression requires side-by-side comparison with same seed |
| P3 | `tests/regression/test_destvi_upstream.py` added | DestVI regression was missing entirely |
| P4 | All empty `tests/*/__init__.py` files removed | Not needed for pytest; caused unnecessary noise |
| P5 | `@pytest.mark.optional` removed from all tests | All tests should run by default |
| P6 | `pandas>=2.0`, `numpy>=2.0` minimum versions set | Modern-only package; no legacy support needed |
| P7 | `.pre-commit-config.yaml` updated to match scvi-tools (added blacken-docs, prettier, mdformat, markdownlint-fix, forbid-rej hooks) | Consistency with upstream tooling |
| P8 | `docs/conf.py` expanded with full Sphinx extensions matching scvi-tools | Required for proper ReadTheDocs builds |
| P9 | `.readthedocs.yaml` updated to ubuntu-24.04 | Match scvi-tools; ubuntu-22.04 EOL approaching |
| P10 | `.codecov.yaml` expanded with per-flag tracking | Match scvi-tools coverage reporting |
| P11 | `test_gpu.yml` changed to label-triggered (`cuda tests` / `all tests`) | Match scvi-tools pattern; prevents wasted self-hosted runner time |
| P12 | `docs/tutorials/` restored with scVIVA, DestVI, resolVI notebooks | Were accidentally absent from feature branch |
| P13 | `plot_cell_type_map` (all-types path): writes proportions to `adata.obs` before plotting | Fix: was passing column names not in obs to scanpy |
| P14 | DestVI `get_latent_representation`: preserve tuple when `return_dist=True` on RAPIDS | Fix: `cp.asarray(tuple)` broke tuple unpacking |
| P15 | ResolVI `differential_expression`: raise `ValueError` when `library_scaling=True` and `weights!="importance"` | Fix: parameter was silently ignored |
=======
## Implementation Plan Summary

The full plan is at `docs/superpowers/plans/2026-03-25-spatialvi-implementation.md`.
All 21 tasks were completed on 2026-03-25. Final state: **50/50 tests pass**, package builds successfully.

| # | Task | Description |
|---|------|-------------|
| 1 | Package Scaffolding | `pyproject.toml`, `README.md`, `CHANGELOG.md`, `LICENSE`, `.gitignore`, `.pre-commit-config.yaml`, `.readthedocs.yaml`, `.editorconfig`, `.markdownlint.yaml`, `.codecov.yaml`, `Dockerfile`, `CONTRIBUTING.md` |
| 2 | GitHub Workflows & Issue Templates | `build.yml`, `test_linux.yml` (3.12/3.13/3.14*), `test_macos.yml`, `test_gpu.yml`, `release.yml`; ISSUE_TEMPLATE, PR template |
| 3 | Package Skeleton | `src/spatialvi/__init__.py` (lazy imports), `_constants.py`, `_settings.py`, `model/__init__.py`, `external/__init__.py`, `train/_config.py` |
| 4 | Custom AnnData Fields | `src/spatialvi/data/_fields.py`: `SpatialCoordsField` (2D/3D validation), `NeighborhoodGraphField` (CSR→dense conversion) |
| 5 | `SpatialBaseModel` | `src/spatialvi/model/base/_spatial_base.py`: `setup_spatialdata`, `from_spatialdata`, `get_latent_representation(backend=)`, `plot_spatial_embedding`, `plot_spatial_predictions` |
| 6 | `SpatialNeighborhoodMixin` | `src/spatialvi/model/base/_neighborhood_mixin.py`: `compute_neighbors(backend="squidpy"\|"rapids")`, `_setup_neighbor_field` |
| 7 | `SpatialDeconvolutionMixin` | `src/spatialvi/model/base/_deconvolution_mixin.py`: `get_proportions_df()`, `plot_cell_type_map()` |
| 8 | `utils/_spatial.py` | Shared spatial helper utilities |
| 9 | Base Infrastructure Smoke Test | Full import + fixture validation; `tests/conftest.py` with `_MinimalSpatialModel`, `make_spatial_adata()` |
| 10 | scVIVA Module | `src/spatialvi/module/_nichevae.py` (nicheVAE), `_nichevae_components.py` (Encoder, DirichletDecoder, NicheDecoder), `_nichevae_log_likelihood.py` |
| 11 | scVIVA Model + DE | `src/spatialvi/model/_scviva.py` (SCVIVA class, MRO with explicit `get_latent_representation` override), `model/_scviva_de/` (niche DE sub-package) |
| 12 | scVIVA Tests | `tests/model/test_scviva.py`: setup_anndata, train, get_latent_representation (CPU + RAPIDS optional), compute_neighbors |
| 13 | DestVI Module | `src/spatialvi/module/_mrdeconv.py` (MRDeconv, ported from scvi-tools) |
| 14 | DestVI Model | `src/spatialvi/model/_destvi.py` (DestVI class with `SpatialDeconvolutionMixin`, `from_rna_model` inherited) |
| 15 | DestVI Tests | `tests/model/test_destvi.py`: from_rna_model, get_proportions, get_proportions_df, get_latent_representation |
| 16 | ResolVI Module | `src/spatialvi/module/_resolvae.py` (RESOLVAE Pyro module, ported from scvi-tools) |
| 17 | ResolVI Model | `src/spatialvi/model/_resolvi.py` (ResolVI class: SpatialNeighborhoodMixin + PyroSviTrainMixin + ResolVIPredictiveMixin); vendored `_resolvi_predictive.py` |
| 18 | ResolVI Tests | `tests/model/test_resolvi.py`: setup_anndata, compute_neighbors, train, get_latent_representation |
| 19 | Documentation | `docs/conf.py` (Sphinx + myst_nb), `docs/user_guide/models/scviva.md`, `destvi.md`, `resolvi.md` |
| 20 | Final Integration & Smoke Test | Full test suite (25/25 pass), package build (`dist/*.whl`, `dist/*.tar.gz`), lazy import validation |
| 21 | Code Consolidation & Regression | `tests/regression/test_scviva_upstream.py`, `test_resolvi_upstream.py`; 50/50 tests pass total |

\* Python 3.14 CI job uses `continue-on-error: true` — aspirational target.
