# spatialvi-tools: Planning Decisions & Alternatives

**Date:** 2026-03-24
**Authors:** Ori Kronfeld + Claude
**Purpose:** Record of all design questions raised during planning, alternatives considered,
and the rationale for each final decision. Intended for team review and future reference.

---

## Q1: Dependency strategy — how tightly coupled to scvi-tools?

### Alternatives considered

| Option | Description | Pro | Con |
|--------|-------------|-----|-----|
| **A (chosen)** | Thin wrapper — import base classes from scvi | No reinvention; free upstream updates; ecosystem alignment | Tied to scvi internal API (e.g., `_de_core`, `AnnDataManager`) |
| B | Partially independent — copy and own only the needed base code | Fewer moving-target dependencies | Maintenance burden; diverges from scvi conventions |
| C | Fully independent — build the whole stack from primitives (torch, lightning, pyro) | Maximum control | Enormous scope; loses all scvi infrastructure |

### Decision: A
scvi-tools is a first-class scverse citizen. Wrapping it means we inherit AnnData conventions,
the training loop, data registry, minification, hub integration, and all future improvements
for free. The cost (coupling to scvi internals) is accepted.

---

## Q2: Where do the 3 models live in the package?

### Alternatives considered

| Option | Description |
|--------|-------------|
| A | Flat model layout — all 3 in `spatialvi.model`, mirroring scvi's core model namespace |
| B | Sub-package per model — `model/scviva/`, `model/destvi/`, `model/resolvi/` |
| C | Domain-grouped — `model/deconvolution/`, `model/denoising/`, `model/niche/` |

### Initial preference: A (flat)
The team initially agreed on flat layout with a shared `model/base/`.

### Refinement during planning
The user clarified: the structure should mirror scvi-tools exactly, with a `model/` folder for
core models and an `external/` folder reserved for future models. However, **all 3 current
models (scVIVA, ResolVI, DestVI) are core models**, not external — even though scVIVA and
ResolVI are in `scvi.external`. In spatialvi-tools, these are first-class citizens equivalent
to SCVI/TOTALVI in scvi-tools.

### Decision: Flat core models + empty external/
- `spatialvi.model.SCVIVA`, `spatialvi.model.DestVI`, `spatialvi.model.ResolVI`
- `spatialvi.external` exists but is empty in v1
- Future candidates for external: amici, harreman, sparl, starfysh, vivs, nolan, ppi

---

## Q3: How to structure shared code?

### Alternatives considered

| Option | Description |
|--------|-------------|
| A | Single shared mixin layer — all shared logic as composable mixins |
| B | `SpatialBaseModel` intermediate class — one base class all 3 inherit from |
| C | Protocol + composition — Python Protocols, no inheritance |

### Decision: Hybrid A+B
- **`SpatialBaseModel`** (approach B) for logic every spatial model needs:
  setup_spatialdata, spatial coord registration, latent representation with RAPIDS,
  spatial plotting.
- **Optional mixins** (approach A) for capabilities that only some models have:
  `SpatialNeighborhoodMixin` (scVIVA + ResolVI), `SpatialDeconvolutionMixin` (DestVI only).
- Approach C was rejected as un-idiomatic for scvi-style codebases.

---

## Q4: How to handle RAPIDS acceleration?

### Alternatives considered

| Option | Description |
|--------|-------------|
| A | `RAPIDSAccelerationMixin` — separate mixin any model can inherit |
| **B (chosen)** | `backend` parameter on relevant functions |
| C | Separate `rapids`-namespaced functions (e.g., `model.rapids.get_latent_representation()`) |

### Decision: B
RAPIDS as a `backend="rapids"` parameter on `compute_neighbors()` and
`get_latent_representation()`. Cleaner API, consistent with scikit-learn conventions, no
inheritance complexity. The function handles dispatch internally.

---

## Q5: Python version support?

### Alternatives considered
- 3.10+ (broadest compatibility)
- 3.11+ (previous spatialvi-tools2 attempt)
- 3.12–3.14 (modern Python)

### Decision: 3.12–3.14, default 3.13
- Minimum 3.12 (drops 3.11 — no loss for a new package)
- 3.13 is the primary/default version for CI and docs
- 3.14 is aspirational — CI job uses `continue-on-error: true` because PyTorch and
  scvi-tools wheel availability for 3.14 is not guaranteed at v1 release time

---

## Q6: CI strategy?

### Alternatives considered

| Option | Description |
|--------|-------------|
| A | Lean — Linux CPU + pre-commit only |
| B | Full scvi-style — Linux/macOS/Windows, GPU, multiple Pythons |
| **C+GPU (chosen)** | Linux CPU (multi-Python) + macOS CPU + GPU + pre-commit + ReadTheDocs |

### Decision: C + GPU
Spatial transcriptomics workflows are GPU-intensive by nature. GPU CI was added to the
"middle ground" option C. Windows runners were dropped (not in scvi-tools external CI either).

---

## Q7: What spatial ecosystem integrations are in scope for v1?

All four integration areas were confirmed as v1 scope:

| Integration | Entry point | Notes |
|-------------|-------------|-------|
| SpatialData | `setup_spatialdata()` + `from_spatialdata()` | Alternative constructor pattern |
| squidpy | `compute_neighbors(backend="squidpy")` | Default neighbor backend |
| RAPIDS | `backend="rapids"` parameter | On `compute_neighbors` and `get_latent_representation` |
| Spatial visualization | `plot_spatial_embedding()`, `plot_spatial_predictions()`, `plot_cell_type_map()` | In SpatialBaseModel + SpatialDeconvolutionMixin |

---

## Issues Raised by Spec Review & Resolutions

These issues were caught during the automated spec review and resolved before implementation:

| ID | Issue | Resolution |
|----|-------|------------|
| C1 | ResolVI's upstream `_prepare_data` computes neighbors — would conflict with `SpatialNeighborhoodMixin` | `spatialvi.ResolVI.setup_anndata` calls `super().setup_anndata(prepare_data=False)`; `SpatialNeighborhoodMixin` takes full ownership |
| C2 | `get_latent_representation` override mechanism was unspecified | Explicitly defined in `SpatialBaseModel`; model classes do NOT define their own version |
| I1 | `PyroSviTrainMixin`/`PyroSampleMixin` import path missing | Documented: both live in `scvi.model.base`, not `scvi.train` |
| I2 | `ResolVIPredictiveMixin` was silently dropped from ResolVI MRO | Retained; provides `get_neighbor_abundance`, `get_normalized_expression_importance` |
| I3 | `setup_spatialdata` vs `from_spatialdata` roles ambiguous | `setup_spatialdata` = field registration classmethod; `from_spatialdata` = convenience constructor classmethod |
| I4 | `NeighborhoodGraphField` sparse handling unverified | Stores dense numpy arrays (matches upstream); squidpy CSR output is converted via `.toarray()` on registration |
| I5 | Python 3.14 presented as guaranteed | Marked aspirational; CI job uses `continue-on-error: true` |
| S1 | Inheritance diagram misrepresented mixin direction | Fixed: diagram now shows composition, not inheritance from SpatialBaseModel |
| S2 | `train/_config.py` purpose underspecified | v1 re-exports scvi; placeholder for future spatial training configs |
| S3 | `CondSCVI` / `from_rna_model` workflow missing | Documented: `CondSCVI` imported directly from `scvi.model`; `from_rna_model` inherited unchanged |
| S4 | No lazy import strategy | Documented: `__getattr__`-based lazy loading in `__init__.py` |
| S5 | No test fixture for `SpatialBaseModel` in isolation | `conftest.py` defines `_MinimalSpatialModel` fixture |

---

## Open Questions (for team discussion)

1. **Naming**: Should the package be `spatialvi-tools` (PyPI) + `spatialvi` (import name)?
   Or `spatialvi` for both? Confirm before first PyPI release.

2. **CondSCVI**: Should `spatialvi` eventually wrap or re-export `CondSCVI` for a one-stop
   DestVI workflow? Or keep the cross-package dependency explicit?

3. **scVIVA niche differential expression**: The upstream scVIVA has a full
   `differential_expression/` sub-package (`_niche_de_core.py`, `_marker_classifier.py`,
   `_results_dataclass.py`). Is niche DE in scope for v1 or v2?

4. **SpatialData write-back**: After inference, should models write predictions back into the
   `SpatialData` object (e.g., `sdata["table"].obsm["scviva_latent"]`)? Useful for scverse
   pipelines but requires keeping a reference to `sdata`.

5. **RAPIDS version pinning**: RAPIDS has strict CUDA version requirements. Should `rapids`
   be a separate optional extra with a warning on install, or left to the user to manage?
