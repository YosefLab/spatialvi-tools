# Spatial Predictive Mixin Generalization & DE Coverage Design

**Date:** 2026-06-07
**Status:** Approved for implementation
**Authors:** Ori Kronfeld + Claude

---

## Goal

Two related improvements:

1. **Generalize `ResolVIPredictiveMixin`** — rename it to `SpatialPredictiveMixin`, apply it to
   SCVIVA as well as ResolVI, and add a PyTorch-native default for
   `get_normalized_expression_importance` so non-Pyro models share the same public surface.

2. **Close DE coverage gaps** — `differential_expression` is currently only in SCVIVA and
   ResolVI. Add it to GIMVI and RNAStereoscope (the only other models where DE is semantically
   meaningful). Add missing tests for every function across every model.

---

## Scope

### In scope

- Rename `ResolVIPredictiveMixin` → `SpatialPredictiveMixin`; keep backward-compat alias.
- Add `SpatialPredictiveMixin` to SCVIVA MRO.
- Add PyTorch-default `get_normalized_expression_importance` to `SpatialPredictiveMixin`.
- Move Pyro-specific overrides (`get_latent_representation`, `get_normalized_expression`,
  `get_normalized_expression_importance`) from the mixin into ResolVI's class body.
- Add `get_neighbor_abundance` to SCVIVA via `SpatialPredictiveMixin`.
- Add `RNASeqMixin` to GIMVI and `RNAStereoscope` to unlock `differential_expression`.
- Add missing tests: `test_spatial_predictive.py` (mixin unit), SCVIVA DE + predictive,
  GIMVI DE, RNAStereoscope DE, differential_abundance and normalized_expression gaps.

### Out of scope

- DestVI, SpatialStereoscope, Tangram: no DE addition (deconvolution/mapping semantics).
- `get_neighbor_abundance` for DestVI, GIMVI, Stereoscope, Tangram.
- Any new training logic or module changes.

---

## Design

### 1. Mixin Restructure

#### New file: `src/scviva/model/base/_spatial_predictive.py`

```
class SpatialPredictiveMixin:
    get_normalized_expression_importance()  # PyTorch default (see below)
    get_neighbor_abundance()                # moved unchanged from old mixin
```

**`get_normalized_expression_importance` (PyTorch default):**
Uses `self.get_importance_weights()` (from `RNASeqMixin`) combined with
`self.get_normalized_expression()` to produce importance-weighted normalized expression.
Signature matches the current ResolVI version for API consistency.

**`get_latent_representation` — removed from mixin.**
Already lives in `SpatialBaseModel`. The Pyro-specific override moves to ResolVI's class body.

**`get_normalized_expression` — removed from mixin.**
Already in `BaseModelClass`. The Pyro-specific override moves to ResolVI's class body.

#### Backward-compat shim: `_resolvi_predictive.py` (keep file, replace contents)

```python
from ._spatial_predictive import SpatialPredictiveMixin as ResolVIPredictiveMixin
__all__ = ["ResolVIPredictiveMixin"]
```

#### `src/scviva/model/base/__init__.py`

Export both `SpatialPredictiveMixin` (new) and `ResolVIPredictiveMixin` (alias).

---

### 2. Model Changes

#### SCVIVA (`src/scviva/model/_scviva.py`)

Add `SpatialPredictiveMixin` to MRO:

```python
class SCVIVA(
    SpatialPredictiveMixin,   # new — adds get_normalized_expression_importance, get_neighbor_abundance
    SpatialNeighborhoodMixin,
    EmbeddingMixin,
    RNASeqMixin,
    VAEMixin,
    ...
):
```

No new methods in the class body — everything comes from the mixin.

#### ResolVI (`src/scviva/model/_resolvi.py`)

Replace `ResolVIPredictiveMixin` with `SpatialPredictiveMixin` in MRO. Add Pyro-specific
overrides to the class body:

```python
class ResolVI(
    SpatialPredictiveMixin,   # replaces ResolVIPredictiveMixin
    SpatialNeighborhoodMixin,
    SpatialBaseModel,
    PyroSviTrainMixin,
    PyroSampleMixin,
    ArchesMixin,
    BaseModelClass,
):
    # Pyro-specific overrides (moved from old mixin):
    def get_latent_representation(self, ...): ...       # uses pyro.infer.Predictive
    def get_normalized_expression(self, ...): ...       # uses pyro.infer.Predictive
    def get_normalized_expression_importance(self, ...): ...  # Pyro path
```

The Pyro overrides are moved verbatim — no logic changes.

#### GIMVI (`src/scviva/model/_gimvi.py`)

Add `RNASeqMixin` to MRO:

```python
from scvi.model.base import RNASeqMixin
class GIMVI(RNASeqMixin, SpatialBaseModel): ...
```

`differential_expression` becomes available via `RNASeqMixin`. No new methods needed.

> **Caveat:** `RNASeqMixin` exposes additional methods beyond `differential_expression`
> (e.g. `get_importance_weights`, `posterior_predictive_sample`, `get_feature_correlation_matrix`).
> GIMVI has a dual-dataset interface — these extra methods may not work correctly.
> Only `differential_expression` is confirmed in scope. If MRO verification reveals conflicts,
> the fallback is to add a standalone `differential_expression` method to GIMVI's class body
> that delegates to `scvi.model._utils._de_core` directly, matching what `RNASeqMixin` calls.

#### RNAStereoscope (`src/scviva/external/stereoscope/_model.py`)

Add `RNASeqMixin`:

```python
from scvi.model.base import RNASeqMixin
class RNAStereoscope(RNASeqMixin, UnsupervisedTrainingMixin, SpatialBaseModel): ...
```

---

### 3. Tests

#### New: `tests/base/test_spatial_predictive.py`

Mixin-level unit tests using a minimal trained model (avoids full per-model training overhead):

| Test | What it checks |
|------|---------------|
| `test_get_neighbor_abundance_shape` | output shape `(n_obs, n_cell_types)`, no NaNs |
| `test_get_normalized_expression_importance_pytorch` | shape matches `get_normalized_expression`, no NaNs |

#### Additions to `tests/model/test_scviva.py`

| Test | What it checks |
|------|---------------|
| `test_scviva_normalized_expression` | shape `(n_obs, n_genes)`, no NaNs |
| `test_scviva_normalized_expression_importance` | shape `(n_obs, n_genes)`, no NaNs |
| `test_scviva_neighbor_abundance` | shape `(n_obs, n_labels)`, sums to 1 per row |
| `test_scviva_differential_expression` | returns DataFrame; ported from upstream `test_scviva_differential` |
| `test_scviva_differential_abundance` | returns DataFrame |

#### Additions to `tests/model/test_gimvi.py`

| Test | What it checks |
|------|---------------|
| `test_gimvi_differential_expression` | runs without error, returns DataFrame |

#### Additions to `tests/external/test_stereoscope.py`

| Test | What it checks |
|------|---------------|
| `test_rna_stereoscope_differential_expression` | runs without error, returns DataFrame |

#### Additions to existing test files (gap closure)

| File | Test | Gap being closed |
|------|------|-----------------|
| `test_destvi.py` | `test_destvi_normalized_expression` | no NE test exists |
| `test_destvi.py` | `test_destvi_differential_abundance` | no DA test exists |
| `test_gimvi.py` | `test_gimvi_normalized_expression` | no NE test exists |
| `test_gimvi.py` | `test_gimvi_differential_abundance` | no DA test exists |
| `test_resolvi.py` | `test_resolvi_neighbor_abundance` | `get_neighbor_abundance` has no unit test (only regression) |
| `test_stereoscope.py` | `test_spatial_stereoscope_normalized_expression` | no NE test exists |
| `test_stereoscope.py` | `test_spatial_stereoscope_differential_abundance` | no DA test exists |

---

## File Change Summary

| File | Change |
|------|--------|
| `src/scviva/model/base/_spatial_predictive.py` | **new** — `SpatialPredictiveMixin` |
| `src/scviva/model/base/_resolvi_predictive.py` | **replace** with alias shim |
| `src/scviva/model/base/__init__.py` | export `SpatialPredictiveMixin` + alias |
| `src/scviva/model/_scviva.py` | add `SpatialPredictiveMixin` to MRO |
| `src/scviva/model/_resolvi.py` | swap to `SpatialPredictiveMixin`; add Pyro overrides to class body |
| `src/scviva/model/_gimvi.py` | add `RNASeqMixin` to MRO |
| `src/scviva/external/stereoscope/_model.py` | add `RNASeqMixin` to `RNAStereoscope` |
| `tests/base/test_spatial_predictive.py` | **new** |
| `tests/model/test_scviva.py` | 5 new tests |
| `tests/model/test_gimvi.py` | 3 new tests |
| `tests/external/test_stereoscope.py` | 4 new tests |
| `tests/model/test_destvi.py` | 2 new tests |
| `tests/model/test_resolvi.py` | 1 new test |

---

## MRO Verification

After changes, verify no `TypeError` on class definition:

```python
print(SCVIVA.__mro__)
print(ResolVI.__mro__)
print(GIMVI.__mro__)
print(RNAStereoscope.__mro__)
```

`SpatialPredictiveMixin` has no base class (pure mixin), so no diamond conflicts are introduced.
`RNASeqMixin` from scvi-tools is already used in SCVIVA without conflict — safe to add to GIMVI
and RNAStereoscope.

---

## What Does NOT Change

- DestVI, SpatialStereoscope, Tangram: zero model-level changes.
- All existing test names and assertions: no regressions.
- `differential_expression` implementations in SCVIVA and ResolVI: unchanged.
- The `_scviva_de/` sub-package: unchanged.
- Pyro internals in ResolVI: moved verbatim, no logic changes.

---

## Test Count Impact

| Date | Tests |
|------|-------|
| 2026-04-12 (GIMVI + Stereoscope) | 76/76 |
| After this change | ~91/91 (approx. +15 new tests) |
