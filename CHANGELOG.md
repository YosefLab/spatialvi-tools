# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Tangram** model for mapping single-cell RNA-seq data to spatial transcriptomics
  (`scviva.external.Tangram`). Torch reimplementation of the Tangram algorithm
  supporting both "cells" and "constrained" modes for cell-to-spot mapping
- **`SpatialPredictiveMixin`** (`scviva.model.base`): generalizes the former
  ResolVI-only predictive mixin into a shared mixin providing `get_neighbor_abundance`
  and `_get_label_names` for any spatial model with a neighbor graph. ResolVI keeps
  its Pyro-specific posterior-sampling behavior via an override; SCVIVA now composes
  the same mixin, overriding `get_neighbor_abundance` to return its precomputed
  observed niche composition instead of a posterior-sampled quantity
- **Harreman** downstream tool for inferring metabolic exchanges and cell-cell
  communication in tissues using spatial transcriptomics (`scviva.tl.harreman`,
  plotting via `scviva.pl.harreman`). Provides a scanpy-style functional API
  (`harreman.tl`/`harreman.hs`/`harreman.vs`/`harreman.pp`/`harreman.ds`/`harreman.pl`)
  plus a stateful `HarremanAnalysis` wrapper class that can integrate outputs from
  DestVI, ResolVI, and SCVIVA. Ported from the open scvi-tools PR
  [scverse/scvi-tools#3806](https://github.com/scverse/scvi-tools/pull/3806)
- **`scviva.tl`/`scviva.pl`** top-level namespace aliases (`scviva/__init__.py`),
  mirroring scanpy's `sc.tl`/`sc.pl` convention, resolving lazily to
  `scviva.tools`/`scviva.plotting` (the underlying package names and folder layout are
  unchanged; the alias avoids eagerly importing heavy optional dependencies at plain
  `import scviva` time)
- `.github/workflows/test_linux_optional.yml`: dedicated CI job for `@pytest.mark.optional`
  integration tests (real model training), mirroring scvi-tools' `test_linux_optional.yml`.
  Runs on schedule, `workflow_dispatch`, or the `optional tests`/`all tests` PR labels via
  a new `--optional` pytest flag (`tests/conftest.py`); the default test job no longer runs
  these slow/heavy tests

### Changed

- **Package renamed** from `spatialvi-tools` to `scviva-tools` on PyPI; the importable
  module name (`import spatialvi`) is unchanged
- All documentation, GitHub workflow references, and install-hint strings updated to
  `scviva-tools` (e.g. `pip install "scviva-tools[spatial]"`)
- **Breaking**: `ResolVIPredictiveMixin` removed (`src/scviva/model/base/_resolvi_predictive.py`
  deleted). Use `SpatialPredictiveMixin` instead
- `ResolVI` now inherits `SpatialPredictiveMixin` in place of `ResolVIPredictiveMixin`
- `SpatialBaseModel` gained a shared `_maybe_rapids` helper for `backend="rapids"`
  dispatch, deduplicating the cupy-cast logic previously copy-pasted in both
  `ResolVI.get_latent_representation` and `SCVIVA.get_latent_representation`
- `docs/api/developer.md`: `ResolVIPredictiveMixin` entry replaced with
  `SpatialPredictiveMixin`
- **Harreman**: rewired all internal imports from the upstream `scvi.external.harreman`
  namespace to this repo's actual `scviva.tools.harreman` / `scviva.plotting.harreman`
  layout (the initial port never updated them, so the package was unimportable);
  populated `src/scviva/tools/__init__.py` and `src/scviva/plotting/__init__.py`/
  `src/scviva/plotting/harreman/__init__.py` to expose `HarremanAnalysis` and the
  plotting functions; added `numba`, `pooch`, `seaborn`, `statsmodels` to the `spatial`
  extra (imported unconditionally by Harreman but previously undeclared)

### Fixed

- **Tangram**: Fixed tuple bug in `setup_mudata` that prevented var_names validation
  from raising when sc and spatial modalities have mismatched genes
- **Tangram**: Fixed `get_mapper_matrix` to correctly apply the learned filter when
  `constrained=True`, ensuring downstream projection methods respect the target_count
  constraint
- **Harreman**: fixed `MODEL_RESOLVI` constant (`"RESOLVI"` → `"ResolVI"`) so
  `HarremanAnalysis` correctly recognizes this repo's `scviva.model.ResolVI` during
  model integration
- **Harreman**: fixed the `pl` (plotting) accessor, which previously imported from a
  nonexistent `plots` submodule under `scviva.tools.harreman` instead of the actual
  `scviva.plotting.harreman` package

### Tests

- All `model.train()` calls across the test suite reduced from `max_epochs=2` to
  `max_epochs=1` to cut unnecessary wall-clock time
- Removed spurious `accelerator="cpu"` from every test that does not explicitly verify
  CPU inference; kept only in `test_*_get_latent_*_cpu` tests that exercise the
  `backend="cpu"` inference path
- Added `tests/base/test_spatial_predictive.py` covering the shared
  `SpatialPredictiveMixin`; extended ResolVI, DestVI, and SCVIVA test modules for the
  predictive-mixin refactor
- Fixed the 4 Harreman test files (`tests/tools/harreman/*`, `tests/plotting/harreman/*`)
  to import from `scviva.tools.harreman`/`scviva.plotting.harreman`/`scviva.model`
  instead of the nonexistent `scvi.external.harreman`/`scvi.external`, which previously
  caused collection errors; added the missing `adata_spatial` fixture to
  `tests/plotting/harreman/test_harreman_analysis.py`; removed the stray
  `tests/tools/harreman/__init__.py` (no other test subpackage in this repo uses one)

### Notes

- Tangram regression tests against scvi-tools upstream are not included because the
  upstream implementation uses JAX/Flax while scviva-tools uses PyTorch, making
  direct comparison tests impractical

## [0.1.3] - 2026-04-13

### Changed

- **Dead code removal**: dropped MLflow settings (`mlflow_set_tracking_uri`, `mlflow_set_experiment`)
  and all JAX/XLA environment-variable logic from `SpatialviConfig`; removed the now-unused
  `import os` from `_settings.py`
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
- `src/spatialvi/utils/_gimvi_utils.py` moved to `src/spatialvi/model/utils/_gimvi_utils.py`
- `src/spatialvi/train/_gimvi_task.py` renamed to `src/spatialvi/train/_gimvi_trainingplans.py`
  to match scvi-tools naming convention

### Tests

- Added 5 unit tests for `get_niche_indexes` (`tests/model/test_scviva_utils.py`)

### Documentation

- Added `docs/installation.md` and `docs/faq.md`
- Added `docs/api/` with user and developer API reference stubs
- Added `docs/developer/` with contributing-code and maintenance guides
- Added `docs/user_guide/background/` (variational inference, DE, spatial methods)
- Added `docs/user_guide/use_cases/` (saving/loading, training configuration, downstream analysis)
- Updated `docs/index.md` and `docs/user_guide/index.md` toctrees

## [0.1.2] - 2026-04-12

### Fixed

- `RNADeconv.__init__`: read `model_kwargs['ct_weight']` instead of the incorrect
  `model_kwargs['ct_prop']`; the mismatched key caused a `KeyError` whenever
  `ct_weight` was passed, making class-reweighted training unusable
- `JVAE.generative`: replace `y.type.squeeze(-1)` with `y.squeeze(-1).long()` for the
  gene-label dispersion branch; `y.type` is Python's built-in `type()` bound on the
  tensor object, so any run with `dispersion='gene-label'` crashed before reaching
  the `one_hot` encoding
- Added regression tests for both fixes (all 78 tests pass)

## [0.1.1] - 2026-04-12

### Added

- **GIMVI** model for joint imputation of missing genes across paired scRNA-seq and
  spatial transcriptomics data, based on :cite:p:`Lopez19`
- **Stereoscope** model for probabilistic cell-type deconvolution of spatial
  transcriptomics spots (`spatialvi.external.Stereoscope`)
- Tutorials: `gimvi_tutorial`, `stereoscope_heart_LV_tutorial`,
  `cell2location_lymph_node_spatial_tutorial`, and `tangram_scvi_tools`
- Regression test suites for GIMVI and Stereoscope (upstream parity tests)

## [0.1.0] - 2026-03-25

### Added

- Initial package with DestVI, ResolVI, and scVIVA models
- `SpatialBaseModel` shared base infrastructure
- `SpatialNeighborhoodMixin` for neighbor graph computation (squidpy + RAPIDS backends)
- `SpatialDeconvolutionMixin` for cell-type deconvolution
- SpatialData integration via `setup_spatialdata()` and `from_spatialdata()`
- squidpy neighbor computation backend
- RAPIDS acceleration backend (`backend="rapids"`)
- Custom AnnData fields: `SpatialCoordsField`, `NeighborhoodGraphField`
