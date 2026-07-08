# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2026-07-08

### Fixed

- **Harreman**: pandas 3.0 always-on copy-on-write broke three in-place mutation sites
  that relied on `.values` returning a writable view:
  - `create_modules` raised `ValueError: underlying array is read-only` on
    `np.fill_diagonal`; now forces a real copy via `to_numpy(copy=True)`
  - `run_cell_communication_analysis` hit the same `fill_diagonal` failure on the
    symmetric LC Z-score matrix; now mutates a copy and writes it back via `.loc[:, :]`
  - `extract_lr_pairs` raised `ValueError: Length of indexer and values mismatch`
    because pandas 3.0's default Arrow-backed string dtype no longer allows storing
    variable-length arrays into single-element slots; ligand/receptor columns now use
    `to_numpy(dtype=object, copy=True)` to restore plain, guaranteed-writable
    object-array semantics

## [0.1.5] - 2026-07-06

### Added

- **Tangram** model for mapping scRNA-seq to spatial transcriptomics
  (`scviva.external.Tangram`), a PyTorch reimplementation supporting "cells" and
  "constrained" cell-to-spot mapping modes
- **Harreman** downstream tool for cell-cell communication and metabolic exchange
  analysis (`scviva.tl.harreman`, `scviva.pl.harreman`), integrating outputs from
  DestVI, ResolVI, and SCVIVA
- **`SpatialPredictiveMixin`**: unified predictive mixin shared by ResolVI and SCVIVA,
  replacing the ResolVI-only `ResolVIPredictiveMixin`
- **`scviva.tl`/`scviva.pl`** top-level namespace aliases, mirroring scanpy's
  `sc.tl`/`sc.pl` convention
- Dedicated CI job for optional/slow integration tests (real model training)

### Changed

- **Package renamed** from `spatialvi-tools` to `scviva-tools` on PyPI (import name
  `scviva` unchanged); docs and install instructions updated accordingly

### Fixed

- **Tangram**: gene-mismatch validation in `setup_mudata`, and `get_mapper_matrix`
  filtering under `constrained=True`
- **Harreman**: import/namespace bugs from the initial port that made the package
  unimportable, plus a broken plotting accessor and a model-recognition constant typo

## [0.1.3] - 2026-04-13

### Changed

- **Cleanup**: dropped dead code (MLflow/JAX settings, a stale config re-export, a
  trivial method alias, stale in-source dev comments) and simplified the lazy model map
- **Refactored** `SCVIVA`'s predictive methods to share a common decoder helper, and
  tidied up several module-level imports; moved a few internal modules under `utils/`
  for consistency with scvi-tools conventions

### Fixed

- `scikit-learn>=1.4` moved from test-only to core dependencies (required by SCVIVA/ResolVI
  at runtime); removed redundant `anndata`/`scanpy` test-extra pins

### Documentation

- Added installation guide, FAQ, API reference stubs, developer/contributing guides, and
  user-guide background/use-case pages

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
