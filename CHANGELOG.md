# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `scviva.dataloaders.GraphDataLoader` and `GraphDataSplitter` for graph-aware
  spatial mini-batches backed by scvi-tools `AnnDataLoader`/`DataSplitter`

### Changed

- **Package renamed** from `spatialvi-tools` to `scviva-tools` on PyPI; the importable
  module name (`import spatialvi`) is unchanged
- All documentation, GitHub workflow references, and install-hint strings updated to
  `scviva-tools` (e.g. `pip install "scviva-tools[spatial]"`)
- `ResolVI` now defaults to `GraphDataSplitter`; graph training can use a guarded
  model-side neighbor expression cache and omit per-batch `x_n` transfers when enabled
- `SCVIVA` now defaults to `GraphDataSplitter` for training, using registered
  `niche_indexes`/`niche_distances` graph fields while omitting raw neighbor expression prefetch
- Updated graph dataloader design, implementation plan, and AI handoff docs from
  upstream `scvi-tools` paths to this `scviva-tools` project layout

### Fixed

- `GraphDataSplitter` no longer leaks semisupervised kwargs (e.g. `n_samples_per_label`)
  into `AnnDataLoader`, preventing `TypeError` in semisupervised ResolVI workflows;
  label-resampling is now applied at index level before constructing `GraphDataLoader`

### Tests

- All `model.train()` calls across the test suite reduced from `max_epochs=2` to
  `max_epochs=1` to cut unnecessary wall-clock time
- Removed spurious `accelerator="cpu"` from every test that does not explicitly verify
  CPU inference; kept only in `test_*_get_latent_*_cpu` tests that exercise the
  `backend="cpu"` inference path
- Added graph dataloader unit tests and ResolVI graph integration tests under
  `tests/dataloaders/` and `tests/model/`
- Added SCVIVA graph dataloader integration tests covering the default splitter and niche graph
  field forwarding
- Expanded SCVIVA tests to cover save/load round-trip, downstream APIs
  (`get_elbo`, `get_composition_error`, `get_niche_error`, `predict_neighborhood`,
  `predict_niche_activation`), differential expression with `DifferentialExpressionResults`
  and GPC result, scArches query workflows (fewer-features and same-features), and all
  dispersion modes (`gene`, `gene-batch`, `gene-label`, `gene-cell`)
- Expanded SCVIVA graph dataloader tests to cover graph train + downstream APIs, save/load,
  differential expression, scArches query, and four `@pytest.mark.benchmark` tests
  (speed comparison, ELBO comparison, ELBO decrease, latent shape)
- Added AnnDataLoader-path legacy class `SCVIVALegacy` for SCVIVA benchmark comparisons,
  mirroring the `ResolVILegacy` pattern in `test_resolvi_graph_dataloader.py`
- Expanded DestVI tests to cover `both`/`latent` amortization modes, `get_proportions`
  row-sum correctness, `get_gamma` shape, and `get_scale_for_ct` shape
- Expanded GIMVI tests to cover save/load with prefix, `model_library_size=[True, True]`,
  reinit-and-retrain, `get_imputed_values(normalized=False)`, and load-with-wrong-genes error
- Added `tests/regression/test_graph_vs_ann_spatial_quality.py` with five
  `@pytest.mark.benchmark` tests documenting practical GraphDataLoader advantages:
  - **Edge features**: every `GraphDataLoader` batch is a PyG `Data` object with `edge_index`
    and `edge_attr` (distance weights), enabling future GNN message-passing layers;
    `AnnDataLoader` produces flat dicts with no graph structure
  - **Biological conservation parity**: NMI, ARI, and silhouette (via `scib-metrics`) are
    equal within tolerance across both paths on spatially structured synthetic data, confirming
    graph dataloading does not degrade latent quality
  - **Spatial neighborhood preservation (NNP)**: fraction of spatial k-NN that survive in latent
    space is comparable across paths, establishing the honest baseline
  - **Graph connectivity**: scib-metrics `graph_connectivity` by domain confirms both paths
    meet the minimum connectivity floor required for downstream clustering/trajectory
  - **Cache correctness**: `cache_neighbor_expression=True` populates a finite device-resident
    tensor of shape `(n_obs, n_genes)`, eliminating per-batch AnnTorchDataset random-access reads

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
