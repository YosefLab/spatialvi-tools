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
