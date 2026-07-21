# Dead-code candidates: VISION / Harreman

Produced by a repo-wide audit for the VISION tutorial work (see
`docs/tutorials/Vision_tutorial.ipynb` / `DestVI_tutorial.ipynb`). Nothing
here has been removed — this is a review list for Ori and Oier before any
deletion.

## Confirmed dead (zero call sites anywhere in the repo)

- **`src/scviva/tools/vision/tools/diffexp.py:699`** — `_calc_frac()`. Grep
  for `_calc_frac` across the whole repo returns only its own `def` line.
  Superseded by an inline duplicate: `_RankGenes._basic_stats` (same file,
  lines 283-291) reimplements the same "fraction of cells with nonzero
  expression" logic via a local `get_nonzeros` lambda instead of calling
  this function. Likely a leftover from the scanpy `rank_genes_groups` port.

- **`src/scviva/tools/harreman/_analysis.py:622`**
  `_HarremanHsAccessor.integrate_vision_hotspot_results`. Its own docstring
  calls it a deprecated alias for backward compatibility with code written
  against `ha.hs.` before the method moved to `ha.vs.`. Zero call sites in
  `src/`, `tests/`, or any tutorial notebook, and it's excluded from
  `tests/tools/harreman/test_harreman_analysis.py`'s `test_hs_accessor_methods`,
  which otherwise smoke-tests every other `ha.hs.*` method. The live
  implementation is `_HarremanVsAccessor.integrate_vision_hotspot_results`
  (`_analysis.py:679`), which is the one actually used (e.g. referenced from
  `Vision_tutorial.ipynb`'s takeaways as `ha.vs.integrate_vision_hotspot_results()`).

## Untested / not exercised as code (lower confidence — may be intended public API)

- **`src/scviva/tools/vision/_analysis.py:981-1029`** — `class
  _VisionTlAccessor` and its 8 methods (`va.tl.compute_knn_weights`,
  `compute_plasticity_scores`, `cluster_cells_tree`, `apply_micro_clustering`,
  `pool_matrix`, `pool_metadata`, `generate_projections`,
  `rank_genes_groups`). Added specifically to mirror Harreman's `ha.tl.*`
  accessor, but `va.tl.*` was never called as executable code anywhere in
  `tests/`, `src/`, or any notebook — only named in markdown prose in
  `Vision_tutorial.ipynb`. Contrast with Harreman's `_HarremanTlAccessor`,
  which has a dedicated test and real call sites. Not necessarily dead, just
  unverified; consider adding a smoke test (mirroring Harreman's) rather
  than removing.

- **`src/scviva/tools/vision/_constants.py:17,21,23,24`** — 4 constants
  (`TREE_CLUSTERS_OBS_KEY`, `SIGNATURES_VARM_KEY`, `NORM_DATA_KEY_UNS_KEY`,
  `SIGNATURE_VARM_KEY_UNS_KEY`) defined but never referenced anywhere else —
  not even as the underlying string literal (e.g. `"VISION_Clusters_Tree"`)
  used as a magic string in the code that actually writes those keys (e.g.
  `phylo/tree_clustering.py:261` uses a plain `obs_key` parameter default,
  not the constant). Looks like leftovers from the `_constants.py` refactor
  that were never wired in.

## Overlapping, not literal duplication (consolidation candidate, not deletion)

- **`src/scviva/tools/vision/tools/knn.py:98-166`** `compute_knn_weights()`
  vs. **`src/scviva/tools/harreman/tools/knn.py:365-421`**
  `compute_weights()`. Both build an exponential/Gaussian-kernel,
  L1-row-normalized KNN weight matrix, but from different inputs and
  bandwidth conventions: the VISION version builds its own KNN search from a
  latent space and sets per-cell bandwidth to the max K-neighbor distance;
  the Harreman version takes a precomputed `adata.obsp["distances"]` and
  uses a `neighborhood_factor`-based bandwidth rule. Worth a consolidation
  review given the repo's stated goal of harmonizing VISION/Harreman
  architecture, but not a straightforward delete-one-keep-the-other case.

## Housekeeping (not a code issue)

- Stale `.pyc` files under `src/scviva/tools/vision/__pycache__/` for
  pre-refactor flat-layout module paths (e.g. `diffexp.cpython-*.pyc`,
  `filters`, `knn`, `microclusters`, `phylo`, `projections`, `signature`,
  `_gearys_c`, `_normalization`, `_utils`) — leftovers from the
  flat-`vision/*.py` → `vision/tools|phylo|preprocessing/*.py` reorg. Every
  current `.py` file is imported by some package `__init__.py`; this is just
  stale bytecode cache that can be cleaned (`find . -name __pycache__ -exec
  rm -rf {} +`), unrelated to source dead-code.
