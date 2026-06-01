# External AMICI and Starfysh Integration Design

**Date:** 2026-06-01
**Status:** Draft for review
**Scope:** Design only. No implementation and no commits in this step.
**Sources:**
- AMICI upstream: `/Users/orikr/PycharmProjects/amici/src/amici`
- Starfysh upstream: `/Users/orikr/PycharmProjects/starfysh/starfysh`
- SPARL digestion reference: `/Users/orikr/PycharmProjects/spatialvi-tools2/src/scviva/imaging/sparl`
- Architecture reference: `/Users/orikr/PycharmProjects/spatialvi-tools/docs/architecture/scviva-tools-block-diagram_inc_sparl_harreman.html`

---

## Goal

Integrate AMICI and Starfysh into `scviva-tools` as external spatial transcriptomics models using the same "digest into scviva building blocks" pattern used for SPARL, Tangram, and Stereoscope.

The integration should start from the simplest useful core modeling surface and move gradually toward harder upstream capabilities. The target design accounts for the full upstream capability surface, but implementation should not assume every capability must land immediately.

---

## Placement Decision

AMICI and Starfysh belong in `scviva.external`, not `scviva.model`, because they are imported methods with their own upstream identity and APIs.

Both belong to the spatial transcriptomics track:

- `AMICI` is a neighborhood-aware spatial transcriptomics model.
- `Starfysh` is a spatial deconvolution model. Its PoE path uses histology, but the model remains a spatial transcriptomics/deconvolution method, not an imaging backbone like SPARL.

SPARL remains the reference for integration style, not for model category. SPARL was digested into a scviva wrapper plus a small module adapter instead of exposing raw upstream internals directly. AMICI and Starfysh should follow the same package discipline.

---

## High-Level Architecture

```text
scviva.external
├── amici
│   ├── AMICI                  # scviva wrapper, inherits SpatialBaseModel
│   ├── AMICIModule            # PyTorch/scvi module
│   ├── spatial dataloader     # neighbor-aware AnnData loading
│   ├── interpretation         # attention, counterfactual, explained variance, ablation
│   └── callbacks              # optional training/logging callbacks
└── starfysh
    ├── Starfysh               # scviva wrapper, inherits SpatialBaseModel + deconv mixin
    ├── StarfyshModule         # expression AVAE core
    ├── StarfyshPoEModule      # optional histology PoE core
    ├── preprocessing          # Visium args, signatures, anchors, library smoothing
    ├── archetypes             # archetypal analysis and anchor refinement
    └── plotting/postanalysis  # optional plots and result helpers
```

The wrapper classes own the scviva-facing API. The modules preserve the upstream math. Helper packages are added only when they support a tested public method.

---

## Shared Building Blocks

### Existing scviva foundations

Both integrations should reuse existing scviva foundations where possible:

- `SpatialBaseModel` for spatial model identity, SpatialData entry points, and spatial plotting.
- `SpatialDeconvolutionMixin` for Starfysh proportion formatting and cell-type maps.
- `SpatialNeighborhoodMixin` concepts for AMICI neighbor registration, while preserving AMICI's label-aware neighbor exclusion behavior.
- `AnnDataManager` field registration and scvi data-loader conventions.
- Existing `tests/external` style from Tangram and Stereoscope.

### New shared pieces to consider

AMICI has a stronger neighbor-data-loader abstraction than the current mixin alone. If useful after the first slice, extract a reusable neighbor dataset/dataloader into `scviva.dataloaders` or `scviva.data` rather than keeping it private to AMICI forever.

Starfysh has preprocessing, signature handling, anchor detection, and image-patch extraction. These should start private to `scviva.external.starfysh` and become shared utilities only if a second model needs them.

---

## Capability Inventory

### AMICI upstream capabilities

Core:

- `AMICI` scvi-style model class.
- `AMICIModule` with attention over spatial neighbors.
- `SpatialAnnDataLoader` and `SpatialAnnTorchDataset` for neighborhood-aware loading.
- Nearest-neighbor computation with label exclusion and optional cell radius adjustment.
- `get_predictions`, `get_nn_embed`, and gene residual contribution utilities.

Interpretation:

- Attention pattern extraction with vanilla, value-weighted, info-weighted, and gene-weighted flavors.
- Communication hub analysis.
- Counterfactual attention patterns.
- Counterfactual length-scale summaries.
- Explained variance scores and plots.
- Neighbor/cell-type/head ablation scores.

Training/logging:

- W&B training runner and training mixin.
- Attention penalty callback.
- Model interpretation logging callback.

### Starfysh upstream capabilities

Core:

- `AVAE` expression-only deconvolution model.
- `AVAE_PoE` expression plus histology product-of-experts model.
- Negative binomial likelihood helper.
- Multi-restart training wrapper.
- Evaluation helpers that write inference/generative outputs to AnnData.
- Cell-type-specific inferred expression.
- PoE image reconstruction utilities.

Preprocessing and data:

- Visium preprocessing.
- Signature loading and filtering.
- `VisiumArguments` and integrated-data argument variants.
- Windowed library-size smoothing.
- Signature gene score calculation.
- Anchor spot detection and refinement.
- Histology image preprocessing and patch extraction.

Analysis and plotting:

- Archetypal analysis.
- Marker finding and archetype assignment.
- Anchor refinement from archetypes.
- Spatial feature plots, cell-type fraction plots, UMAP plots.
- Reconstruction plots, density plots, correlation maps, Moran/LISA/SCI helpers.

---

## Phased Integration

### Phase 1: Smallest core surface

Goal: make both models importable and testable under `scviva.external` without committing to every upstream helper.

AMICI:

- Create `scviva.external.amici`.
- Add `AMICIModule`, minimal components, constants, and a wrapper `AMICI`.
- Make `AMICI` inherit `SpatialBaseModel` plus the needed scvi training mixins.
- Implement `setup_anndata` with labels, expression layer, spatial coordinates, and AMICI neighbor arrays.
- Add a tiny synthetic-data test for setup, construction, one forward pass or one short train, and prediction shape.

Starfysh:

- Create `scviva.external.starfysh`.
- Add expression-only `StarfyshModule` based on upstream `AVAE`.
- Add wrapper `Starfysh` inheriting `SpatialDeconvolutionMixin` and `SpatialBaseModel`.
- Implement a small `setup_anndata` contract for raw counts, spatial coordinates, signature matrix, library-size input, and cell-type names.
- Add a tiny synthetic-data test for setup, construction, one train step or one epoch, proportions shape, and no NaNs.

Phase 1 explicitly excludes Starfysh PoE, Starfysh full Visium preprocessing, AMICI interpretation modules, and plotting.

### Phase 2: Output contract and basic user methods

Goal: make both models useful in the scviva pipeline.

AMICI:

- Add `get_predictions`.
- Add residual retrieval.
- Add `get_nn_embed` if the data-loader contract is stable.
- Store outputs under predictable keys in `adata.obsm` when requested.

Starfysh:

- Add deconvolution/proportion retrieval through the `SpatialDeconvolutionMixin` contract.
- Add `get_latent_representation` or a Starfysh-specific latent getter that writes `qz_m`-like outputs to `adata.obsm`.
- Add `model_eval`-equivalent behavior as a method that returns structured outputs and optionally writes to AnnData.
- Add cell-type-specific inferred expression.

### Phase 3: Reusable neighbor and deconvolution infrastructure

Goal: reduce duplication only after core behavior is proven.

AMICI:

- Decide whether AMICI's `SpatialAnnDataLoader` should become shared `scviva.dataloaders`.
- Align neighbor keys with scviva conventions while preserving AMICI-specific keys needed by the module.
- Add tests for sparse, dense, and DataFrame-backed neighbor loading if feasible.

Starfysh:

- Tighten `SpatialDeconvolutionMixin` compatibility.
- Move repeated proportion formatting or result writing into focused helper functions if shared with Stereoscope or DestVI.

### Phase 4: Starfysh PoE and histology

Goal: add Starfysh's histology-aware mode without confusing it with the SPARL imaging track.

- Add `StarfyshPoEModule` based on upstream `AVAE_PoE`.
- Add image-patch extraction and registration under `scviva.external.starfysh`, not `scviva.imaging`.
- Reuse image loading ideas from `ImagingBaseModel` only where the data shape and semantics match.
- Add tests with tiny in-memory or temporary image patches.
- Add image reconstruction utilities after PoE inference is stable.

### Phase 5: AMICI interpretation modules

Goal: expose AMICI's analysis value once core training and predictions are reliable.

- Add attention pattern module.
- Add counterfactual attention module.
- Add explained variance module.
- Add ablation module.
- Add save methods and DataFrame shape tests.
- Add plotting tests only for smoke-level figure creation, not visual correctness.

### Phase 6: Starfysh preprocessing, archetypes, and plotting

Goal: port richer workflow helpers selectively.

- Add `VisiumArguments`-like preprocessing only if it fits scviva style.
- Add signature loading/filtering helpers.
- Add anchor spot detection and refinement.
- Add archetypal analysis.
- Add plotting and post-analysis helpers.

This phase should be selective. File-system-heavy Visium loaders, broad plotting utilities, and simulation-specific helpers can remain deferred if they do not serve the scviva pipeline.

### Phase 7: Optional logging and external workflow features

Goal: add convenience features only after the scientific API is stable.

- Add AMICI W&B callbacks as optional extras.
- Keep W&B imports lazy or optional.
- Add Starfysh file-loading convenience wrappers only if examples require them.

---

## Public API Shape

Initial API should be conservative:

```python
from scviva.external import AMICI, Starfysh
from scviva.external.amici import AMICIModule
from scviva.external.starfysh import StarfyshModule
```

Later API can expand:

```python
from scviva.external.starfysh import StarfyshPoEModule, ArchetypalAnalysis
from scviva.external.amici import (
    AMICIAttentionModule,
    AMICICounterfactualAttentionModule,
    AMICIExplainedVarianceModule,
    AMICIAblationModule,
)
```

Avoid top-level `scviva.AMICI` and `scviva.Starfysh` until the APIs are stable. Existing external models are already accessed through `scviva.external`, so this is consistent.

---

## Testing Strategy

Tests should land with each phase.

Phase 1 tests:

- Import from `scviva.external`.
- `setup_anndata` registers required fields.
- Model construction succeeds on synthetic AnnData.
- One forward pass or one short training run completes on CPU.
- Output shapes are stable.
- Outputs have no NaNs.

Phase 2 tests:

- Public getters return arrays/DataFrames with expected shapes.
- Optional AnnData writes use documented keys.
- Starfysh proportions sum to valid ranges where mathematically expected.
- AMICI predictions and residuals align with `adata.n_obs` and `adata.n_vars`.

Later tests:

- AMICI interpretation modules return DataFrames with required columns.
- Starfysh PoE accepts tiny image patches and returns expression plus image outputs.
- Plotting tests use non-interactive Matplotlib backend and assert figure creation only.
- Optional dependency tests skip cleanly when packages are missing.

Regression tests against upstream should be added only after the wrapper behavior is stable. The first priority is scviva contract correctness, not full bitwise equivalence.

---

## Dependency Strategy

Core dependencies should not expand until needed.

Likely optional extras:

- `amici` extra for `einops`, `transformer-lens`, `networkx`, `statsmodels`, `openchord`, `wandb`.
- `starfysh` extra for Starfysh-specific preprocessing and plotting dependencies such as `histomicstk`, `skimage`, `py_pcha`, `skdim`, and `umap-learn`.

Avoid importing optional packages at module import time. Use lazy imports inside methods or skip tests with clear messages.

---

## Risks and Mitigations

AMICI neighbor loading differs from current scviva neighbor mixins.

Mitigation: keep AMICI's loader local during Phase 1, then extract shared infrastructure only after tests show the contract is stable.

Starfysh mixes model code, preprocessing, plotting, and file-system workflows.

Mitigation: start with model and output contracts. Port preprocessing and plotting later only where they serve scviva workflows.

Starfysh PoE uses histology but is not an imaging model.

Mitigation: keep PoE under `scviva.external.starfysh`, document it as optional histology evidence for deconvolution, and do not route it through `scviva.imaging`.

Optional dependencies may make imports fragile.

Mitigation: keep optional imports lazy and tests skippable.

The full upstream capability surface is large.

Mitigation: use the phase ladder as a decision gate. Each phase can stop after useful behavior lands.

---

## Non-Goals for the First Implementation Slice

The first implementation slice will not include:

- Starfysh PoE/histology.
- Starfysh file loading from Visium folders.
- Starfysh archetypal analysis.
- Starfysh plotting/post-analysis helpers.
- AMICI attention/counterfactual/explained-variance/ablation modules.
- AMICI W&B logging and training callbacks.
- Documentation tutorials.
- Full upstream regression equivalence.

These are intentionally later phases.

---

## Approval Gate

After this design is reviewed, the next step is an implementation plan for Phase 1 only:

- `scviva.external.amici` minimal model/module/setup/test surface.
- `scviva.external.starfysh` expression-only model/module/setup/test surface.
- No commits unless explicitly requested.
