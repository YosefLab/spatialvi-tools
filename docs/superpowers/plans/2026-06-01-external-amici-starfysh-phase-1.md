# External AMICI and Starfysh Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the smallest useful `scviva.external.amici` and `scviva.external.starfysh` model surfaces with tests and an architecture diagram update.

**Architecture:** Both integrations live under `scviva.external` and reuse `SpatialBaseModel`, scvi field registration, and existing external-package export patterns. AMICI keeps its attention-neighbor model/module logic in a compact Phase 1 form; Starfysh keeps the expression AVAE deconvolution logic in a compact Phase 1 form. Larger upstream capabilities remain later phases.

**Tech Stack:** Python, torch, scvi-tools, AnnDataManager, pytest, existing scviva base classes and mixins.

**Spec:** `docs/superpowers/specs/2026-06-01-external-amici-starfysh-design.md`

**User constraints:** Do not commit. Keep simple. Tests go in `tests/external`. Update `docs/architecture/scviva-tools-block-diagram.html` after the phase. Reuse scviva code and keep the model/module logic from AMICI and Starfysh.

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
