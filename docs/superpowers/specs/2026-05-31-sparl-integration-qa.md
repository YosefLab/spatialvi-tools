# SPARL Integration — Q&A Log

Running record of clarifying questions and answers during SPARL integration brainstorming.
Used to inform the final design spec.

---

## Q1 — Image input format

**Q:** How do you expect SPARL's image inputs to be provided in the scviva-tools context?

Options considered:
- A) Disk paths in AnnData `obs` metadata; SPARL's existing ImcDataset/JUMPCPDataset reads them at training time
- B) Images pre-loaded into AnnData `obsm['X_img']` as `(n_cells, C, H, W)` array
- C) Via SpatialData object; `setup_spatialdata()` / `from_spatialdata()` extracts patches
- D) No tight data coupling; wrapper only calls SPARL training and writes embeddings back to AnnData

**A (partial, superseded by Q2 context):** Not yet answered directly — user clarified the usage mode first (see below).

---

## Q2 — Usage mode: inference-only + optional fine-tuning

**User clarification (before Q1 was answered):**

> scviva-tools will NOT use SPARL in pre-training mode.
> Users load a pre-trained model from a model hub and run inference on their data.
> Fine-tuning on user data is in scope, but foundation model SSL pre-training is NOT.

**Impact on design:**

| Concern | Before clarification | After clarification |
|---------|---------------------|---------------------|
| Training infrastructure | Must wrap Lightning Fabric + DDP + DINO/iBOT SSL loop | Not needed — pre-training stays in SPARL standalone |
| SSLMetaArch | In scope | Out of scope for scviva-tools |
| ClassificationMetaArch | Maybe | In scope — needed for fine-tuning |
| Complex augmentations (global/local crops, masking) | In scope | Out of scope — inference uses simple normalization |
| Data loading complexity | Full ImcDataset/JUMPCPDataset pipeline | Simplified — just image loading + normalization for inference |
| Model loading | Checkpoint from training run | Load from model hub (HuggingFace or custom) |
| scvi-tools TrainingPlan | Maybe | Relevant only for fine-tuning head |
| Losses (DINO, iBOT, KoLeo) | In scope | Out of scope |
| Core value scviva-tools adds | Training wrapper | Inference API + AnnData I/O + SpatialData interop + ecosystem integration |

**Summary:** Integration scope narrows dramatically. scviva-tools wraps SPARL for:
1. Loading pre-trained backbone from hub
2. Running inference → CLS token embeddings → AnnData `obsm`
3. Optional fine-tuning (linear probe or full fine-tune) via scvi-tools TrainingPlan
4. SpatialData interop (via existing `setup_spatialdata()` pattern)

---

## Open questions (to be answered in subsequent brainstorming turns)

- ~~**Q3:** What kind of fine-tuning is in scope?~~ **Answered: D — no fine-tuning in v1. Inference + embeddings only.**
- **Q4:** What are the fine-tuning targets? (cell type annotation, spatial niche, perturbation response, other)
- ~~**Q5:** How are per-cell images provided?~~ **Answered: D — both disk paths in `obs` (`setup_anndata()`) and SpatialData (`from_spatialdata()`). Two entry points, same inference engine.**
- ~~**Q6:** Which model hub?~~ **Answered: C — `from_pretrained()` accepts either a HuggingFace repo ID string or a local checkpoint path. Standard scverse pattern.**
- ~~**Q7:** Should the ViT backbone be exposed directly?~~ **Answered: A — high-level only for v1. `get_latent_representation()` is the public API. Backbone exposure deferred.**
- ~~**Q8:** Multi-GPU inference?~~ **Deferred — not in v1 scope.**

---

## Integration approach decision

**Chosen: Approach 2 — new `scviva/imaging/` submodule with `ImagingBaseModel` base class.**

Rationale: more image-based models are planned beyond SPARL. `ImagingBaseModel` will serve as the shared foundation for all of them, avoiding future refactoring.
