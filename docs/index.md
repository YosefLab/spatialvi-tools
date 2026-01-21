# spatialvi‑tools documentation

Welcome to the **spatialvi‑tools** documentation.  This library unifies
multiple algorithms for spatial omics analysis under a consistent
interface.  It is inspired by the structure of `scvi‑tools` and provides
wrappers around several state‑of‑the‑art methods, allowing you to
combine them seamlessly in your analyses.

## Overview of included models

| Model           | Purpose | Key reference |
|-----------------|---------|---------------|
| **NolanModel**  | Uncovers spatial niches and maps the tissue graph using a self‑supervised approach【688704965977238†L7-L13】 | Bakulin *et al.* 2025 |
| **LambdaModel** | Performs reference‑free cell type annotation with LLMs【245548828756402†L24-L32】 | Bakulin *et al.* 2025 |
| **PPIInference**| Computes prediction‑powered confidence intervals for statistical estimands【569057324233913†L11-L14】 | Angelopoulos *et al.* 2023 |
| **VIVSModel**   | Selects important genes/features via variational inference【666593781079131†L1-L8】 | YosefLab 2023 |
| **HarremanModel** | Infers metabolic exchanges and spatial correlation statistics【504187329151369†L11-L13】 | Harreman team 2025 |
| **AmiciModel**  | Uses cross‑attention to infer cell–cell interactions【767658563887085†L7-L10】 | Hong *et al.* 2025 |
| **StarfyshModel** | Performs reference‑free deconvolution and histology integration【835840889957633†L6-L10】 | He *et al.* 2024 |
| **SparlModel**  | Learns latent representations from spatial proteomics【218342748980084†L8-L10】 | Binder *et al.* 2025 |

Each wrapper class follows a simple pattern: initialise the model with
an `AnnData` object, call `.train()` to fit the underlying algorithm,
then call `.predict()` (or `.infer()`) to compute embeddings or
annotations.  Many of the wrapped algorithms have their own rich
configuration options; consult the individual docstrings or the
upstream packages for details.

This documentation is a work in progress.  For now, please refer to
the README files of the original projects for comprehensive tutorials
and theoretical background.  You can find links to those projects in
the top‑level `README.md` of this repository.