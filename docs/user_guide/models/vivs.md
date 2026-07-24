# VIVS

**VIVS** {cite:p}`Boyeau24` (Variational Inference for Variable Selection; Python class {class}`~scviva.model.VIVS`) identifies which genes in a count matrix `X` are conditionally dependent on an external response `Y` — protein expression, niche composition, or any other `obsm` feature — using a conditional randomization test (CRT).

VIVS fits two components: a generative VAE over `X` (reusing scvi-tools' standard `VAE` module, or an already-trained scviva-tools spatial model such as {class}`~scviva.model.SCVIVA`, {class}`~scviva.model.DestVI`, {class}`~scviva.model.ResolVI`, or {class}`~scviva.model.GIMVI`), used only to sample conditional "knockoff" replacements of each gene; and an importance-score network predicting `Y` from `X`, whose per-cell negative log-likelihood is the CRT test statistic. For each gene (or, in the hierarchical variant, each cluster of correlated genes), the statistic is recomputed after substituting in a knockoff sample, yielding a calibrated, BH-corrected p-value for that gene's/cluster's conditional importance to `Y`.

The advantages of VIVS are:

-   Conditional (not marginal) dependence testing — controls for the rest of the transcriptome when assessing each gene's relevance to `Y`.
-   Calibrated false-discovery-rate control via the CRT + Benjamini-Hochberg correction.
-   A hierarchical variant (`get_hier_importance`) that tests at multiple resolutions of correlated gene clusters, improving power for co-regulated genes.
-   Can reuse an already-trained spatial model (e.g. a niche-aware {class}`~scviva.model.SCVIVA` model) as the knockoff sampler, directly integrating with the rest of this toolkit.

The limitations of VIVS include:

-   The knockoff sampler's quality (how well the generative VAE models `p(X_g | X_{-g})`) bounds the test's power.
-   Runtime scales with the number of genes tested; filtering to a smaller gene set (`select_genes`) is recommended above a few thousand genes.
-   Ported from VIVS's original JAX implementation; large-scale runtime is not guaranteed to match the original's `vmap`/`jit`-optimized performance (see `use_vmap` on `get_importance`/`get_hier_importance`).

```{topic} Tutorials:

-   {doc}`/tutorials/VIVS_niche_gene_selection`
```

## Preliminaries

VIVS takes as input a raw-count gene expression matrix `X` and a response matrix `Y` (registered via `y_obsm_key` in {meth}`~scviva.model.VIVS.setup_anndata`). Training proceeds in two sequential phases: first the generative VAE over `X` is fit to convergence (or supplied pretrained via `x_model`), then it is frozen and the importance-score network is fit to predict `Y` from `X`. This order is required for CRT validity — the knockoff sampler must not be contaminated by information about `Y`.
