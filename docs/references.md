# References

BibTeX entries are in [`references.bib`](references.bib).

---

## Core Models

### scVIVA

> **Levy et al. (2025)**
> *scVIVA: a probabilistic framework for representation of cells and their environments in spatial transcriptomics*
> bioRxiv. doi: [10.1101/2025.06.01.657182](https://doi.org/10.1101/2025.06.01.657182)
> `cite:Levy25`

scVIVA models gene expression as a function of both the cell's intrinsic state and its
microenvironment (niche). Niche-conditioned decoders disentangle cell-intrinsic from
environment-driven variation, enabling niche-aware differential expression.

### DestVI

> **Lopez et al. (2022)**
> *DestVI identifies continuums of cell types in spatial transcriptomics data*
> *Nature Biotechnology*. doi: [10.1038/s41587-022-01272-8](https://doi.org/10.1038/s41587-022-01272-8)
> `cite:Lopez22`

DestVI performs multi-resolution deconvolution of spatial transcriptomics spots into
cell-type compositions using a conditional SCVI reference trained on scRNA-seq.

### ResolVI

> **Ergen & Yosef (2025)**
> *ResolVI - addressing noise and bias in spatial transcriptomics*
> bioRxiv. doi: [10.1101/2025.01.20.634005](https://doi.org/10.1101/2025.01.20.634005)
> `cite:Ergen25`

ResolVI corrects segmentation errors, background signal, and cell-size bias in
cellular-resolution spatial transcriptomics (Xenium, MERFISH, CosMx) using a
Pyro-based probabilistic model with neighbor-aware decoders.

---

## Foundation: scvi-tools

> **Gayoso et al. (2022)**
> *A Python library for probabilistic analysis of single-cell omics data*
> *Nature Biotechnology*, 40, 163–166. doi: [10.1038/s41587-021-01206-w](https://doi.org/10.1038/s41587-021-01206-w)
> `cite:Gayoso22`

> **Lopez et al. (2018)**
> *Deep generative modeling for single-cell transcriptomics*
> *Nature Methods*, 15, 1053–1058. doi: [10.1038/s41592-018-0229-2](https://doi.org/10.1038/s41592-018-0229-2)
> `cite:Lopez18`

> **Ergen et al. (2025)**
> *Scvi-hub: an actionable repository for model-driven single-cell analysis*
> *Nature Methods*, 22, 1836–1845. doi: [10.1038/s41592-025-02799-9](https://doi.org/10.1038/s41592-025-02799-9)
> `cite:Ergen25-2`

---

## scverse Ecosystem

> **Virshup et al. (2024)** — The scverse project
> *Nature Biotechnology*, 42, 333–336. doi: [10.1038/s41587-023-01733-8](https://doi.org/10.1038/s41587-023-01733-8)
> `cite:Virshup24`

> **Palla et al. (2022)** — squidpy
> *Nature Methods*, 19, 171–178. doi: [10.1038/s41592-021-01358-2](https://doi.org/10.1038/s41592-021-01358-2)
> `cite:Palla22`

> **Wolf et al. (2018)** — scanpy
> *Genome Biology*, 19, 15. doi: [10.1186/s13059-017-1382-0](https://doi.org/10.1186/s13059-017-1382-0)
> `cite:Wolf18`

---

## Deep Learning Frameworks

> **Paszke et al. (2019)** — PyTorch. NeurIPS 2019. `cite:Paszke19`

> **Falcon et al. (2019)** — PyTorch Lightning. `cite:Falcon19`

> **Bingham et al. (2019)** — Pyro. *JMLR*, 20(28). `cite:Bingham19`

---

## GPU Acceleration

> **NVIDIA Corporation** — RAPIDS (cuML / cuGraph). https://rapids.ai `cite:RAPIDS`

---

## Future External Models

| Model | Reference | Key |
|-------|-----------|-----|
| Stereoscope | Andersson et al. (2020), *Commun Biol* | `cite:Andersson20` |
| Tangram | Biancalani et al. (2021), *Nature Methods* | `cite:Biancalani21` |
| Cell2location | Kleshchevnikov et al. (2022), *Nature Biotechnology* | `cite:Kleshchevnikov22` |
| starfysh | Chang et al. (2023) — BibTeX entry TBD | — |
| VIVS / SPARL / others | References TBD | — |
