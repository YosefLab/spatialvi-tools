# References

This document lists the primary references for all models, methods, and tools used in
`spatialvi-tools`. BibTeX entries are in [`references.bib`](references.bib).

---

## Core Models

### scVIVA

> **Levy et al. (2025)**
> *scVIVA: Variational Auto-Encoder with Niche Decoders for Spatial Transcriptomics*
> Preprint / Publication.
> [cite:Levy25]

scVIVA models gene expression as a function of both the cell's intrinsic state and its
microenvironment (niche). It uses niche-conditioned decoders to disentangle cell-intrinsic
from environment-driven variation, enabling niche-aware differential expression.

### DestVI

> **Lopez et al. (2022)**
> *DestVI: multi-resolution deconvolution of spatial transcriptomics data*
> *Nature Methods*, 19, 1438–1444.
> [cite:Lopez22]

DestVI performs multi-resolution deconvolution of spatial transcriptomics spots into
cell-type compositions. It uses a conditional SCVI (CondSCVI) model trained on scRNA-seq
as a reference, then maps cell types to spatial spots using a deconvolution module (MRDeconv).

### ResolVI

> **Marks et al. (2025)**
> *ResolVI: Addressing Noise and Bias in Single-Cell Resolved Spatial Transcriptomics*
> Preprint.
> [cite:Marks25]

ResolVI corrects segmentation errors, background signal, and cell-size bias in
cellular-resolution spatial transcriptomics data (Xenium, MERFISH, CosMx). It uses
a Pyro-based probabilistic model with neighbor-aware decoders.

---

## Foundation: scvi-tools

> **Gayoso et al. (2022)**
> *A Python library for probabilistic analysis of single-cell omics data*
> *Nature Biotechnology*, 40, 163–166.
> [cite:Gayoso22]

`spatialvi-tools` is built on top of scvi-tools. All base model classes, training
infrastructure, and AnnData registration utilities are inherited from scvi-tools.

> **Lopez et al. (2018)**
> *Deep generative modeling for single-cell transcriptomics*
> *Nature Methods*, 15, 1053–1058.
> [cite:Lopez18]

The original SCVI model underpinning scvi-tools and the VAE architecture used by scVIVA.

---

## scverse Ecosystem

### AnnData

> **Virshup et al. (2024)**
> *The scverse project provides a computational ecosystem for single-cell omics data analysis*
> *Nature Biotechnology*, 42, 333–336.
> [cite:Virshup24]

AnnData is the primary data structure used throughout spatialvi-tools.

### SpatialData

> **Marconato et al. (2024)**
> *SpatialData: an open and universal data framework for spatial omics*
> *Nature Methods*.
> [cite:Marconato24]

SpatialData provides the unified framework for spatial omics data. `spatialvi-tools`
supports `setup_spatialdata()` and `from_spatialdata()` constructors for all models.

### squidpy

> **Palla et al. (2022)**
> *Squidpy: a scalable framework for spatial omics analysis*
> *Nature Methods*, 19, 171–178.
> [cite:Palla22]

squidpy is used as the default backend for spatial neighbor graph computation
(`compute_neighbors(backend="squidpy")`).

### scanpy

> **Wolf et al. (2018)**
> *SCANPY: large-scale single-cell gene expression data analysis*
> *Genome Biology*, 19, 15.
> [cite:Wolf18]

scanpy is a core dependency for preprocessing and visualization.

---

## GPU Acceleration

### RAPIDS (cuML / cuGraph)

> **Raschka et al. (2020)** and NVIDIA Corporation
> *Machine Learning in Python: Main developments and technology trends in data science,
> machine learning, and artificial intelligence*
> [cite:RAPIDS]

RAPIDS provides GPU-accelerated alternatives to scikit-learn (cuML) and NetworkX (cuGraph).
Used via `backend="rapids"` parameter in `compute_neighbors()` and
`get_latent_representation()`.

---

## Deep Learning Framework

### PyTorch

> **Paszke et al. (2019)**
> *PyTorch: An Imperative Style, High-Performance Deep Gradient Computing Library*
> *NeurIPS 2019*.
> [cite:Paszke19]

### PyTorch Lightning

> **Falcon et al. (2019)**
> *PyTorch Lightning*
> GitHub. https://github.com/Lightning-AI/pytorch-lightning
> [cite:Falcon19]

### Pyro

> **Bingham et al. (2019)**
> *Pyro: Deep Universal Probabilistic Programming*
> *Journal of Machine Learning Research*, 20(28), 1–6.
> [cite:Bingham19]

Pyro is used as the probabilistic programming backend for ResolVI.

---

## Future External Models (tracked for later integration)

These references correspond to models in `spatialvi.external` (not yet in v1):

| Model | Reference |
|-------|-----------|
| Stereoscope | Andersson et al. (2020), *Commun Biol* |
| Tangram | Biancalani et al. (2021), *Nature Methods* |
| Cell2location | Kleshchevnikov et al. (2022), *Nature Biotechnology* |
| starfysh | Chang et al. (2023), preprint |
| VIVS | (reference TBD) |
| SPARL | (reference TBD) |
