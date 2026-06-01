# scviva-tools

**scviva-tools** is a consolidated spatial transcriptomics analysis toolkit built on top of [scvi-tools](https://scvi-tools.org), exposing ResolVI, DestVI, and scVIVA through a clean, unified API.

```{toctree}
:maxdepth: 1
:hidden:

installation
user_guide/index
architecture/index
api/index
developer/index
tutorials/index
faq
references
```

## Models

::::{grid} 3
:gutter: 2

:::{grid-item-card} ResolVI
:link: user_guide/models/resolvi
:link-type: doc

Denoising and segmentation-error correction for single-cell resolved spatial transcriptomics.
:::

:::{grid-item-card} DestVI
:link: user_guide/models/destvi
:link-type: doc

Cell-type deconvolution of spatial transcriptomics spots with sub-cell-type resolution.
:::

:::{grid-item-card} scVIVA
:link: user_guide/models/scviva
:link-type: doc

Niche-aware variational inference for cellular microenvironment modeling.
:::

::::

## Architecture

::::{grid} 1
:gutter: 2

:::{grid-item-card} scviva-tools block diagram
:link: architecture/index
:link-type: doc

Visual map of the package layers, inheritance, mixins, model groups, and data-flow rail.
:::

::::
