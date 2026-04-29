# scviva-tools

**scviva-tools** is a consolidated spatial transcriptomics analysis toolkit built on top of [scvi-tools](https://scvi-tools.org), exposing ResolVI, DestVI, and scVIVA through a clean, unified API.

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
