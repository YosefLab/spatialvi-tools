# spatialvi-tools

Consolidated spatial transcriptomics analysis toolkit based on [scvi-tools](https://scvi-tools.org).

Provides three spatial transcriptomics models as first-class citizens:
- **scVIVA** — niche-aware representation learning (Levy et al., 2025)
- **DestVI** — multi-resolution cell-type deconvolution (Lopez et al., 2022)
- **ResolVI** — noise and bias correction for cellular-resolution ST (Ergen & Yosef, 2025)

## Installation

```bash
pip install spatialvi-tools
# With SpatialData and squidpy integration:
pip install "spatialvi-tools[spatial]"
```

## Quick Start

```python
import spatialvi

# scVIVA
spatialvi.SCVIVA.setup_anndata(adata, layer="counts", spatial_key="spatial")
model = spatialvi.SCVIVA(adata)
model.train()

# DestVI
import scvi
scvi.model.CondSCVI.setup_anndata(sc_adata, labels_key="cell_type", layer="counts")
sc_model = scvi.model.CondSCVI(sc_adata)
sc_model.train()

spatialvi.DestVI.setup_anndata(st_adata, layer="counts")
st_model = spatialvi.DestVI.from_rna_model(st_adata, sc_model)
st_model.train()

# ResolVI
spatialvi.ResolVI.setup_anndata(adata, layer="counts", spatial_key="spatial")
model = spatialvi.ResolVI(adata)
model.train()
```

## References

See [docs/references.md](docs/references.md) for full citations.
