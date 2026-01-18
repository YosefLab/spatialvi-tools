# spatialvi-tools

**spatialvi-tools** is a Python package for spatial transcriptomics analysis using deep learning. It provides a unified interface for various spatial methods including deconvolution, niche analysis, and cell-cell interaction inference.

## Key Features

- **Spatial VAE Models**: Variational autoencoders that incorporate spatial context
- **Deconvolution Methods**: Cell type proportion estimation for bulk spatial spots
- **Niche Analysis**: Modeling cellular microenvironments
- **Cell-Cell Interactions**: Inference of spatial cell-cell communication
- **Integration with scvi-tools**: Seamless use of scVIVA, ResolVI, and DestVI

## Installation

```bash
pip install spatialvi-tools
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import spatialvi

# Load data
adata = spatialvi.data.synthetic_spatial()

# Compute spatial neighbors
spatialvi.data.compute_spatial_neighbors(adata)

# Setup and train model
SpatialVAE.setup_anndata(adata, spatial_key="spatial")
model = spatialvi.model.SpatialVAE(adata)
model.train(max_epochs=100)

# Get latent representation
latent = model.get_latent_representation()
```

## Contents

```{toctree}
:maxdepth: 2

installation
tutorials/index
api/index
changelog
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
