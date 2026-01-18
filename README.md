# spatialvi-tools

[![CI](https://github.com/YosefLab/spatialvi-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/YosefLab/spatialvi-tools/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/spatialvi-tools/badge/?version=latest)](https://spatialvi-tools.readthedocs.io/)
[![PyPI](https://img.shields.io/pypi/v/spatialvi-tools.svg)](https://pypi.org/project/spatialvi-tools/)
[![Python Version](https://img.shields.io/pypi/pyversions/spatialvi-tools)](https://pypi.org/project/spatialvi-tools/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A consolidated toolkit for spatial transcriptomics analysis using variational inference methods.

## Overview

spatialvi-tools provides a unified framework for analyzing spatial transcriptomics data, bringing together multiple specialized models under a consistent API built on [scvi-tools](https://scvi-tools.org/).

### Key Features

- **Spatial VAE Models**: Deep generative models that incorporate spatial context for improved latent representations
- **Deconvolution**: Cell type proportion estimation from spatial spots using reference single-cell data
- **Niche Analysis**: Cellular microenvironment modeling with attention-based neighbor aggregation
- **Cell-Cell Interactions**: AMICI module for modeling cell-cell communication through cross-attention
- **External Model Wrappers**: Unified interfaces to DestVI, scVIVA, ResolVI, Starfysh, Harreman, VIVS, and more

## Installation

```bash
pip install spatialvi-tools
```

For development installation:

```bash
git clone https://github.com/YosefLab/spatialvi-tools.git
cd spatialvi-tools
pip install -e ".[dev,test]"
```

## Quick Start

```python
import scanpy as sc
from spatialvi.data import compute_spatial_neighbors, compute_niche_composition
from spatialvi.model import SpatialVAE

# Load spatial transcriptomics data
adata = sc.read_h5ad("spatial_data.h5ad")

# Compute spatial neighbors
compute_spatial_neighbors(adata, n_neighbors=20)

# Setup and train model
SpatialVAE.setup_anndata(
    adata,
    batch_key="batch",
    spatial_key="spatial",
)
model = SpatialVAE(adata)
model.train(max_epochs=100)

# Get latent representation
latent = model.get_latent_representation()
```

## Available Models

### Core Modules

| Module | Description |
|--------|-------------|
| `SpatialVAEModule` | VAE with spatial context encoding |
| `DeconvolutionModule` | Cell type deconvolution with Dirichlet priors |
| `NicheModule` | Niche-aware analysis with cell type prediction |

### External Model Integrations

| Model | Description |
|-------|-------------|
| `DestVI` | Multi-resolution spatial deconvolution |
| `scVIVA` | Spatial environment effect modeling |
| `ResolVI` | Cellular-resolved spatial denoising |
| `AMICI` | Cell-cell interaction inference |
| `Starfysh` | Spatial deconvolution with histology |
| `Harreman` | Metabolic exchange modeling |
| `VIVS` | Variable importance analysis |

## Architecture

```
spatialvi-tools/
├── src/spatialvi/
│   ├── module/          # Core PyTorch modules
│   ├── model/           # High-level model classes
│   ├── nn/              # Neural network components
│   ├── external/        # External model wrappers
│   ├── data/            # Data preprocessing
│   ├── train/           # Training utilities
│   └── utils/           # Metrics and visualization
├── tests/               # Comprehensive test suite
└── docs/                # Documentation
```

## Documentation

Full documentation is available at [spatialvi-tools.readthedocs.io](https://spatialvi-tools.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -e ".[dev,test]"`)
4. Install pre-commit hooks (`pre-commit install`)
5. Make your changes and add tests
6. Run tests (`pytest tests/`)
7. Submit a pull request

## Citation

If you use spatialvi-tools in your research, please cite:

```bibtex
@software{spatialvi_tools,
  title = {spatialvi-tools: A unified toolkit for spatial transcriptomics analysis},
  author = {YosefLab},
  year = {2024},
  url = {https://github.com/YosefLab/spatialvi-tools}
}
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [scvi-tools](https://scvi-tools.org/)
- Inspired by methods from the single-cell and spatial transcriptomics communities
