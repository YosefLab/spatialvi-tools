# Changelog

All notable changes to spatialvi-tools will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release preparation

## [0.1.0] - 2024-01-15

### Added

#### Core Framework
- `BaseSpatialModel` - Base class for all spatial transcriptomics models
- `BaseSpatialModule` - Base PyTorch Lightning module with spatial-specific functionality
- Spatial-aware data loading with `AnnDataManager` integration
- Custom registry keys for spatial data (`SPATIAL_REGISTRY_KEYS`)

#### Modules
- `SpatialVAEModule` - Variational autoencoder with spatial context encoding
- `DeconvolutionModule` - Cell type deconvolution for spatial spots
- `NicheModule` - Cellular niche modeling with attention mechanisms

#### Neural Network Components
- `SpatialEncoder` - Encoder with neighbor aggregation
- `GraphEncoder` - Graph convolutional encoder
- `AttentionEncoder` - Transformer-based encoder
- `SpatialDecoder` - Spatial-aware decoder
- `SpatialAttention`, `CrossAttention`, `NeighborAttention` - Attention layers
- `GATLayer` - Graph attention layer
- `SpatialConv`, `GraphConv` - Spatial/graph convolution layers
- `PositionalEncoding` - Spatial positional encoding

#### Training
- `SpatialTrainingPlan` - Training plan with spatial loss
- `NicheTrainingPlan` - Semi-supervised niche training
- `DeconvolutionTrainingPlan` - Deconvolution-specific training
- `SpatialMetricsCallback` - Spatial metrics during training
- `NeighborSamplingCallback` - Neighbor-aware batch sampling
- `EarlyStoppingOnSpatialLoss` - Spatial loss-based early stopping
- `SpatialRegularizationScheduler` - Regularization scheduling

#### Data Utilities
- `compute_spatial_neighbors()` - K-nearest neighbor computation
- `compute_niche_composition()` - Neighborhood cell type composition
- `normalize_spatial()` - Spatial coordinate normalization
- `filter_by_spatial_density()` - Density-based filtering
- `add_spatial_noise()` - Data augmentation
- `get_neighbor_expression()` - Neighbor expression aggregation
- `synthetic_spatial()` - Synthetic spatial data generation
- `synthetic_scrna()` - Synthetic scRNA-seq data generation

#### Custom Data Fields
- `SpatialCoordinatesField` - Spatial coordinate registration
- `NeighborIndexField` - Neighbor index registration
- `NeighborDistanceField` - Neighbor distance registration
- `NicheCompositionField` - Niche composition registration

#### External Model Integrations
- `AMICI` - Cell-cell interaction inference via cross-attention
- `VIVS` - Variable importance via variance statistics
- `Starfysh` - Spatial deconvolution with histology integration
- `Harreman` - Metabolic exchange inference
- `Nolan` - Self-supervised spatial niche detection (NicheExplorer)
- `Lambda` - LLM-based cell type annotation
- `PPIInference` - Prediction-powered statistical inference
- `SPARL` - Spatial proteomics representation learning

#### scvi-tools Wrappers
- `scVIVA` - Cellular microenvironment modeling
- `ResolVI` - Spatial denoising for cell-resolved data
- `DestVI` - Multi-resolution spatial deconvolution

#### Utilities
- `spatial_autocorrelation()` - Moran's I and Geary's C computation
- `compute_morans_i()` - Moran's I statistic
- `silhouette_spatial()` - Spatial-aware silhouette score
- `plot_spatial()` - Spatial visualization
- `plot_proportions()` - Cell type proportion plots
- `plot_interactions()` - Interaction matrix visualization
- `plot_niche_composition()` - Niche composition plots

### Dependencies
- Python >= 3.10
- PyTorch >= 2.0
- scvi-tools >= 1.1.0
- AnnData >= 0.10
- Scanpy >= 1.9

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.1.0 | 2024-01-15 | Initial release |

[Unreleased]: https://github.com/YosefLab/spatialvi-tools/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/YosefLab/spatialvi-tools/releases/tag/v0.1.0
