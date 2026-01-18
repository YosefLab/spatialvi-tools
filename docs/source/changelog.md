# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] (Unreleased)

### Added
- Initial release of spatialvi-tools
- Core spatial VAE model (`SpatialVAE`) for spatial-aware dimensionality reduction
- External model integrations:
  - AMICI: Cross-attention cell-cell interaction model
  - VIVS: Variable importance via variance statistics
  - Starfysh: Spatial deconvolution with histology
  - Harreman: Metabolic exchange via spatial correlation
  - Nolan: Self-supervised spatial niche detection (NicheExplorer)
  - Lambda: LLM-based cell type annotation
  - PPIInference: Prediction-powered statistical inference
  - SPARL: Spatial proteomics representation learning
- scvi-tools integrations:
  - scVIVA: Cellular microenvironment modeling
  - ResolVI: Spatial denoising for cell-resolved data
  - DestVI: Spatial deconvolution
- Comprehensive data utilities for spatial preprocessing
- Spatial metrics and visualization functions
- Comprehensive test suite with 295+ tests

### Dependencies
- Requires Python >= 3.11
- Requires scvi-tools >= 1.1.0
- Requires PyTorch >= 2.0
