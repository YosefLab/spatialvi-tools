# Tutorials

This section contains tutorials for spatial transcriptomics analysis using spatialvi-tools.

## Spatial Denoising and Correction

### ResolVI Tutorial
Learn how to use ResolVI for noise correction in cellular-resolved spatial transcriptomics data.
ResolVI addresses erroneous co-expression patterns after cellular segmentation and unspecific background noise.

```{toctree}
:maxdepth: 1

resolvi_tutorial
```

## Spatial Deconvolution

### DestVI Tutorial
Multi-resolution deconvolution of spatial transcriptomics using DestVI.
Learn how to estimate cell type proportions and intra-cell type gene expression from 10x Visium data.

```{toctree}
:maxdepth: 1

destvi_tutorial
```

### Starfysh Tutorial
Spatial deconvolution with histology integration using Starfysh.
Learn how to deconvolve spatial spots using archetypal factors per cell type.

```{toctree}
:maxdepth: 1

starfysh_tutorial
```

## Cellular Microenvironments

### scVIVA Tutorial
Learn how to use scVIVA for modeling cellular microenvironments and performing niche-aware differential expression.

```{toctree}
:maxdepth: 1

scviva_tutorial
```

### Nolan (NicheExplorer) Tutorial
Self-supervised spatial niche detection using Nolan/NicheExplorer.
Learn to identify spatial tissue domains without requiring cell type annotations.

```{toctree}
:maxdepth: 1

nolan_tutorial
```

## Cell-Cell Interactions

### AMICI Tutorial
Cell-cell interaction inference using attention-based cross-attention mechanisms.
Learn how to model how neighboring cells influence gene expression.

```{toctree}
:maxdepth: 1

amici_tutorial
```

### Harreman Tutorial
Metabolic exchange inference between spatially proximal cells.
Analyze correlations between metabolic gene expression and spatial proximity.

```{toctree}
:maxdepth: 1

harreman_tutorial
```

## Spatial Feature Detection

### VIVS Tutorial
Variable Importance via Variance Statistics for identifying spatially variable genes.
Compare observed and expected variance at different spatial scales.

```{toctree}
:maxdepth: 1

vivs_tutorial
```

## Cell Type Annotation

### Lambda Tutorial
LLM-based automatic cell type annotation without reference datasets.
Use large language models to annotate cell clusters based on marker genes.

```{toctree}
:maxdepth: 1

lambda_tutorial
```

## Spatial Proteomics

### SPARL Tutorial
Spatial proteomics representation learning for IMC/CODEX data.
Learn spatial-aware representations from protein expression measurements.

```{toctree}
:maxdepth: 1

sparl_tutorial
```

## Statistical Inference

### PPI Tutorial
Prediction-powered inference for spatial transcriptomics.
Leverage ML predictions to improve statistical inference with limited labels.

```{toctree}
:maxdepth: 1

ppi_tutorial
```

## Additional Resources

For more information about the models, please refer to the [API Reference](../source/api/index.rst).

## Getting Started

Before running these tutorials, make sure you have installed spatialvi-tools:

```bash
pip install spatialvi-tools
```

Most tutorials require GPU acceleration for reasonable training times. We recommend using Google Colab or a local machine with a CUDA-compatible GPU.

### External Package Dependencies

Some external models require additional packages:

```bash
# For Nolan/NicheExplorer
pip install nolan

# For Lambda (LLM annotation)
pip install LAMBDA

# For SPARL
pip install sparl

# For PPI inference
pip install ppi-python
```
