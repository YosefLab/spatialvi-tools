"""External model integrations for spatial transcriptomics.

This module provides unified interfaces to external spatial
transcriptomics methods and integrations with scvi-tools
spatial models.

Models included:
- AMICI: Cross-attention cell-cell interaction model
- VIVS: Variable importance via variance statistics
- Starfysh: Spatial deconvolution with histology
- Harreman: Metabolic exchange via spatial correlation
- Nolan: Self-supervised spatial niche detection (NicheExplorer)
- Lambda: LLM-based cell type annotation
- PPIInference: Prediction-powered statistical inference
- SPARL: Spatial proteomics representation learning

scvi-tools integrations:
- scVIVA: Cellular microenvironment modeling
- ResolVI: Spatial denoising for cell-resolved data
- DestVI: Spatial deconvolution
"""

from __future__ import annotations

# AMICI - Cell-cell interaction
from spatialvi.external.amici import AMICI
from spatialvi.external.destvi import DestVI

# Harreman - Metabolic exchange
from spatialvi.external.harreman import Harreman

# Lambda - LLM-based annotation
from spatialvi.external.lambda_model import Lambda

# Nolan - Spatial niche detection (NicheExplorer)
from spatialvi.external.nolan import Nolan

# PPI - Prediction-powered inference
from spatialvi.external.ppi import PPIInference
from spatialvi.external.resolvi import ResolVI

# scvi-tools integrations (wrappers for consistency)
from spatialvi.external.scviva import scVIVA

# SPARL - Spatial proteomics
from spatialvi.external.sparl import SPARL

# Starfysh - Spatial deconvolution with histology
from spatialvi.external.starfysh import Starfysh

# VIVS - Variable importance
from spatialvi.external.vivs import VIVS

__all__ = [
    # External models
    "AMICI",
    "VIVS",
    "Starfysh",
    "Harreman",
    "Nolan",
    "Lambda",
    "PPIInference",
    "SPARL",
    # scvi-tools integrations
    "scVIVA",
    "ResolVI",
    "DestVI",
]
