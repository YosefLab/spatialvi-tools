"""scVIVA wrapper for cellular microenvironment modeling.

scVIVA (Single-Cell resolved spatial Variational Inference with niche analysis)
models cellular microenvironments and spatial neighborhoods.

This module provides both wrapper access to the scvi-tools implementation
and the underlying module classes for advanced usage.
"""

from __future__ import annotations

from ._components import DirichletDecoder, Encoder, NicheDecoder
from ._constants import SCVIVA_MODULE_KEYS, SCVIVA_REGISTRY_KEYS
from ._model import scVIVA
from ._module import NicheLossOutput, nicheVAE
from ._utils import (
    compute_niche_composition,
    compute_niche_differential_genes,
    compute_niche_heterogeneity,
    compute_niche_interaction_strength,
    compute_spatial_entropy,
    identify_boundary_cells,
    identify_niche_clusters,
    visualize_niche_embedding,
)

__all__ = [
    # Model wrapper
    "scVIVA",
    # Module classes
    "nicheVAE",
    "NicheLossOutput",
    # Components
    "Encoder",
    "DirichletDecoder",
    "NicheDecoder",
    # Constants
    "SCVIVA_MODULE_KEYS",
    "SCVIVA_REGISTRY_KEYS",
    # Utility functions
    "compute_niche_composition",
    "compute_niche_differential_genes",
    "compute_niche_heterogeneity",
    "compute_niche_interaction_strength",
    "compute_spatial_entropy",
    "identify_boundary_cells",
    "identify_niche_clusters",
    "visualize_niche_embedding",
]
