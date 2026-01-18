"""DestVI wrapper for spatial deconvolution.

DestVI performs multi-resolution spatial deconvolution to estimate
cell type proportions and sub-cell-type variation in spatial spots.

This module provides both wrapper access to the scvi-tools implementation
and the underlying module classes for advanced usage.
"""

from __future__ import annotations

from ._model import CondSCVI, DestVI
from ._module import DestVIModule, MRDeconv
from ._utils import (
    compute_cell_type_abundance,
    compute_colocalization,
    compute_niche_enrichment,
    compute_spatial_autocorrelation,
    identify_dominant_cell_type,
    validate_reference_overlap,
)

__all__ = [
    # Model wrappers
    "CondSCVI",
    "DestVI",
    # Module classes
    "MRDeconv",
    "DestVIModule",
    # Utility functions
    "compute_cell_type_abundance",
    "compute_colocalization",
    "compute_niche_enrichment",
    "compute_spatial_autocorrelation",
    "identify_dominant_cell_type",
    "validate_reference_overlap",
]
