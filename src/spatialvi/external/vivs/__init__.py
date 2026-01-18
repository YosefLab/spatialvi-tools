"""VIVS: Variable Importance via Variance Statistics.

VIVS provides variable selection for spatial transcriptomics data
by identifying genes with significant spatial patterns.
"""

from __future__ import annotations

from ._model import VIVS
from ._module import MultiScaleVIVS, VIVSModule
from ._utils import (
    compute_fdr,
    compute_multiscale_neighbors,
    rank_genes_by_spatial_variance,
    z_to_pvalue,
)

__all__ = [
    "VIVS",
    "MultiScaleVIVS",
    "VIVSModule",
    "compute_fdr",
    "compute_multiscale_neighbors",
    "rank_genes_by_spatial_variance",
    "z_to_pvalue",
]
