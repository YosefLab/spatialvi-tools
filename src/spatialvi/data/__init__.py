"""Data loading and preprocessing utilities for spatial transcriptomics."""

from __future__ import annotations

from ._fields import (
    SpatialCoordinatesField,
    NeighborIndexField,
    NeighborDistanceField,
    NicheCompositionField,
)
from ._preprocessing import (
    compute_spatial_neighbors,
    compute_niche_composition,
    normalize_spatial,
    filter_by_spatial_density,
    add_spatial_noise,
    get_neighbor_expression,
)
from ._datasets import (
    synthetic_spatial,
    synthetic_scrna,
)

__all__ = [
    # Fields
    "SpatialCoordinatesField",
    "NeighborIndexField",
    "NeighborDistanceField",
    "NicheCompositionField",
    # Preprocessing
    "compute_spatial_neighbors",
    "compute_niche_composition",
    "normalize_spatial",
    "filter_by_spatial_density",
    "add_spatial_noise",
    "get_neighbor_expression",
    # Datasets
    "synthetic_spatial",
    "synthetic_scrna",
]
