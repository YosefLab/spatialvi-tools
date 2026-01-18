"""Data loading and preprocessing utilities for spatial transcriptomics."""

from __future__ import annotations

from ._datasets import (
    synthetic_scrna,
    synthetic_spatial,
)
from ._fields import (
    NeighborDistanceField,
    NeighborIndexField,
    NicheCompositionField,
    SpatialCoordinatesField,
)
from ._preprocessing import (
    add_spatial_noise,
    compute_niche_composition,
    compute_spatial_neighbors,
    filter_by_spatial_density,
    get_neighbor_expression,
    normalize_spatial,
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
