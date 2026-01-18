"""Nolan (NicheExplorer) model for spatial niche detection.

NOLAN (NO Label ANalysis) identifies spatial niches using
self-supervised learning without requiring cell type annotations.
"""

from __future__ import annotations

from ._model import Nolan
from ._module import NicheClusteringHead, NolanModule
from ._utils import (
    compute_grid_size,
    create_niche_graph,
    evaluate_niche_clustering,
    sample_spatial_crops,
)

__all__ = [
    "Nolan",
    "NicheClusteringHead",
    "NolanModule",
    "compute_grid_size",
    "create_niche_graph",
    "evaluate_niche_clustering",
    "sample_spatial_crops",
]
