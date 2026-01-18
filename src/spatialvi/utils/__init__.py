"""Utility functions for spatialvi-tools."""

from __future__ import annotations

from ._metrics import (
    compute_gearys_c,
    compute_morans_i,
    silhouette_spatial,
    spatial_autocorrelation,
)
from ._visualization import (
    plot_interactions,
    plot_niche_composition,
    plot_proportions,
    plot_spatial,
)

__all__ = [
    # Metrics
    "spatial_autocorrelation",
    "compute_morans_i",
    "compute_gearys_c",
    "silhouette_spatial",
    # Visualization
    "plot_spatial",
    "plot_proportions",
    "plot_interactions",
    "plot_niche_composition",
]
