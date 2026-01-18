"""Utility functions for spatialvi-tools."""

from __future__ import annotations

from ._metrics import (
    spatial_autocorrelation,
    compute_morans_i,
    compute_gearys_c,
    silhouette_spatial,
)
from ._visualization import (
    plot_spatial,
    plot_proportions,
    plot_interactions,
    plot_niche_composition,
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
