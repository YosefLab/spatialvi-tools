"""Utility functions for Nolan niche detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_grid_size(
    adata: AnnData,
    spatial_key: str = "spatial",
    expected_num_cells: int = 50,
    batch_key: str | None = None,
) -> tuple[float, float, int]:
    """Compute optimal grid size for NOLAN crops.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates.
    spatial_key
        Key in obsm for spatial coordinates.
    expected_num_cells
        Target number of cells per crop.
    batch_key
        Optional batch key for per-batch computation.

    Returns
    -------
    Tuple of (crop_radius, mean_cell_count, max_num_cells).
    """
    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    if batch_key is not None and batch_key in adata.obs:
        # Compute per batch
        batches = adata.obs[batch_key].unique()
        radii = []
        counts = []

        for batch in batches:
            mask = adata.obs[batch_key] == batch
            batch_coords = coords[mask]

            radius, mean_count, _ = _compute_single_grid_size(batch_coords, expected_num_cells)
            radii.append(radius)
            counts.append(mean_count)

        return np.mean(radii), np.mean(counts), int(np.max(counts) * 1.5)
    else:
        return _compute_single_grid_size(coords, expected_num_cells)


def _compute_single_grid_size(
    coords: NDArray,
    expected_num_cells: int,
) -> tuple[float, float, int]:
    """Compute grid size for single coordinate set."""
    from sklearn.neighbors import BallTree

    n_cells = coords.shape[0]

    # Estimate density
    area = (coords[:, 0].max() - coords[:, 0].min()) * (coords[:, 1].max() - coords[:, 1].min())
    density = n_cells / area

    # Target radius for expected cell count
    # Area = pi * r^2, cells = density * area
    target_area = expected_num_cells / density
    radius = np.sqrt(target_area / np.pi)

    # Verify with actual neighbor counting
    tree = BallTree(coords)
    counts = tree.query_radius(coords, r=radius, count_only=True)
    mean_count = np.mean(counts)

    # Adjust radius if needed
    if mean_count < expected_num_cells * 0.8:
        radius *= 1.2
    elif mean_count > expected_num_cells * 1.5:
        radius *= 0.8

    # Recompute counts
    counts = tree.query_radius(coords, r=radius, count_only=True)
    mean_count = np.mean(counts)
    max_count = np.max(counts)

    return radius, mean_count, int(max_count * 1.2)


def sample_spatial_crops(
    coords: NDArray,
    crop_radius: float,
    n_crops: int,
    seed: int | None = None,
) -> list[NDArray]:
    """Sample spatial crops for training.

    Parameters
    ----------
    coords
        Spatial coordinates of shape (n_cells, 2).
    crop_radius
        Radius of each crop.
    n_crops
        Number of crops to sample.
    seed
        Random seed.

    Returns
    -------
    List of cell index arrays for each crop.
    """
    from sklearn.neighbors import BallTree

    if seed is not None:
        np.random.seed(seed)

    n_cells = coords.shape[0]
    tree = BallTree(coords)

    crops = []
    for _ in range(n_crops):
        # Sample random center
        center_idx = np.random.randint(0, n_cells)
        center = coords[center_idx : center_idx + 1]

        # Get cells within radius
        indices = tree.query_radius(center, r=crop_radius)[0]
        crops.append(indices)

    return crops


def create_niche_graph(
    adata: AnnData,
    niche_key: str = "niche_cluster",
    spatial_key: str = "spatial",
    min_edge_weight: float = 0.1,
) -> dict:
    """Create niche adjacency graph.

    Parameters
    ----------
    adata
        AnnData with niche assignments.
    niche_key
        Key in obs for niche cluster assignments.
    spatial_key
        Key in obsm for spatial coordinates.
    min_edge_weight
        Minimum edge weight to include.

    Returns
    -------
    Dictionary with nodes and edges.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    niches = adata.obs[niche_key].values
    unique_niches = np.unique(niches)
    n_niches = len(unique_niches)

    # Compute neighbor relationships
    nn = NearestNeighbors(n_neighbors=20)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)

    # Count niche-niche adjacencies
    adjacency = np.zeros((n_niches, n_niches))

    for i, niche in enumerate(niches):
        niche_idx = np.where(unique_niches == niche)[0][0]
        neighbor_niches = niches[indices[i, 1:]]
        for neighbor_niche in neighbor_niches:
            neighbor_idx = np.where(unique_niches == neighbor_niche)[0][0]
            adjacency[niche_idx, neighbor_idx] += 1

    # Normalize
    adjacency = adjacency / adjacency.sum()

    # Create graph
    nodes = [{"id": str(n), "size": np.sum(niches == n)} for n in unique_niches]

    edges = []
    for i in range(n_niches):
        for j in range(i + 1, n_niches):
            weight = adjacency[i, j] + adjacency[j, i]
            if weight >= min_edge_weight:
                edges.append(
                    {
                        "source": str(unique_niches[i]),
                        "target": str(unique_niches[j]),
                        "weight": float(weight),
                    }
                )

    return {"nodes": nodes, "edges": edges}


def evaluate_niche_clustering(
    adata: AnnData,
    niche_key: str = "niche_cluster",
    labels_key: str | None = None,
    spatial_key: str = "spatial",
) -> dict[str, float]:
    """Evaluate niche clustering quality.

    Parameters
    ----------
    adata
        AnnData with niche assignments.
    niche_key
        Key for niche clusters.
    labels_key
        Optional ground truth cell type labels.
    spatial_key
        Key for spatial coordinates.

    Returns
    -------
    Dictionary of metrics.
    """
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    niches = adata.obs[niche_key].values

    metrics = {}

    # Silhouette score on spatial coordinates
    if len(np.unique(niches)) > 1:
        metrics["spatial_silhouette"] = silhouette_score(coords, niches)

    # Comparison with cell type labels if available
    if labels_key is not None and labels_key in adata.obs:
        labels = adata.obs[labels_key].values
        if hasattr(labels, "codes"):
            labels = labels.codes

        metrics["ari_with_labels"] = adjusted_rand_score(labels, niches)
        metrics["nmi_with_labels"] = normalized_mutual_info_score(labels, niches)

    return metrics
