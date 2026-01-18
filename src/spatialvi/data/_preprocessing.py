"""Preprocessing utilities for spatial transcriptomics data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse import issparse

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_spatial_neighbors(
    adata: AnnData,
    spatial_key: str = "spatial",
    n_neighbors: int = 20,
    index_key: str = "nn_index",
    dist_key: str = "nn_dist",
    metric: Literal["euclidean", "manhattan", "cosine"] = "euclidean",
    copy: bool = False,
) -> AnnData | None:
    """Compute spatial nearest neighbors.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates.
    spatial_key
        Key in obsm for spatial coordinates.
    n_neighbors
        Number of neighbors to compute.
    index_key
        Key to store neighbor indices in obsm.
    dist_key
        Key to store neighbor distances in obsm.
    metric
        Distance metric to use.
    copy
        Whether to return a copy of the AnnData.

    Returns
    -------
    AnnData with neighbor information if copy=True, else None.
    """
    from sklearn.neighbors import NearestNeighbors

    adata = adata.copy() if copy else adata

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    logger.info(f"Computing {n_neighbors} spatial neighbors using {metric} metric")

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=metric)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Remove self from neighbors
    adata.obsm[index_key] = indices[:, 1:].astype(np.int64)
    adata.obsm[dist_key] = distances[:, 1:].astype(np.float32)

    # Store metadata
    if "spatial" not in adata.uns:
        adata.uns["spatial"] = {}
    adata.uns["spatial"]["n_neighbors"] = n_neighbors
    adata.uns["spatial"]["neighbor_metric"] = metric

    logger.info(f"Stored neighbor indices in obsm['{index_key}']")
    logger.info(f"Stored neighbor distances in obsm['{dist_key}']")

    if copy:
        return adata


def compute_niche_composition(
    adata: AnnData,
    labels_key: str,
    index_key: str = "nn_index",
    composition_key: str = "niche_composition",
    copy: bool = False,
) -> AnnData | None:
    """Compute neighborhood cell type composition.

    Parameters
    ----------
    adata
        AnnData object with neighbor indices.
    labels_key
        Key in obs for cell type labels.
    index_key
        Key in obsm for neighbor indices.
    composition_key
        Key to store composition in obsm.
    copy
        Whether to return a copy of the AnnData.

    Returns
    -------
    AnnData with composition if copy=True, else None.
    """
    adata = adata.copy() if copy else adata

    if index_key not in adata.obsm:
        raise ValueError(f"Neighbor indices not found at obsm['{index_key}']. Run compute_spatial_neighbors first.")

    labels = adata.obs[labels_key].values
    if hasattr(labels, "cat"):
        categories = labels.cat.categories
        labels = labels.cat.codes.values
        n_labels = len(categories)
    else:
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])

    nn_indices = adata.obsm[index_key]
    n_obs, n_neighbors = nn_indices.shape

    logger.info(f"Computing niche composition for {n_labels} cell types")

    # Compute composition
    composition = np.zeros((n_obs, n_labels), dtype=np.float32)
    for i in range(n_obs):
        neighbor_labels = labels[nn_indices[i]]
        for label in neighbor_labels:
            composition[i, label] += 1
    composition /= n_neighbors

    adata.obsm[composition_key] = composition

    logger.info(f"Stored niche composition in obsm['{composition_key}']")

    if copy:
        return adata


def normalize_spatial(
    adata: AnnData,
    spatial_key: str = "spatial",
    method: Literal["minmax", "zscore", "center"] = "minmax",
    copy: bool = False,
) -> AnnData | None:
    """Normalize spatial coordinates.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates.
    spatial_key
        Key in obsm for spatial coordinates.
    method
        Normalization method:
        - "minmax": Scale to [0, 1]
        - "zscore": Zero mean, unit variance
        - "center": Center around zero
    copy
        Whether to return a copy of the AnnData.

    Returns
    -------
    AnnData with normalized coordinates if copy=True, else None.
    """
    adata = adata.copy() if copy else adata

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values.copy()
    else:
        coords = coords.copy()

    if method == "minmax":
        coords_min = coords.min(axis=0, keepdims=True)
        coords_max = coords.max(axis=0, keepdims=True)
        coords = (coords - coords_min) / (coords_max - coords_min + 1e-8)
    elif method == "zscore":
        coords_mean = coords.mean(axis=0, keepdims=True)
        coords_std = coords.std(axis=0, keepdims=True)
        coords = (coords - coords_mean) / (coords_std + 1e-8)
    elif method == "center":
        coords_mean = coords.mean(axis=0, keepdims=True)
        coords = coords - coords_mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    adata.obsm[spatial_key] = coords.astype(np.float32)

    logger.info(f"Normalized spatial coordinates using '{method}' method")

    if copy:
        return adata


def filter_by_spatial_density(
    adata: AnnData,
    spatial_key: str = "spatial",
    min_density: float | None = None,
    max_density: float | None = None,
    n_neighbors: int = 10,
    copy: bool = False,
) -> AnnData | None:
    """Filter cells by local spatial density.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates.
    spatial_key
        Key in obsm for spatial coordinates.
    min_density
        Minimum density threshold (cells per unit area).
    max_density
        Maximum density threshold.
    n_neighbors
        Number of neighbors for density estimation.
    copy
        Whether to return a copy of the AnnData.

    Returns
    -------
    Filtered AnnData if copy=True, else None.
    """
    from sklearn.neighbors import NearestNeighbors

    adata = adata.copy() if copy else adata

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    # Compute local density using k-nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords)

    # Density is inversely proportional to average neighbor distance
    avg_dist = distances[:, 1:].mean(axis=1)
    density = 1.0 / (avg_dist + 1e-8)

    # Normalize density
    density = density / density.max()

    # Filter
    mask = np.ones(adata.n_obs, dtype=bool)
    if min_density is not None:
        mask &= density >= min_density
    if max_density is not None:
        mask &= density <= max_density

    n_filtered = (~mask).sum()
    logger.info(f"Filtering {n_filtered} cells by spatial density")

    adata._inplace_subset_obs(mask)

    # Store density
    adata.obs["spatial_density"] = density[mask]

    if copy:
        return adata


def add_spatial_noise(
    adata: AnnData,
    spatial_key: str = "spatial",
    noise_scale: float = 0.01,
    seed: int | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Add small noise to spatial coordinates.

    Useful for breaking ties in coordinates that are on a regular grid.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates.
    spatial_key
        Key in obsm for spatial coordinates.
    noise_scale
        Scale of noise relative to coordinate range.
    seed
        Random seed for reproducibility.
    copy
        Whether to return a copy of the AnnData.

    Returns
    -------
    AnnData with noisy coordinates if copy=True, else None.
    """
    adata = adata.copy() if copy else adata

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values.copy()
    else:
        coords = coords.copy()

    rng = np.random.default_rng(seed)

    # Compute coordinate range
    coord_range = coords.max(axis=0) - coords.min(axis=0)

    # Add noise
    noise = rng.normal(0, noise_scale * coord_range, size=coords.shape)
    coords = coords + noise

    adata.obsm[spatial_key] = coords.astype(np.float32)

    logger.info(f"Added spatial noise with scale {noise_scale}")

    if copy:
        return adata


def get_neighbor_expression(
    adata: AnnData,
    index_key: str = "nn_index",
    layer: str | None = None,
    aggregation: Literal["mean", "sum", "max"] = "mean",
) -> NDArray:
    """Get aggregated expression of neighboring cells.

    Parameters
    ----------
    adata
        AnnData object with neighbor indices.
    index_key
        Key in obsm for neighbor indices.
    layer
        Layer to use for expression. If None, uses X.
    aggregation
        Aggregation method.

    Returns
    -------
    Aggregated neighbor expression array of shape (n_cells, n_genes).
    """
    if index_key not in adata.obsm:
        raise ValueError(f"Neighbor indices not found at obsm['{index_key}']. Run compute_spatial_neighbors first.")

    nn_indices = adata.obsm[index_key]
    n_obs, n_neighbors = nn_indices.shape

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if issparse(X):
        X = X.toarray()

    # Get neighbor expression
    neighbor_expr = X[nn_indices.astype(int)]  # (n_obs, n_neighbors, n_genes)

    # Aggregate
    if aggregation == "mean":
        result = neighbor_expr.mean(axis=1)
    elif aggregation == "sum":
        result = neighbor_expr.sum(axis=1)
    elif aggregation == "max":
        result = neighbor_expr.max(axis=1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return result
