"""Metrics for spatial analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def spatial_autocorrelation(
    adata: "AnnData",
    var_names: list[str] | None = None,
    spatial_key: str = "spatial",
    n_neighbors: int = 20,
    method: str = "moran",
    layer: str | None = None,
) -> dict[str, float]:
    """Compute spatial autocorrelation for genes.

    Parameters
    ----------
    adata
        AnnData object.
    var_names
        List of genes to compute. If None, computes for all.
    spatial_key
        Key in obsm for spatial coordinates.
    n_neighbors
        Number of neighbors for weight matrix.
    method
        Autocorrelation method: "moran" or "geary".
    layer
        Layer to use. If None, uses X.

    Returns
    -------
    Dictionary mapping gene names to autocorrelation values.
    """
    if var_names is None:
        var_names = adata.var_names.tolist()

    # Build spatial weight matrix
    W = _build_spatial_weights(adata, spatial_key, n_neighbors)

    # Get expression data
    if layer is not None:
        X = adata[:, var_names].layers[layer]
    else:
        X = adata[:, var_names].X

    if sparse.issparse(X):
        X = X.toarray()

    results = {}
    for i, gene in enumerate(var_names):
        x = X[:, i]
        if method == "moran":
            results[gene] = _compute_moran(x, W)
        elif method == "geary":
            results[gene] = _compute_geary(x, W)
        else:
            raise ValueError(f"Unknown method: {method}")

    return results


def compute_morans_i(
    values: "NDArray",
    weights: "NDArray",
) -> float:
    """Compute Moran's I statistic.

    Parameters
    ----------
    values
        Values to compute autocorrelation for.
    weights
        Spatial weight matrix.

    Returns
    -------
    Moran's I value.
    """
    return _compute_moran(values, weights)


def compute_gearys_c(
    values: "NDArray",
    weights: "NDArray",
) -> float:
    """Compute Geary's C statistic.

    Parameters
    ----------
    values
        Values to compute autocorrelation for.
    weights
        Spatial weight matrix.

    Returns
    -------
    Geary's C value.
    """
    return _compute_geary(values, weights)


def _build_spatial_weights(
    adata: "AnnData",
    spatial_key: str,
    n_neighbors: int,
) -> "NDArray":
    """Build spatial weight matrix."""
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    n_obs = coords.shape[0]

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Build weight matrix (binary)
    W = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        W[i, indices[i, 1:]] = 1

    # Row-normalize
    W = W / W.sum(axis=1, keepdims=True)

    return W


def _compute_moran(x: "NDArray", W: "NDArray") -> float:
    """Compute Moran's I."""
    n = len(x)
    x_dev = x - x.mean()
    num = n * np.sum(W * np.outer(x_dev, x_dev))
    denom = np.sum(W) * np.sum(x_dev ** 2)

    if denom == 0:
        return 0.0

    return num / denom


def _compute_geary(x: "NDArray", W: "NDArray") -> float:
    """Compute Geary's C."""
    n = len(x)
    x_dev = x - x.mean()

    # Pairwise squared differences
    diff_sq = np.subtract.outer(x, x) ** 2

    num = (n - 1) * np.sum(W * diff_sq)
    denom = 2 * np.sum(W) * np.sum(x_dev ** 2)

    if denom == 0:
        return 1.0

    return num / denom


def silhouette_spatial(
    adata: "AnnData",
    labels_key: str,
    spatial_key: str = "spatial",
) -> float:
    """Compute silhouette score using spatial coordinates.

    Parameters
    ----------
    adata
        AnnData object.
    labels_key
        Key in obs for cluster labels.
    spatial_key
        Key in obsm for spatial coordinates.

    Returns
    -------
    Silhouette score.
    """
    from sklearn.metrics import silhouette_score

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    labels = adata.obs[labels_key].values
    if hasattr(labels, "codes"):
        labels = labels.codes

    return silhouette_score(coords, labels)


def niche_purity(
    adata: "AnnData",
    labels_key: str,
    neighbor_key: str = "nn_index",
) -> float:
    """Compute niche purity (fraction of neighbors with same label).

    Parameters
    ----------
    adata
        AnnData object with neighbor indices.
    labels_key
        Key in obs for labels.
    neighbor_key
        Key in obsm for neighbor indices.

    Returns
    -------
    Average niche purity.
    """
    if neighbor_key not in adata.obsm:
        raise ValueError(f"Neighbor indices not found at obsm['{neighbor_key}']")

    labels = adata.obs[labels_key].values
    if hasattr(labels, "codes"):
        labels = labels.codes

    nn_indices = adata.obsm[neighbor_key]
    n_obs, n_neighbors = nn_indices.shape

    purities = []
    for i in range(n_obs):
        cell_label = labels[i]
        neighbor_labels = labels[nn_indices[i].astype(int)]
        purity = (neighbor_labels == cell_label).mean()
        purities.append(purity)

    return np.mean(purities)


def spatial_mixing_score(
    adata: "AnnData",
    batch_key: str,
    neighbor_key: str = "nn_index",
) -> float:
    """Compute spatial mixing score for batch effect assessment.

    Higher values indicate better mixing of batches in space.

    Parameters
    ----------
    adata
        AnnData object with neighbor indices.
    batch_key
        Key in obs for batch labels.
    neighbor_key
        Key in obsm for neighbor indices.

    Returns
    -------
    Spatial mixing score.
    """
    if neighbor_key not in adata.obsm:
        raise ValueError(f"Neighbor indices not found at obsm['{neighbor_key}']")

    batches = adata.obs[batch_key].values
    if hasattr(batches, "codes"):
        batches = batches.codes

    nn_indices = adata.obsm[neighbor_key]
    n_obs, n_neighbors = nn_indices.shape
    n_batches = len(np.unique(batches))

    # For each cell, count neighbors from different batches
    mixing_scores = []
    for i in range(n_obs):
        cell_batch = batches[i]
        neighbor_batches = batches[nn_indices[i].astype(int)]
        n_different = (neighbor_batches != cell_batch).sum()
        # Normalize by expected value under random mixing
        expected = n_neighbors * (n_batches - 1) / n_batches
        mixing_scores.append(n_different / expected if expected > 0 else 0)

    return np.mean(mixing_scores)
