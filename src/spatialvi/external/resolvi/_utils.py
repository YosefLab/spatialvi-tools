"""Utility functions for ResolVI spatial denoising."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_background_signal(
    adata: AnnData,
    spatial_key: str = "spatial",
    n_neighbors: int = 50,
    method: str = "local_median",
) -> NDArray:
    """Estimate local background signal levels.

    Parameters
    ----------
    adata
        AnnData with spatial coordinates.
    spatial_key
        Key for spatial coordinates in obsm.
    n_neighbors
        Number of neighbors for local estimation.
    method
        Estimation method ("local_median", "local_min", "percentile").

    Returns
    -------
    Background signal estimate per gene.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Find spatial neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    indices = indices[:, 1:]

    # Compute local background per cell
    n_cells, n_genes = X.shape
    background = np.zeros((n_cells, n_genes))

    for i in range(n_cells):
        neighbor_expr = X[indices[i]]
        if method == "local_median":
            background[i] = np.median(neighbor_expr, axis=0)
        elif method == "local_min":
            background[i] = np.min(neighbor_expr, axis=0)
        elif method == "percentile":
            background[i] = np.percentile(neighbor_expr, 10, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

    return background


def compute_segmentation_confidence(
    adata: AnnData,
    area_key: str | None = None,
    n_transcripts_key: str | None = None,
) -> NDArray:
    """Estimate segmentation confidence per cell.

    Parameters
    ----------
    adata
        AnnData with cell metadata.
    area_key
        Key for cell area in obs.
    n_transcripts_key
        Key for transcript count in obs.

    Returns
    -------
    Confidence scores (0-1) per cell.
    """
    scores = np.ones(adata.n_obs)

    # Penalize very small or large cells
    if area_key and area_key in adata.obs:
        areas = adata.obs[area_key].values.astype(float)
        median_area = np.median(areas)
        area_ratio = areas / median_area

        # Cells with extreme sizes get lower confidence
        area_score = np.exp(-np.abs(np.log(area_ratio + 1e-8)))
        scores *= area_score

    # Penalize cells with very few transcripts
    if n_transcripts_key and n_transcripts_key in adata.obs:
        n_trans = adata.obs[n_transcripts_key].values.astype(float)
        median_trans = np.median(n_trans)
        trans_score = np.minimum(n_trans / median_trans, 1.0)
        scores *= trans_score

    # Also use total counts from X matrix
    total_counts = np.asarray(adata.X.sum(axis=1)).flatten()
    median_counts = np.median(total_counts)
    count_score = np.minimum(total_counts / median_counts, 1.0)
    scores *= count_score

    return scores


def identify_contaminated_cells(
    adata: AnnData,
    background_fraction: NDArray,
    threshold: float = 0.5,
) -> NDArray:
    """Identify cells with high background contamination.

    Parameters
    ----------
    adata
        AnnData object.
    background_fraction
        Background fraction per cell from ResolVI.
    threshold
        Threshold for flagging contamination.

    Returns
    -------
    Boolean array indicating contaminated cells.
    """
    return background_fraction > threshold


def compute_signal_to_noise(
    raw_expression: NDArray,
    denoised_expression: NDArray,
) -> NDArray:
    """Compute signal-to-noise improvement from denoising.

    Parameters
    ----------
    raw_expression
        Raw expression matrix.
    denoised_expression
        Denoised expression matrix.

    Returns
    -------
    SNR improvement per gene.
    """
    raw_var = np.var(raw_expression, axis=0)
    noise_var = np.var(raw_expression - denoised_expression, axis=0)

    # SNR = signal variance / noise variance
    signal_var = np.maximum(raw_var - noise_var, 0)
    snr = np.where(noise_var > 0, signal_var / noise_var, np.inf)

    return snr


def filter_low_quality_cells(
    adata: AnnData,
    min_counts: int | None = None,
    min_genes: int | None = None,
    max_background: float | None = None,
    background_key: str = "background_fraction",
) -> NDArray:
    """Filter low quality cells based on multiple criteria.

    Parameters
    ----------
    adata
        AnnData object.
    min_counts
        Minimum total counts.
    min_genes
        Minimum number of expressed genes.
    max_background
        Maximum background fraction.
    background_key
        Key for background fraction in obs.

    Returns
    -------
    Boolean array of cells passing filters.
    """
    keep = np.ones(adata.n_obs, dtype=bool)

    # Count filters
    total_counts = np.asarray(adata.X.sum(axis=1)).flatten()
    if min_counts is not None:
        keep &= total_counts >= min_counts

    # Gene filters
    n_genes = np.asarray((adata.X > 0).sum(axis=1)).flatten()
    if min_genes is not None:
        keep &= n_genes >= min_genes

    # Background filter
    if max_background is not None and background_key in adata.obs:
        background = adata.obs[background_key].values
        keep &= background <= max_background

    return keep


def compute_spatial_smoothness(
    adata: AnnData,
    expression: NDArray,
    spatial_key: str = "spatial",
    n_neighbors: int = 10,
) -> float:
    """Compute spatial smoothness of expression.

    Parameters
    ----------
    adata
        AnnData with spatial coordinates.
    expression
        Expression matrix to evaluate.
    spatial_key
        Key for spatial coordinates.
    n_neighbors
        Number of neighbors.

    Returns
    -------
    Mean spatial smoothness score (lower is smoother).
    """
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    if hasattr(expression, "toarray"):
        expression = expression.toarray()

    # Find neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    indices = indices[:, 1:]

    # Compute local variance
    n_cells = len(coords)
    local_var = np.zeros(n_cells)

    for i in range(n_cells):
        neighbor_expr = expression[indices[i]]
        cell_expr = expression[i]
        local_var[i] = np.mean((neighbor_expr - cell_expr) ** 2)

    return float(np.mean(local_var))


def compare_denoising_quality(
    raw: NDArray,
    denoised: NDArray,
    adata: AnnData,
    spatial_key: str = "spatial",
) -> dict[str, float]:
    """Compare quality metrics before and after denoising.

    Parameters
    ----------
    raw
        Raw expression matrix.
    denoised
        Denoised expression matrix.
    adata
        AnnData with spatial coordinates.
    spatial_key
        Key for spatial coordinates.

    Returns
    -------
    Dictionary with quality metrics.
    """
    metrics = {}

    # Sparsity
    metrics["raw_sparsity"] = float(np.mean(raw == 0))
    metrics["denoised_sparsity"] = float(np.mean(denoised == 0))

    # Mean expression
    metrics["raw_mean"] = float(np.mean(raw))
    metrics["denoised_mean"] = float(np.mean(denoised))

    # Coefficient of variation
    raw_cv = np.std(raw, axis=0) / (np.mean(raw, axis=0) + 1e-8)
    denoised_cv = np.std(denoised, axis=0) / (np.mean(denoised, axis=0) + 1e-8)
    metrics["raw_mean_cv"] = float(np.mean(raw_cv))
    metrics["denoised_mean_cv"] = float(np.mean(denoised_cv))

    # Spatial smoothness
    metrics["raw_spatial_var"] = compute_spatial_smoothness(adata, raw, spatial_key)
    metrics["denoised_spatial_var"] = compute_spatial_smoothness(adata, denoised, spatial_key)

    return metrics
