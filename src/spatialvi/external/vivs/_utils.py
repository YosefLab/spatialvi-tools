"""Utility functions for VIVS."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_multiscale_neighbors(
    adata: AnnData,
    spatial_key: str = "spatial",
    scales: list[int] | None = None,
    metric: str = "euclidean",
) -> list[NDArray]:
    """Compute spatial neighbors at multiple scales.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates.
    spatial_key
        Key in obsm for spatial coordinates.
    scales
        List of neighbor counts for each scale.
    metric
        Distance metric to use.

    Returns
    -------
    List of neighbor index arrays, one per scale.
    """
    from sklearn.neighbors import NearestNeighbors

    if scales is None:
        scales = [10, 20, 50, 100, 200]

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    neighbor_indices_list = []
    for k in scales:
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)
        neighbor_indices_list.append(indices[:, 1:].astype(np.int64))

    return neighbor_indices_list


def compute_fdr(p_values: NDArray, alpha: float = 0.05) -> tuple[NDArray, NDArray]:
    """Compute FDR-adjusted p-values using Benjamini-Hochberg.

    Parameters
    ----------
    p_values
        Array of p-values.
    alpha
        Significance level.

    Returns
    -------
    Tuple of (fdr_values, significant_mask).
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH procedure
    threshold = np.arange(1, n + 1) * alpha / n
    below_threshold = sorted_p <= threshold

    if np.any(below_threshold):
        max_idx = np.where(below_threshold)[0][-1]
        significant = np.zeros(n, dtype=bool)
        significant[sorted_idx[: max_idx + 1]] = True
    else:
        significant = np.zeros(n, dtype=bool)

    # Compute FDR-adjusted p-values
    fdr = np.zeros(n)
    fdr[sorted_idx] = np.minimum.accumulate(sorted_p * n / np.arange(1, n + 1)[::-1])[::-1]
    fdr = np.clip(fdr, 0, 1)

    return fdr, significant


def z_to_pvalue(z_scores: NDArray, two_sided: bool = True) -> NDArray:
    """Convert z-scores to p-values.

    Parameters
    ----------
    z_scores
        Array of z-scores.
    two_sided
        Whether to use two-sided test.

    Returns
    -------
    Array of p-values.
    """
    from scipy import stats

    if two_sided:
        return 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    else:
        return 1 - stats.norm.cdf(z_scores)


def rank_genes_by_spatial_variance(
    adata: AnnData,
    scores_key: str = "vivs_importance",
    n_top: int | None = None,
    ascending: bool = False,
) -> list[str]:
    """Rank genes by spatial variance scores.

    Parameters
    ----------
    adata
        AnnData object with VIVS scores in var.
    scores_key
        Key in var for scores.
    n_top
        Number of top genes to return.
    ascending
        Whether to sort ascending.

    Returns
    -------
    List of ranked gene names.
    """
    if scores_key not in adata.var:
        raise ValueError(f"Score key '{scores_key}' not found in adata.var")

    scores = adata.var[scores_key].values
    sorted_idx = np.argsort(scores)

    if not ascending:
        sorted_idx = sorted_idx[::-1]

    ranked_genes = adata.var_names[sorted_idx].tolist()

    if n_top is not None:
        ranked_genes = ranked_genes[:n_top]

    return ranked_genes
