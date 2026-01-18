"""Utility functions for Starfysh."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_reference_signatures(
    adata_ref: AnnData,
    cell_type_key: str,
    genes: list[str] | None = None,
    normalize: bool = True,
) -> tuple[NDArray, list[str]]:
    """Compute reference expression signatures per cell type.

    Parameters
    ----------
    adata_ref
        Reference AnnData with cell type annotations.
    cell_type_key
        Key in obs for cell type labels.
    genes
        Genes to use. If None, uses all.
    normalize
        Whether to normalize signatures.

    Returns
    -------
    Tuple of (signatures array, cell type names).
    """
    if genes is not None:
        adata_ref = adata_ref[:, genes]

    X = adata_ref.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    labels = adata_ref.obs[cell_type_key]
    cell_types = labels.cat.categories.tolist() if hasattr(labels.cat, "categories") else np.unique(labels).tolist()
    n_cell_types = len(cell_types)
    n_genes = X.shape[1]

    signatures = np.zeros((n_cell_types, n_genes))
    for i, ct in enumerate(cell_types):
        mask = labels == ct
        signatures[i] = X[mask].mean(axis=0)

    if normalize:
        signatures = signatures / (signatures.sum(axis=1, keepdims=True) + 1e-8)

    return signatures, cell_types


def find_marker_genes(
    adata_ref: AnnData,
    cell_type_key: str,
    n_markers: int = 50,
    method: str = "wilcoxon",
) -> dict[str, list[str]]:
    """Find marker genes for each cell type.

    Parameters
    ----------
    adata_ref
        Reference AnnData.
    cell_type_key
        Key for cell type labels.
    n_markers
        Number of marker genes per cell type.
    method
        Method for marker detection.

    Returns
    -------
    Dictionary mapping cell types to marker gene lists.
    """
    import scanpy as sc

    adata = adata_ref.copy()
    sc.tl.rank_genes_groups(adata, groupby=cell_type_key, method=method)

    markers = {}
    for ct in adata.obs[cell_type_key].unique():
        markers[ct] = adata.uns["rank_genes_groups"]["names"][ct][:n_markers].tolist()

    return markers


def validate_input_data(
    adata_spatial: AnnData,
    adata_ref: AnnData,
    cell_type_key: str,
) -> tuple[list[str], list[str]]:
    """Validate input data for Starfysh.

    Parameters
    ----------
    adata_spatial
        Spatial AnnData.
    adata_ref
        Reference AnnData.
    cell_type_key
        Cell type key in reference.

    Returns
    -------
    Tuple of (common_genes, cell_types).

    Raises
    ------
    ValueError
        If validation fails.
    """
    # Check cell type key exists
    if cell_type_key not in adata_ref.obs:
        raise ValueError(f"Cell type key '{cell_type_key}' not found in reference obs")

    # Find common genes
    common_genes = adata_spatial.var_names.intersection(adata_ref.var_names).tolist()

    if len(common_genes) == 0:
        raise ValueError("No common genes found between spatial and reference data")

    if len(common_genes) < 50:
        logger.warning(f"Only {len(common_genes)} common genes found. Consider using more genes.")

    # Get cell types
    labels = adata_ref.obs[cell_type_key]
    if hasattr(labels.cat, "categories"):
        cell_types = labels.cat.categories.tolist()
    else:
        cell_types = np.unique(labels).tolist()

    logger.info(f"Found {len(common_genes)} common genes and {len(cell_types)} cell types")

    return common_genes, cell_types


def proportions_to_counts(
    proportions: NDArray,
    total_counts: NDArray | int,
) -> NDArray:
    """Convert proportions to estimated cell counts.

    Parameters
    ----------
    proportions
        Proportion array of shape (n_spots, n_cell_types).
    total_counts
        Total cell counts per spot (scalar or array).

    Returns
    -------
    Estimated cell counts.
    """
    if isinstance(total_counts, (int, float)):
        total_counts = np.full(proportions.shape[0], total_counts)

    counts = proportions * total_counts[:, np.newaxis]
    return np.round(counts).astype(int)


def evaluate_deconvolution(
    predicted: NDArray,
    true_proportions: NDArray,
) -> dict[str, float]:
    """Evaluate deconvolution accuracy.

    Parameters
    ----------
    predicted
        Predicted proportions.
    true_proportions
        Ground truth proportions.

    Returns
    -------
    Dictionary of metrics (RMSE, MAE, correlation).
    """
    from scipy import stats

    # RMSE
    rmse = np.sqrt(np.mean((predicted - true_proportions) ** 2))

    # MAE
    mae = np.mean(np.abs(predicted - true_proportions))

    # Pearson correlation per cell type
    n_cell_types = predicted.shape[1]
    correlations = []
    for i in range(n_cell_types):
        r, _ = stats.pearsonr(predicted[:, i], true_proportions[:, i])
        correlations.append(r)

    mean_correlation = np.nanmean(correlations)

    # Cosine similarity
    cos_sim = np.mean(
        np.sum(predicted * true_proportions, axis=1)
        / (np.linalg.norm(predicted, axis=1) * np.linalg.norm(true_proportions, axis=1) + 1e-8)
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "mean_correlation": mean_correlation,
        "cosine_similarity": cos_sim,
    }
