"""Utility functions for DestVI spatial deconvolution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray
    import pandas as pd

logger = logging.getLogger(__name__)


def compute_cell_type_abundance(
    proportions: "NDArray" | "pd.DataFrame",
    normalize: bool = True,
) -> "NDArray":
    """Compute total cell type abundance across spatial spots.

    Parameters
    ----------
    proportions
        Cell type proportion matrix (spots x cell_types).
    normalize
        Whether to normalize to sum to 1.

    Returns
    -------
    Total abundance per cell type.
    """
    if hasattr(proportions, "values"):
        proportions = proportions.values

    abundance = proportions.sum(axis=0)

    if normalize:
        abundance = abundance / abundance.sum()

    return abundance


def compute_spatial_autocorrelation(
    adata: "AnnData",
    proportions: "NDArray" | "pd.DataFrame",
    spatial_key: str = "spatial",
    n_neighbors: int = 6,
) -> dict[str, float]:
    """Compute Moran's I for cell type proportions.

    Parameters
    ----------
    adata
        AnnData with spatial coordinates.
    proportions
        Cell type proportion matrix.
    spatial_key
        Key for spatial coordinates in obsm.
    n_neighbors
        Number of neighbors for spatial weights.

    Returns
    -------
    Dictionary mapping cell types to Moran's I values.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    if hasattr(proportions, "values"):
        prop_array = proportions.values
        cell_types = proportions.columns.tolist()
    else:
        prop_array = proportions
        cell_types = [f"CellType_{i}" for i in range(proportions.shape[1])]

    # Build spatial weights
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    indices = indices[:, 1:]  # Exclude self

    n_spots = len(coords)
    W = np.zeros((n_spots, n_spots))
    for i in range(n_spots):
        W[i, indices[i]] = 1.0 / n_neighbors

    # Compute Moran's I for each cell type
    morans_i = {}
    for j, ct in enumerate(cell_types):
        x = prop_array[:, j]
        x_centered = x - x.mean()
        n = len(x)

        numerator = n * np.sum(W * np.outer(x_centered, x_centered))
        denominator = np.sum(W) * np.sum(x_centered**2)

        if denominator > 0:
            morans_i[ct] = numerator / denominator
        else:
            morans_i[ct] = 0.0

    return morans_i


def identify_dominant_cell_type(
    proportions: "NDArray" | "pd.DataFrame",
    threshold: float = 0.3,
) -> "NDArray":
    """Identify dominant cell type per spot.

    Parameters
    ----------
    proportions
        Cell type proportion matrix.
    threshold
        Minimum proportion to be considered dominant.

    Returns
    -------
    Array of dominant cell type indices (-1 if none dominant).
    """
    if hasattr(proportions, "values"):
        prop_array = proportions.values
    else:
        prop_array = proportions

    max_props = prop_array.max(axis=1)
    dominant = prop_array.argmax(axis=1)

    # Set to -1 where no cell type is dominant
    dominant[max_props < threshold] = -1

    return dominant


def compute_colocalization(
    proportions: "NDArray" | "pd.DataFrame",
    method: str = "pearson",
) -> "NDArray":
    """Compute cell type colocalization matrix.

    Parameters
    ----------
    proportions
        Cell type proportion matrix.
    method
        Correlation method.

    Returns
    -------
    Colocalization matrix (cell_types x cell_types).
    """
    if hasattr(proportions, "values"):
        prop_array = proportions.values
    else:
        prop_array = proportions

    if method == "pearson":
        return np.corrcoef(prop_array.T)
    elif method == "spearman":
        from scipy import stats

        n_types = prop_array.shape[1]
        coloc = np.zeros((n_types, n_types))
        for i in range(n_types):
            for j in range(i, n_types):
                r, _ = stats.spearmanr(prop_array[:, i], prop_array[:, j])
                coloc[i, j] = r
                coloc[j, i] = r
        return coloc
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_niche_enrichment(
    adata: "AnnData",
    proportions: "NDArray" | "pd.DataFrame",
    region_key: str,
) -> "pd.DataFrame":
    """Compute cell type enrichment per spatial region.

    Parameters
    ----------
    adata
        AnnData with region annotations.
    proportions
        Cell type proportion matrix.
    region_key
        Key for region labels in obs.

    Returns
    -------
    DataFrame with enrichment scores (regions x cell_types).
    """
    import pandas as pd

    if hasattr(proportions, "values"):
        prop_array = proportions.values
        cell_types = proportions.columns.tolist()
    else:
        prop_array = proportions
        cell_types = [f"CellType_{i}" for i in range(proportions.shape[1])]

    regions = adata.obs[region_key].values
    unique_regions = np.unique(regions)

    # Global mean proportions
    global_mean = prop_array.mean(axis=0)

    # Enrichment per region
    enrichment = np.zeros((len(unique_regions), len(cell_types)))
    for i, region in enumerate(unique_regions):
        mask = regions == region
        region_mean = prop_array[mask].mean(axis=0)
        # Log fold change
        enrichment[i] = np.log2((region_mean + 1e-8) / (global_mean + 1e-8))

    return pd.DataFrame(
        enrichment,
        index=unique_regions,
        columns=cell_types,
    )


def validate_reference_overlap(
    sc_adata: "AnnData",
    st_adata: "AnnData",
    min_genes: int = 100,
) -> dict[str, int | list[str]]:
    """Validate gene overlap between reference and spatial data.

    Parameters
    ----------
    sc_adata
        Single-cell reference AnnData.
    st_adata
        Spatial transcriptomics AnnData.
    min_genes
        Minimum required gene overlap.

    Returns
    -------
    Dictionary with overlap statistics.
    """
    sc_genes = set(sc_adata.var_names)
    st_genes = set(st_adata.var_names)

    shared = sc_genes & st_genes
    sc_only = sc_genes - st_genes
    st_only = st_genes - sc_genes

    result = {
        "n_shared": len(shared),
        "n_sc_only": len(sc_only),
        "n_st_only": len(st_only),
        "shared_genes": sorted(list(shared)),
        "is_valid": len(shared) >= min_genes,
    }

    if len(shared) < min_genes:
        logger.warning(
            f"Only {len(shared)} shared genes found. "
            f"Minimum {min_genes} required for reliable deconvolution."
        )

    return result
