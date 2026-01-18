"""Utility functions for AMICI cell-cell interaction analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_interaction_neighbors(
    adata: AnnData,
    spatial_key: str = "spatial",
    n_neighbors: int = 20,
    max_dist: float | None = None,
) -> tuple[NDArray, NDArray]:
    """Compute spatial neighbors for interaction analysis.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates.
    spatial_key
        Key in obsm for spatial coordinates.
    n_neighbors
        Number of neighbors to compute.
    max_dist
        Maximum distance for neighbors.

    Returns
    -------
    Tuple of (neighbor_indices, neighbor_distances).
    """
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Exclude self
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    # Apply distance threshold if specified
    if max_dist is not None:
        mask = distances <= max_dist
        # Pad with -1 for missing neighbors
        indices = np.where(mask, indices, -1)
        distances = np.where(mask, distances, np.inf)

    return indices.astype(np.int64), distances.astype(np.float32)


def build_interaction_matrix(
    adata: AnnData,
    labels_key: str,
    neighbor_indices: NDArray,
) -> pd.DataFrame:
    """Build cell type interaction frequency matrix.

    Parameters
    ----------
    adata
        AnnData object.
    labels_key
        Key in obs for cell type labels.
    neighbor_indices
        Neighbor index array.

    Returns
    -------
    DataFrame with interaction frequencies.
    """
    labels = adata.obs[labels_key].values
    if hasattr(labels, "codes"):
        label_codes = labels.codes
        label_names = labels.categories.tolist()
    else:
        unique_labels = np.unique(labels)
        label_codes = np.searchsorted(unique_labels, labels)
        label_names = unique_labels.tolist()

    n_types = len(label_names)
    interaction_counts = np.zeros((n_types, n_types))

    for i, cell_type in enumerate(label_codes):
        for neighbor_idx in neighbor_indices[i]:
            if neighbor_idx >= 0:
                neighbor_type = label_codes[neighbor_idx]
                interaction_counts[cell_type, neighbor_type] += 1

    # Normalize by cell type frequencies
    cell_counts = np.bincount(label_codes, minlength=n_types)
    expected = np.outer(cell_counts, cell_counts) / (len(labels) * neighbor_indices.shape[1])

    # Compute enrichment (log odds ratio)
    observed = interaction_counts / interaction_counts.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        enrichment = np.log2((observed + 1e-8) / (expected + 1e-8))
        enrichment = np.nan_to_num(enrichment, nan=0, posinf=0, neginf=0)

    return pd.DataFrame(
        enrichment,
        index=label_names,
        columns=label_names,
    )


def compute_interaction_strength(
    attention_weights: NDArray,
    labels: NDArray,
) -> dict[tuple[str, str], float]:
    """Compute interaction strength from attention weights.

    Parameters
    ----------
    attention_weights
        Attention weights of shape (n_cells, n_neighbors).
    labels
        Cell type labels.

    Returns
    -------
    Dictionary mapping cell type pairs to interaction strength.
    """
    if hasattr(labels, "codes"):
        label_codes = labels.codes
        label_names = labels.categories.tolist()
    else:
        unique_labels = np.unique(labels)
        label_codes = np.searchsorted(unique_labels, labels)
        label_names = unique_labels.tolist()

    # Aggregate attention by cell type pairs
    strengths = {}
    for ct1 in range(len(label_names)):
        for ct2 in range(len(label_names)):
            strengths[(label_names[ct1], label_names[ct2])] = 0.0

    # This is a placeholder - actual implementation would aggregate attention
    # weights based on cell types

    return strengths


def get_ligand_receptor_pairs(
    species: str = "human",
    database: str = "cellchatdb",
) -> pd.DataFrame:
    """Get ligand-receptor pair database.

    Parameters
    ----------
    species
        Species ("human" or "mouse").
    database
        Database to use.

    Returns
    -------
    DataFrame with ligand-receptor pairs.
    """
    # Common ligand-receptor pairs (subset)
    pairs = [
        ("CCL19", "CCR7", "chemokine"),
        ("CCL21", "CCR7", "chemokine"),
        ("CXCL12", "CXCR4", "chemokine"),
        ("IL6", "IL6R", "cytokine"),
        ("IL1B", "IL1R1", "cytokine"),
        ("TNF", "TNFRSF1A", "cytokine"),
        ("TGFB1", "TGFBR1", "growth_factor"),
        ("VEGFA", "FLT1", "growth_factor"),
        ("WNT5A", "FZD5", "signaling"),
        ("NOTCH1", "DLL1", "signaling"),
        ("COL1A1", "ITGA1", "ecm"),
        ("FN1", "ITGA5", "ecm"),
    ]

    df = pd.DataFrame(pairs, columns=["ligand", "receptor", "category"])

    if species == "mouse":
        df["ligand"] = df["ligand"].str.capitalize()
        df["receptor"] = df["receptor"].str.capitalize()

    return df


def filter_expressed_pairs(
    adata: AnnData,
    pairs: pd.DataFrame,
    min_pct: float = 0.1,
) -> pd.DataFrame:
    """Filter L-R pairs to those expressed in data.

    Parameters
    ----------
    adata
        AnnData object.
    pairs
        DataFrame with ligand-receptor pairs.
    min_pct
        Minimum percentage of cells expressing gene.

    Returns
    -------
    Filtered DataFrame.
    """
    expressed_genes = set(adata.var_names)

    # Check expression
    ligand_mask = pairs["ligand"].isin(expressed_genes)
    receptor_mask = pairs["receptor"].isin(expressed_genes)

    filtered = pairs[ligand_mask & receptor_mask].copy()

    if min_pct > 0:
        # Further filter by expression percentage
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        n_cells = X.shape[0]
        pct_threshold = n_cells * min_pct

        keep_mask = []
        for _, row in filtered.iterrows():
            lig_idx = np.where(adata.var_names == row["ligand"])[0][0]
            rec_idx = np.where(adata.var_names == row["receptor"])[0][0]

            lig_expressing = (X[:, lig_idx] > 0).sum()
            rec_expressing = (X[:, rec_idx] > 0).sum()

            keep_mask.append(lig_expressing >= pct_threshold and rec_expressing >= pct_threshold)

        filtered = filtered[keep_mask]

    logger.info(f"Filtered to {len(filtered)} expressed L-R pairs")

    return filtered.reset_index(drop=True)
