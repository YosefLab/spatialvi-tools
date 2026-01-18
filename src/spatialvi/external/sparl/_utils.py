"""Utility functions for SPARL spatial proteomics analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def normalize_protein_expression(
    X: NDArray,
    method: str = "arcsinh",
    cofactor: float = 5.0,
) -> NDArray:
    """Normalize protein expression data.

    Parameters
    ----------
    X
        Raw protein expression matrix.
    method
        Normalization method ("arcsinh", "log1p", "zscore").
    cofactor
        Cofactor for arcsinh transformation.

    Returns
    -------
    Normalized expression matrix.
    """
    if method == "arcsinh":
        return np.arcsinh(X / cofactor)
    elif method == "log1p":
        return np.log1p(X)
    elif method == "zscore":
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True) + 1e-8
        return (X - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_protein_coexpression(
    X: NDArray,
    method: str = "pearson",
) -> NDArray:
    """Compute protein co-expression matrix.

    Parameters
    ----------
    X
        Protein expression matrix (cells x proteins).
    method
        Correlation method.

    Returns
    -------
    Co-expression matrix (proteins x proteins).
    """
    if method == "pearson":
        return np.corrcoef(X.T)
    elif method == "spearman":
        from scipy import stats

        n_proteins = X.shape[1]
        coexpr = np.zeros((n_proteins, n_proteins))
        for i in range(n_proteins):
            for j in range(i, n_proteins):
                r, _ = stats.spearmanr(X[:, i], X[:, j])
                coexpr[i, j] = r
                coexpr[j, i] = r
        return coexpr
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_protein_communities(
    coexpression: NDArray,
    n_communities: int | None = None,
    resolution: float = 1.0,
) -> NDArray:
    """Detect protein communities from co-expression.

    Parameters
    ----------
    coexpression
        Co-expression matrix.
    n_communities
        Number of communities (if None, uses Leiden).
    resolution
        Resolution for Leiden clustering.

    Returns
    -------
    Community assignments for each protein.
    """
    import networkx as nx

    # Build graph from positive correlations
    G = nx.Graph()
    n_proteins = coexpression.shape[0]

    for i in range(n_proteins):
        G.add_node(i)
        for j in range(i + 1, n_proteins):
            weight = coexpression[i, j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    if n_communities is not None:
        # Use spectral clustering
        from sklearn.cluster import SpectralClustering

        clustering = SpectralClustering(
            n_clusters=n_communities,
            affinity="precomputed",
            assign_labels="discretize",
        )
        adj_matrix = np.maximum(coexpression, 0)
        return clustering.fit_predict(adj_matrix)
    else:
        # Use community detection
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(G, resolution=resolution)
            return np.array([partition[i] for i in range(n_proteins)])
        except ImportError:
            logger.warning("python-louvain not installed, using connected components")
            components = nx.connected_components(G)
            labels = np.zeros(n_proteins, dtype=int)
            for i, comp in enumerate(components):
                for node in comp:
                    labels[node] = i
            return labels


def spatial_protein_enrichment(
    adata: AnnData,
    labels_key: str,
    spatial_key: str = "spatial",
    n_neighbors: int = 20,
) -> NDArray:
    """Compute spatial enrichment of protein expression.

    Parameters
    ----------
    adata
        AnnData object with protein expression.
    labels_key
        Key in obs for region/cluster labels.
    spatial_key
        Key in obsm for spatial coordinates.
    n_neighbors
        Number of spatial neighbors.

    Returns
    -------
    Enrichment matrix (regions x proteins).
    """
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    labels = adata.obs[labels_key].values
    unique_labels = np.unique(labels)

    # Compute neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    indices = indices[:, 1:]  # Exclude self

    # Compute enrichment
    n_regions = len(unique_labels)
    n_proteins = X.shape[1]
    enrichment = np.zeros((n_regions, n_proteins))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        region_cells = np.where(mask)[0]

        # Mean expression in region
        region_expr = X[mask].mean(axis=0)

        # Mean neighbor expression (from outside region)
        neighbor_expr = []
        for cell_idx in region_cells:
            neighbor_mask = ~np.isin(indices[cell_idx], region_cells)
            if neighbor_mask.any():
                neighbor_expr.append(X[indices[cell_idx][neighbor_mask]].mean(axis=0))

        if neighbor_expr:
            neighbor_expr = np.mean(neighbor_expr, axis=0)
            enrichment[i] = np.log2((region_expr + 1) / (neighbor_expr + 1))

    return enrichment


def identify_marker_proteins(
    adata: AnnData,
    labels_key: str,
    n_markers: int = 10,
    method: str = "wilcoxon",
) -> dict[str, list[str]]:
    """Identify marker proteins for each group.

    Parameters
    ----------
    adata
        AnnData object.
    labels_key
        Key for group labels.
    n_markers
        Number of markers per group.
    method
        Statistical method.

    Returns
    -------
    Dictionary mapping groups to marker protein lists.
    """
    import scanpy as sc

    adata = adata.copy()
    sc.tl.rank_genes_groups(adata, groupby=labels_key, method=method)

    markers = {}
    for group in adata.obs[labels_key].unique():
        markers[str(group)] = (
            adata.uns["rank_genes_groups"]["names"][group][:n_markers].tolist()
        )

    return markers


def compute_neighborhood_composition(
    adata: AnnData,
    spatial_key: str = "spatial",
    n_neighbors: int = 20,
) -> NDArray:
    """Compute protein-based neighborhood composition.

    Parameters
    ----------
    adata
        AnnData object.
    spatial_key
        Key for spatial coordinates.
    n_neighbors
        Number of neighbors.

    Returns
    -------
    Neighborhood composition matrix.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Find neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    indices = indices[:, 1:]

    # Compute mean neighbor expression
    n_cells, n_proteins = X.shape
    neighbor_composition = np.zeros((n_cells, n_proteins))

    for i in range(n_cells):
        neighbor_composition[i] = X[indices[i]].mean(axis=0)

    return neighbor_composition
