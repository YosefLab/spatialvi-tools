"""Utility functions for scVIVA niche analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_niche_composition(
    adata: AnnData,
    labels_key: str,
    spatial_key: str = "spatial",
    n_neighbors: int = 20,
) -> NDArray:
    """Compute cell type composition of spatial niches.

    Parameters
    ----------
    adata
        AnnData with spatial coordinates and cell labels.
    labels_key
        Key for cell type labels in obs.
    spatial_key
        Key for spatial coordinates in obsm.
    n_neighbors
        Number of neighbors defining the niche.

    Returns
    -------
    Niche composition matrix (cells x cell_types).
    """
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    labels = adata.obs[labels_key].values
    unique_labels = np.unique(labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    # Find spatial neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    indices = indices[:, 1:]  # Exclude self

    # Compute composition
    n_cells = len(coords)
    n_types = len(unique_labels)
    composition = np.zeros((n_cells, n_types))

    for i in range(n_cells):
        neighbor_labels = labels[indices[i]]
        for label in neighbor_labels:
            composition[i, label_to_idx[label]] += 1
        composition[i] /= n_neighbors

    return composition


def identify_niche_clusters(
    niche_composition: NDArray,
    n_clusters: int | None = None,
    resolution: float = 1.0,
    method: str = "leiden",
) -> NDArray:
    """Cluster cells by niche composition.

    Parameters
    ----------
    niche_composition
        Niche composition matrix.
    n_clusters
        Number of clusters (for kmeans).
    resolution
        Resolution parameter (for leiden).
    method
        Clustering method ("kmeans", "leiden").

    Returns
    -------
    Cluster assignments per cell.
    """
    if method == "kmeans":
        from sklearn.cluster import KMeans

        n_clusters = n_clusters or 10
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(niche_composition)

    elif method == "leiden":
        import anndata
        import scanpy as sc

        # Create temporary AnnData for clustering
        temp_adata = anndata.AnnData(niche_composition)
        sc.pp.neighbors(temp_adata, use_rep="X")
        sc.tl.leiden(temp_adata, resolution=resolution)
        return temp_adata.obs["leiden"].astype(int).values

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_niche_heterogeneity(
    adata: AnnData,
    niche_key: str,
    expression_key: str | None = None,
) -> dict[str, float]:
    """Compute heterogeneity metrics per niche.

    Parameters
    ----------
    adata
        AnnData object.
    niche_key
        Key for niche assignments in obs.
    expression_key
        Key for expression layer (None for X).

    Returns
    -------
    Dictionary mapping niches to heterogeneity scores.
    """
    X = adata.X if expression_key is None else adata.layers[expression_key]
    if hasattr(X, "toarray"):
        X = X.toarray()

    niches = adata.obs[niche_key].values
    unique_niches = np.unique(niches)

    heterogeneity = {}
    for niche in unique_niches:
        mask = niches == niche
        niche_expr = X[mask]

        # Compute coefficient of variation across cells
        cv = np.std(niche_expr, axis=0) / (np.mean(niche_expr, axis=0) + 1e-8)
        heterogeneity[str(niche)] = float(np.mean(cv))

    return heterogeneity


def compute_niche_interaction_strength(
    adata: AnnData,
    labels_key: str,
    spatial_key: str = "spatial",
    n_neighbors: int = 20,
) -> NDArray:
    """Compute cell type interaction strength matrix.

    Parameters
    ----------
    adata
        AnnData with spatial coordinates.
    labels_key
        Key for cell type labels.
    spatial_key
        Key for spatial coordinates.
    n_neighbors
        Number of neighbors.

    Returns
    -------
    Interaction matrix (cell_types x cell_types).
    """
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    labels = adata.obs[labels_key].values
    unique_labels = np.unique(labels)
    n_types = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    # Find neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    indices = indices[:, 1:]

    # Count interactions
    interactions = np.zeros((n_types, n_types))
    for i in range(len(coords)):
        cell_type = label_to_idx[labels[i]]
        for j in indices[i]:
            neighbor_type = label_to_idx[labels[j]]
            interactions[cell_type, neighbor_type] += 1

    # Normalize by cell type counts
    type_counts = np.array([np.sum(labels == l) for l in unique_labels])
    interactions = interactions / (type_counts[:, None] + 1e-8)

    return interactions


def identify_boundary_cells(
    adata: AnnData,
    labels_key: str,
    spatial_key: str = "spatial",
    n_neighbors: int = 10,
    threshold: float = 0.3,
) -> NDArray:
    """Identify cells at boundaries between cell types.

    Parameters
    ----------
    adata
        AnnData with spatial coordinates.
    labels_key
        Key for cell type labels.
    spatial_key
        Key for spatial coordinates.
    n_neighbors
        Number of neighbors.
    threshold
        Minimum fraction of different neighbors.

    Returns
    -------
    Boolean array indicating boundary cells.
    """
    from sklearn.neighbors import NearestNeighbors

    coords = adata.obsm[spatial_key]
    if hasattr(coords, "values"):
        coords = coords.values

    labels = adata.obs[labels_key].values

    # Find neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    indices = indices[:, 1:]

    # Compute fraction of different neighbors
    n_cells = len(coords)
    different_fraction = np.zeros(n_cells)

    for i in range(n_cells):
        cell_label = labels[i]
        neighbor_labels = labels[indices[i]]
        different_fraction[i] = np.mean(neighbor_labels != cell_label)

    return different_fraction >= threshold


def compute_niche_differential_genes(
    adata: AnnData,
    niche_key: str,
    reference_niche: str | None = None,
    n_genes: int = 50,
    method: str = "wilcoxon",
) -> dict[str, list[str]]:
    """Find differentially expressed genes per niche.

    Parameters
    ----------
    adata
        AnnData object.
    niche_key
        Key for niche assignments.
    reference_niche
        Reference niche for comparison (None for vs rest).
    n_genes
        Number of top genes to return.
    method
        Statistical method.

    Returns
    -------
    Dictionary mapping niches to gene lists.
    """
    import scanpy as sc

    adata = adata.copy()
    sc.tl.rank_genes_groups(
        adata,
        groupby=niche_key,
        reference=reference_niche if reference_niche else "rest",
        method=method,
    )

    result = {}
    for niche in adata.obs[niche_key].unique():
        genes = adata.uns["rank_genes_groups"]["names"][niche][:n_genes]
        result[str(niche)] = genes.tolist()

    return result


def compute_spatial_entropy(
    adata: AnnData,
    labels_key: str,
    spatial_key: str = "spatial",
    n_neighbors: int = 20,
) -> NDArray:
    """Compute local spatial entropy of cell type diversity.

    Parameters
    ----------
    adata
        AnnData with spatial coordinates.
    labels_key
        Key for cell type labels.
    spatial_key
        Key for spatial coordinates.
    n_neighbors
        Number of neighbors.

    Returns
    -------
    Entropy values per cell.
    """
    from scipy.stats import entropy

    composition = compute_niche_composition(adata, labels_key, spatial_key, n_neighbors)

    # Compute entropy per cell
    entropies = np.array([entropy(comp + 1e-10) for comp in composition])

    return entropies


def visualize_niche_embedding(
    adata: AnnData,
    niche_effects: NDArray,
    labels_key: str | None = None,
    method: str = "umap",
    n_components: int = 2,
) -> NDArray:
    """Create 2D embedding of niche effects.

    Parameters
    ----------
    adata
        AnnData object (for labels).
    niche_effects
        Niche effects matrix from scVIVA.
    labels_key
        Key for coloring (optional).
    method
        Embedding method ("umap", "tsne", "pca").
    n_components
        Number of output dimensions.

    Returns
    -------
    2D embedding coordinates.
    """
    if method == "umap":
        from umap import UMAP

        reducer = UMAP(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(niche_effects)

    elif method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(niche_effects)

    elif method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components)
        embedding = reducer.fit_transform(niche_effects)

    else:
        raise ValueError(f"Unknown method: {method}")

    return embedding
