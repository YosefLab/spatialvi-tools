from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from anndata import AnnData


def select_genes(
    adata: AnnData,
    n_top_genes: int,
    preselected_genes: list[str] | None = None,
    seed: int = 0,
) -> AnnData:
    """Select a representative gene subset via a highly-variable-genes + clustering heuristic.

    Ported from VIVS's original JAX implementation (:cite:p:`Boyeau24`); pure
    scanpy/sklearn, unchanged by the torch port.

    Parameters
    ----------
    adata
        AnnData with raw counts in ``.X``.
    n_top_genes
        Number of genes to keep.
    preselected_genes
        Gene names to always keep in addition to the selected ones.
    seed
        Random seed for the KMeans clustering step.
    """
    adata_ = adata.copy()
    preselected_genes = preselected_genes if preselected_genes is not None else []

    adata_log = adata_.copy()
    sc.pp.normalize_total(adata_log, target_sum=1e6)
    sc.pp.log1p(adata_log)
    pca_ = PCA(n_components=50).fit(adata_log.X)
    sc.pp.highly_variable_genes(adata_, n_top_genes=n_top_genes, flavor="seurat_v3")

    clusters = KMeans(n_clusters=n_top_genes, random_state=seed, n_init=10).fit_predict(
        pca_.components_.T
    )
    adata_.var.loc[:, "clusters"] = clusters
    adata_.var.index.name = "index"
    selected_genes = (
        adata_.var.reset_index()
        .groupby("clusters")
        .apply(lambda x: x.sort_values("variances_norm").iloc[-1]["index"])
        .values
    )
    union_genes = np.union1d(selected_genes, preselected_genes)
    return adata_[:, adata_.var.index.isin(union_genes)].copy()
