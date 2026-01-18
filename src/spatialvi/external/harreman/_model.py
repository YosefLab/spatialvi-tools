"""Harreman model for metabolic exchange inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats
from scipy.sparse import issparse

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


class Harreman:
    """Harreman model for metabolic exchange inference.

    Harreman infers metabolic exchange between spatially proximal cells
    by analyzing correlations between metabolic gene expression and
    spatial proximity patterns.

    Parameters
    ----------
    adata
        AnnData object with spatial transcriptomics data.
    metabolic_genes
        List of metabolic genes to analyze. If None, uses a default
        set of metabolic pathway genes.
    spatial_key
        Key in obsm for spatial coordinates.
    labels_key
        Key in obs for cell type labels.
    n_neighbors
        Number of spatial neighbors to consider.

    Examples
    --------
    >>> import spatialvi
    >>> adata = spatialvi.data.synthetic_spatial()
    >>> model = Harreman(adata, labels_key="cell_type")
    >>> model.fit()
    >>> exchanges = model.get_metabolic_exchanges()
    """

    def __init__(
        self,
        adata: AnnData,
        metabolic_genes: Sequence[str] | None = None,
        spatial_key: str = "spatial",
        labels_key: str | None = None,
        n_neighbors: int = 20,
    ):
        self.adata = adata
        self.spatial_key = spatial_key
        self.labels_key = labels_key
        self.n_neighbors = n_neighbors

        # Set metabolic genes
        if metabolic_genes is not None:
            self.metabolic_genes = [g for g in metabolic_genes if g in adata.var_names]
        else:
            # Default metabolic genes (subset)
            default_genes = self._get_default_metabolic_genes()
            self.metabolic_genes = [g for g in default_genes if g in adata.var_names]

        logger.info(f"Using {len(self.metabolic_genes)} metabolic genes")

        self._fitted = False
        self._exchange_scores = None
        self._neighbor_indices = None

    def _get_default_metabolic_genes(self) -> list[str]:
        """Get default metabolic pathway genes."""
        # Common metabolic genes (glycolysis, TCA, oxidative phosphorylation)
        return [
            # Glycolysis
            "HK1",
            "HK2",
            "GPI",
            "PFKL",
            "PFKM",
            "ALDOA",
            "ALDOB",
            "GAPDH",
            "PGK1",
            "PGAM1",
            "ENO1",
            "ENO2",
            "PKM",
            "LDHA",
            "LDHB",
            # TCA cycle
            "CS",
            "ACO1",
            "ACO2",
            "IDH1",
            "IDH2",
            "IDH3A",
            "OGDH",
            "SUCLA2",
            "SUCLG1",
            "SDHA",
            "SDHB",
            "FH",
            "MDH1",
            "MDH2",
            # Oxidative phosphorylation
            "NDUFA1",
            "NDUFB1",
            "NDUFS1",
            "NDUFS2",
            "UQCRC1",
            "UQCRC2",
            "COX4I1",
            "COX5A",
            "COX6A1",
            "ATP5A1",
            "ATP5B",
            "ATP5C1",
            # Amino acid metabolism
            "GOT1",
            "GOT2",
            "GPT",
            "GLS",
            "GLUD1",
            # Lipid metabolism
            "FASN",
            "ACACA",
            "SCD",
            "FADS1",
            "FADS2",
            # Transporters
            "SLC2A1",
            "SLC2A3",
            "SLC16A1",
            "SLC16A3",
            "SLC1A5",
        ]

    def fit(
        self,
        n_permutations: int = 100,
        correlation_method: Literal["pearson", "spearman"] = "spearman",
    ) -> Harreman:
        """Fit the Harreman model.

        Parameters
        ----------
        n_permutations
            Number of permutations for null distribution.
        correlation_method
            Correlation method to use.

        Returns
        -------
        Self.
        """
        # Get spatial neighbors
        self._compute_spatial_neighbors()

        # Get expression data for metabolic genes
        gene_mask = [self.adata.var_names.get_loc(g) for g in self.metabolic_genes]
        X = self.adata.X[:, gene_mask]
        if issparse(X):
            X = X.toarray()

        n_cells = X.shape[0]
        n_genes = len(self.metabolic_genes)

        # Compute spatial correlation for each gene
        spatial_corr = np.zeros(n_genes)
        null_corrs = np.zeros((n_genes, n_permutations))

        for g in range(n_genes):
            gene_expr = X[:, g]

            # Compute mean neighbor expression
            neighbor_mean = np.zeros(n_cells)
            for i in range(n_cells):
                neighbor_mean[i] = gene_expr[self._neighbor_indices[i]].mean()

            # Compute correlation
            if correlation_method == "spearman":
                corr, _ = stats.spearmanr(gene_expr, neighbor_mean)
            else:
                corr, _ = stats.pearsonr(gene_expr, neighbor_mean)

            spatial_corr[g] = corr

            # Null distribution
            for p in range(n_permutations):
                perm = np.random.permutation(n_cells)
                perm_neighbor_mean = np.zeros(n_cells)
                for i in range(n_cells):
                    perm_neighbor_mean[i] = gene_expr[perm][self._neighbor_indices[i]].mean()

                if correlation_method == "spearman":
                    null_corr, _ = stats.spearmanr(gene_expr, perm_neighbor_mean)
                else:
                    null_corr, _ = stats.pearsonr(gene_expr, perm_neighbor_mean)

                null_corrs[g, p] = null_corr

        # Compute z-scores and p-values
        null_mean = null_corrs.mean(axis=1)
        null_std = null_corrs.std(axis=1) + 1e-8
        z_scores = (spatial_corr - null_mean) / null_std
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

        self._exchange_scores = pd.DataFrame(
            {
                "gene": self.metabolic_genes,
                "spatial_correlation": spatial_corr,
                "z_score": z_scores,
                "p_value": p_values,
            }
        )

        self._fitted = True

        return self

    def _compute_spatial_neighbors(self) -> None:
        """Compute spatial nearest neighbors."""
        from sklearn.neighbors import NearestNeighbors

        coords = self.adata.obsm[self.spatial_key]
        if hasattr(coords, "values"):
            coords = coords.values

        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nn.fit(coords)
        _, indices = nn.kneighbors(coords)

        self._neighbor_indices = indices[:, 1:]  # Exclude self

    def get_metabolic_exchanges(
        self,
        fdr_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Get significant metabolic exchange genes.

        Parameters
        ----------
        fdr_threshold
            FDR threshold for significance.

        Returns
        -------
        DataFrame with exchange scores and statistics.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        from statsmodels.stats.multitest import multipletests

        results = self._exchange_scores.copy()

        # FDR correction
        _, fdr, _, _ = multipletests(results["p_value"], method="fdr_bh")
        results["fdr"] = fdr

        # Classify as sender/receiver based on correlation sign
        results["exchange_type"] = np.where(
            results["spatial_correlation"] > 0,
            "sender",
            "receiver",
        )

        return results.sort_values("z_score", ascending=False)

    def get_cell_type_exchanges(
        self,
        min_cells: int = 10,
    ) -> pd.DataFrame:
        """Get metabolic exchanges between cell types.

        Parameters
        ----------
        min_cells
            Minimum cells per cell type.

        Returns
        -------
        DataFrame with cell type pair exchanges.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.labels_key is None:
            raise ValueError("labels_key must be provided for cell type analysis")

        labels = self.adata.obs[self.labels_key].values
        if hasattr(labels, "codes"):
            unique_labels = self.adata.obs[self.labels_key].cat.categories.tolist()
            labels = labels.codes
        else:
            unique_labels = list(np.unique(labels))

        # Get expression data
        gene_mask = [self.adata.var_names.get_loc(g) for g in self.metabolic_genes]
        X = self.adata.X[:, gene_mask]
        if issparse(X):
            X = X.toarray()

        # Compute exchanges between cell type pairs
        exchanges = []

        for i, ct1 in enumerate(unique_labels):
            mask1 = labels == i
            if mask1.sum() < min_cells:
                continue

            for j, ct2 in enumerate(unique_labels):
                if i == j:
                    continue

                # Find cells of type 1 with neighbors of type 2
                exchange_cells = []
                for cell_idx in np.where(mask1)[0]:
                    neighbor_labels = labels[self._neighbor_indices[cell_idx]]
                    if (neighbor_labels == j).any():
                        exchange_cells.append(cell_idx)

                if len(exchange_cells) < min_cells:
                    continue

                # Compute mean expression in exchange cells
                exchange_cells = np.array(exchange_cells)
                mean_expr = X[exchange_cells].mean(axis=0)

                for g, gene in enumerate(self.metabolic_genes):
                    exchanges.append(
                        {
                            "sender": ct1,
                            "receiver": ct2,
                            "gene": gene,
                            "mean_expression": mean_expr[g],
                            "n_cells": len(exchange_cells),
                        }
                    )

        return pd.DataFrame(exchanges)

    def to_adata(self) -> None:
        """Store results in AnnData object."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Store gene-level results in var
        results = self.get_metabolic_exchanges()
        for col in ["spatial_correlation", "z_score", "p_value", "fdr"]:
            self.adata.var[f"harreman_{col}"] = np.nan
            for _, row in results.iterrows():
                if row["gene"] in self.adata.var_names:
                    idx = self.adata.var_names.get_loc(row["gene"])
                    self.adata.var.iloc[idx, self.adata.var.columns.get_loc(f"harreman_{col}")] = row[col]
