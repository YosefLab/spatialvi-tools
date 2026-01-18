"""VIVS model for spatial variable selection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

from anndata import AnnData

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class VIVS:
    """VIVS model for identifying spatially variable genes.

    VIVS (Variable Importance via Variance Statistics) identifies
    genes with significant spatial patterns by comparing observed
    and expected variance under different spatial scales.

    Parameters
    ----------
    adata
        AnnData object with spatial transcriptomics data.
    spatial_key
        Key in obsm for spatial coordinates.
    layer
        Layer in adata to use. If None, uses X.
    n_neighbors
        Number of spatial neighbors to consider.
    n_scales
        Number of spatial scales to analyze.

    Examples
    --------
    >>> import spatialvi
    >>> adata = spatialvi.data.synthetic_spatial()
    >>> model = VIVS(adata, spatial_key="spatial")
    >>> model.fit()
    >>> svg = model.get_spatially_variable_genes()
    """

    def __init__(
        self,
        adata: AnnData,
        spatial_key: str = "spatial",
        layer: str | None = None,
        n_neighbors: int = 20,
        n_scales: int = 5,
    ):
        self.adata = adata
        self.spatial_key = spatial_key
        self.layer = layer
        self.n_neighbors = n_neighbors
        self.n_scales = n_scales

        self._fitted = False
        self._importance_scores = None
        self._scale_scores = None

    def fit(
        self,
        n_permutations: int = 100,
        batch_size: int = 100,
        use_gpu: bool = False,
    ) -> "VIVS":
        """Fit the VIVS model.

        Parameters
        ----------
        n_permutations
            Number of permutations for null distribution.
        batch_size
            Batch size for processing genes.
        use_gpu
            Whether to use GPU acceleration.

        Returns
        -------
        Self.
        """
        # Get expression data
        if self.layer is not None:
            X = self.adata.layers[self.layer]
        else:
            X = self.adata.X

        if hasattr(X, "toarray"):
            X = X.toarray()

        # Get spatial coordinates
        coords = self.adata.obsm[self.spatial_key]
        if hasattr(coords, "values"):
            coords = coords.values

        n_cells, n_genes = X.shape

        # Compute spatial neighbors at multiple scales
        from sklearn.neighbors import NearestNeighbors

        scales = np.linspace(
            self.n_neighbors,
            self.n_neighbors * self.n_scales,
            self.n_scales,
            dtype=int,
        )

        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        importance_scores = np.zeros(n_genes)
        scale_scores = np.zeros((n_genes, len(scales)))

        for scale_idx, k in enumerate(scales):
            logger.info(f"Processing scale {scale_idx + 1}/{len(scales)} (k={k})")

            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(coords)
            _, indices = nn.kneighbors(coords)
            neighbor_indices = indices[:, 1:]  # Exclude self

            # Process genes in batches
            for batch_start in range(0, n_genes, batch_size):
                batch_end = min(batch_start + batch_size, n_genes)
                X_batch = X[:, batch_start:batch_end]

                # Convert to tensor
                X_tensor = torch.tensor(X_batch, device=device, dtype=torch.float32)
                idx_tensor = torch.tensor(neighbor_indices, device=device, dtype=torch.long)

                # Compute local variance
                local_var = self._compute_local_variance(X_tensor, idx_tensor)

                # Compute null distribution via permutation
                null_vars = []
                for _ in range(n_permutations):
                    perm = torch.randperm(n_cells, device=device)
                    perm_var = self._compute_local_variance(X_tensor[perm], idx_tensor)
                    null_vars.append(perm_var)

                null_vars = torch.stack(null_vars, dim=0)
                null_mean = null_vars.mean(dim=0)
                null_std = null_vars.std(dim=0) + 1e-8

                # Z-score
                z_scores = (local_var - null_mean) / null_std

                scale_scores[batch_start:batch_end, scale_idx] = z_scores.cpu().numpy()

        # Aggregate across scales
        importance_scores = np.max(np.abs(scale_scores), axis=1)

        self._importance_scores = importance_scores
        self._scale_scores = scale_scores
        self._fitted = True

        return self

    def _compute_local_variance(
        self,
        X: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute local variance for each gene.

        Parameters
        ----------
        X
            Expression tensor of shape (n_cells, n_genes).
        neighbor_indices
            Neighbor indices of shape (n_cells, n_neighbors).

        Returns
        -------
        Local variance of shape (n_genes,).
        """
        n_cells, n_genes = X.shape
        n_neighbors = neighbor_indices.shape[1]

        # Get neighbor values
        neighbor_X = X[neighbor_indices.long()]  # (n_cells, n_neighbors, n_genes)

        # Compute local mean
        local_mean = neighbor_X.mean(dim=1)  # (n_cells, n_genes)

        # Compute deviation from local mean
        deviations = X - local_mean

        # Return variance of deviations
        return deviations.var(dim=0)

    def get_spatially_variable_genes(
        self,
        n_top: int | None = None,
        fdr_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Get spatially variable genes.

        Parameters
        ----------
        n_top
            Number of top genes to return. If None, returns all
            genes below FDR threshold.
        fdr_threshold
            FDR threshold for significance.

        Returns
        -------
        DataFrame with gene names, importance scores, and p-values.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        from scipy import stats

        # Compute p-values from z-scores
        max_z = np.max(np.abs(self._scale_scores), axis=1)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(max_z)))

        # FDR correction
        from statsmodels.stats.multitest import multipletests

        _, fdr, _, _ = multipletests(p_values, method="fdr_bh")

        # Create results DataFrame
        results = pd.DataFrame({
            "gene": self.adata.var_names,
            "importance_score": self._importance_scores,
            "max_z_score": max_z,
            "p_value": p_values,
            "fdr": fdr,
        })

        results = results.sort_values("importance_score", ascending=False)

        if n_top is not None:
            results = results.head(n_top)
        else:
            results = results[results["fdr"] < fdr_threshold]

        return results

    def get_scale_contributions(
        self,
        gene_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Get contribution of each scale to gene importance.

        Parameters
        ----------
        gene_names
            Genes to analyze. If None, uses all genes.

        Returns
        -------
        DataFrame with scale contributions per gene.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if gene_names is None:
            gene_mask = slice(None)
        else:
            gene_mask = [self.adata.var_names.get_loc(g) for g in gene_names]

        scale_names = [f"scale_{i+1}" for i in range(self.n_scales)]

        results = pd.DataFrame(
            self._scale_scores[gene_mask],
            index=self.adata.var_names[gene_mask] if gene_names is None else gene_names,
            columns=scale_names,
        )

        return results

    def to_adata(self) -> None:
        """Store results in AnnData object.

        Adds the following to adata.var:
        - 'vivs_importance': Importance scores
        - 'vivs_pvalue': P-values
        - 'vivs_fdr': FDR-adjusted p-values
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        results = self.get_spatially_variable_genes(n_top=len(self.adata.var_names))

        self.adata.var["vivs_importance"] = results.set_index("gene")["importance_score"]
        self.adata.var["vivs_pvalue"] = results.set_index("gene")["p_value"]
        self.adata.var["vivs_fdr"] = results.set_index("gene")["fdr"]
