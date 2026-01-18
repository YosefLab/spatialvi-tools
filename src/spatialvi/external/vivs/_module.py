"""VIVS module for spatial variable gene detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class VIVSModule(nn.Module):
    """VIVS module for computing spatial variance statistics.

    This module implements the core computations for VIVS:
    - Multi-scale local variance computation
    - Permutation-based null distribution estimation
    - Importance score calculation

    Parameters
    ----------
    n_genes
        Number of genes.
    n_scales
        Number of spatial scales to analyze.
    n_neighbors_base
        Base number of neighbors for the smallest scale.
    use_gpu
        Whether to use GPU acceleration.
    """

    def __init__(
        self,
        n_genes: int,
        n_scales: int = 5,
        n_neighbors_base: int = 20,
        use_gpu: bool = False,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_scales = n_scales
        self.n_neighbors_base = n_neighbors_base

        # Compute scales
        self.scales = torch.tensor(
            np.linspace(n_neighbors_base, n_neighbors_base * n_scales, n_scales, dtype=int)
        )

        # Device
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # Buffer for storing results
        self.register_buffer("scale_scores", torch.zeros(n_genes, n_scales))
        self.register_buffer("importance_scores", torch.zeros(n_genes))

    def compute_local_variance(
        self,
        X: Tensor,
        neighbor_indices: Tensor,
    ) -> Tensor:
        """Compute local variance for each gene.

        Parameters
        ----------
        X
            Expression tensor of shape (n_cells, n_genes).
        neighbor_indices
            Neighbor indices of shape (n_cells, n_neighbors).

        Returns
        -------
        Local variance tensor of shape (n_genes,).
        """
        n_cells, n_genes = X.shape

        # Get neighbor values: (n_cells, n_neighbors, n_genes)
        neighbor_X = X[neighbor_indices.long()]

        # Compute local mean
        local_mean = neighbor_X.mean(dim=1)  # (n_cells, n_genes)

        # Compute deviation from local mean
        deviations = X - local_mean

        # Return variance of deviations
        return deviations.var(dim=0)

    def compute_null_distribution(
        self,
        X: Tensor,
        neighbor_indices: Tensor,
        n_permutations: int = 100,
    ) -> tuple[Tensor, Tensor]:
        """Compute null distribution via permutation.

        Parameters
        ----------
        X
            Expression tensor.
        neighbor_indices
            Neighbor indices.
        n_permutations
            Number of permutations.

        Returns
        -------
        Tuple of (null_mean, null_std).
        """
        n_cells = X.shape[0]
        null_vars = []

        for _ in range(n_permutations):
            perm = torch.randperm(n_cells, device=X.device)
            perm_var = self.compute_local_variance(X[perm], neighbor_indices)
            null_vars.append(perm_var)

        null_vars = torch.stack(null_vars, dim=0)
        null_mean = null_vars.mean(dim=0)
        null_std = null_vars.std(dim=0) + 1e-8

        return null_mean, null_std

    def forward(
        self,
        X: Tensor,
        neighbor_indices: Tensor,
        n_permutations: int = 100,
    ) -> dict[str, Tensor]:
        """Run VIVS analysis for a single scale.

        Parameters
        ----------
        X
            Expression tensor of shape (n_cells, n_genes).
        neighbor_indices
            Neighbor indices of shape (n_cells, n_neighbors).
        n_permutations
            Number of permutations for null distribution.

        Returns
        -------
        Dictionary with z_scores and local_variance.
        """
        # Compute local variance
        local_var = self.compute_local_variance(X, neighbor_indices)

        # Compute null distribution
        null_mean, null_std = self.compute_null_distribution(X, neighbor_indices, n_permutations)

        # Z-score
        z_scores = (local_var - null_mean) / null_std

        return {
            "z_scores": z_scores,
            "local_variance": local_var,
            "null_mean": null_mean,
            "null_std": null_std,
        }


class MultiScaleVIVS(nn.Module):
    """Multi-scale VIVS for robust spatial variable gene detection.

    Parameters
    ----------
    n_genes
        Number of genes.
    scales
        List of neighbor counts for each scale.
    aggregation
        How to aggregate across scales.
    """

    def __init__(
        self,
        n_genes: int,
        scales: list[int] | None = None,
        aggregation: Literal["max", "mean", "weighted"] = "max",
    ):
        super().__init__()
        self.n_genes = n_genes
        self.scales = scales or [10, 20, 50, 100, 200]
        self.n_scales = len(self.scales)
        self.aggregation = aggregation

        # Scale weights for weighted aggregation
        if aggregation == "weighted":
            self.scale_weights = nn.Parameter(torch.ones(self.n_scales) / self.n_scales)

    def aggregate_scores(self, scale_scores: Tensor) -> Tensor:
        """Aggregate z-scores across scales.

        Parameters
        ----------
        scale_scores
            Tensor of shape (n_genes, n_scales).

        Returns
        -------
        Aggregated scores of shape (n_genes,).
        """
        if self.aggregation == "max":
            return torch.max(torch.abs(scale_scores), dim=1)[0]
        elif self.aggregation == "mean":
            return torch.abs(scale_scores).mean(dim=1)
        elif self.aggregation == "weighted":
            weights = torch.softmax(self.scale_weights, dim=0)
            return (torch.abs(scale_scores) * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def forward(
        self,
        X: Tensor,
        neighbor_indices_list: list[Tensor],
        n_permutations: int = 100,
    ) -> dict[str, Tensor]:
        """Run multi-scale VIVS analysis.

        Parameters
        ----------
        X
            Expression tensor.
        neighbor_indices_list
            List of neighbor indices for each scale.
        n_permutations
            Number of permutations.

        Returns
        -------
        Dictionary with importance_scores and scale_scores.
        """
        n_genes = X.shape[1]
        scale_scores = torch.zeros(n_genes, len(neighbor_indices_list), device=X.device)

        for i, neighbor_indices in enumerate(neighbor_indices_list):
            module = VIVSModule(n_genes, n_scales=1)
            results = module(X, neighbor_indices, n_permutations)
            scale_scores[:, i] = results["z_scores"]

        importance_scores = self.aggregate_scores(scale_scores)

        return {
            "importance_scores": importance_scores,
            "scale_scores": scale_scores,
        }
