"""Harreman module for metabolic exchange inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class HarremanModule(nn.Module):
    """Neural network module for Harreman metabolic exchange analysis.

    This module computes spatial correlations between metabolic genes
    and their neighborhood expression patterns using GPU acceleration.

    Parameters
    ----------
    n_genes
        Number of metabolic genes.
    n_neighbors
        Number of spatial neighbors.
    correlation_method
        Method for computing correlations.
    """

    def __init__(
        self,
        n_genes: int,
        n_neighbors: int = 20,
        correlation_method: Literal["pearson", "spearman"] = "pearson",
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_neighbors = n_neighbors
        self.correlation_method = correlation_method

    def compute_neighbor_expression(
        self,
        X: Tensor,
        neighbor_indices: Tensor,
    ) -> Tensor:
        """Compute mean neighbor expression.

        Parameters
        ----------
        X
            Gene expression tensor of shape (n_cells, n_genes).
        neighbor_indices
            Neighbor indices of shape (n_cells, n_neighbors).

        Returns
        -------
        Mean neighbor expression of shape (n_cells, n_genes).
        """
        # Gather neighbor expression
        neighbor_expr = X[neighbor_indices.long()]  # (n_cells, n_neighbors, n_genes)
        return neighbor_expr.mean(dim=1)

    def compute_spatial_correlation(
        self,
        X: Tensor,
        neighbor_mean: Tensor,
    ) -> Tensor:
        """Compute spatial correlation for each gene.

        Parameters
        ----------
        X
            Gene expression tensor.
        neighbor_mean
            Mean neighbor expression.

        Returns
        -------
        Correlation tensor of shape (n_genes,).
        """
        if self.correlation_method == "spearman":
            # Rank transform for Spearman
            X_rank = self._rank_transform(X)
            neighbor_rank = self._rank_transform(neighbor_mean)
            return self._pearson_correlation(X_rank, neighbor_rank)
        else:
            return self._pearson_correlation(X, neighbor_mean)

    def _rank_transform(self, X: Tensor) -> Tensor:
        """Convert to ranks along the cell dimension."""
        sorted_indices = torch.argsort(X, dim=0)
        ranks = torch.empty_like(X)
        for g in range(X.shape[1]):
            ranks[sorted_indices[:, g], g] = torch.arange(
                X.shape[0], dtype=X.dtype, device=X.device
            )
        return ranks

    def _pearson_correlation(self, X: Tensor, Y: Tensor) -> Tensor:
        """Compute Pearson correlation per gene."""
        n = X.shape[0]
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)

        numerator = (X_centered * Y_centered).sum(dim=0)
        denominator = torch.sqrt(
            (X_centered ** 2).sum(dim=0) * (Y_centered ** 2).sum(dim=0) + 1e-8
        )

        return numerator / denominator

    def forward(
        self,
        X: Tensor,
        neighbor_indices: Tensor,
    ) -> dict[str, Tensor]:
        """Run forward pass.

        Parameters
        ----------
        X
            Gene expression tensor.
        neighbor_indices
            Neighbor indices.

        Returns
        -------
        Dictionary with spatial_correlation and neighbor_expression.
        """
        neighbor_mean = self.compute_neighbor_expression(X, neighbor_indices)
        spatial_corr = self.compute_spatial_correlation(X, neighbor_mean)

        return {
            "spatial_correlation": spatial_corr,
            "neighbor_expression": neighbor_mean,
        }


class MetabolicExchangeScorer(nn.Module):
    """Scorer for metabolic exchange between cell types.

    Parameters
    ----------
    n_genes
        Number of metabolic genes.
    n_cell_types
        Number of cell types.
    n_hidden
        Number of hidden units.
    """

    def __init__(
        self,
        n_genes: int,
        n_cell_types: int,
        n_hidden: int = 64,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types

        # Exchange scoring network
        self.exchange_encoder = nn.Sequential(
            nn.Linear(n_genes * 2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        # Cell type interaction weights
        self.interaction_weights = nn.Parameter(
            torch.randn(n_cell_types, n_cell_types, n_hidden)
        )

        # Output head
        self.score_head = nn.Sequential(
            nn.Linear(n_hidden, n_genes),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sender_expr: Tensor,
        receiver_expr: Tensor,
        sender_type: Tensor,
        receiver_type: Tensor,
    ) -> Tensor:
        """Compute exchange scores.

        Parameters
        ----------
        sender_expr
            Sender cell expression.
        receiver_expr
            Receiver cell expression.
        sender_type
            Sender cell type indices.
        receiver_type
            Receiver cell type indices.

        Returns
        -------
        Exchange scores per gene.
        """
        # Concatenate sender and receiver expression
        combined = torch.cat([sender_expr, receiver_expr], dim=-1)
        h = self.exchange_encoder(combined)

        # Get cell type-specific interaction weights
        interaction = self.interaction_weights[sender_type.long(), receiver_type.long()]

        # Combine with interaction weights
        h = h * interaction

        # Compute scores
        scores = self.score_head(h)

        return scores
