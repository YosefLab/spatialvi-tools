"""Nolan module for self-supervised niche detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class NolanModule(nn.Module):
    """Module for NOLAN self-supervised niche learning.

    This module implements the core components of NOLAN:
    - Spatial context encoder
    - Niche embedding network
    - Contrastive learning objective

    Parameters
    ----------
    input_dim
        Dimension of input embeddings.
    hidden_dim
        Hidden layer dimension.
    output_dim
        Dimension of niche embeddings.
    n_layers
        Number of hidden layers.
    dropout_rate
        Dropout rate.
    use_batch_norm
        Whether to use batch normalization.
    temperature
        Temperature for contrastive loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 50,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature

        # Build encoder
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Prediction head for contrastive learning
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, x: Tensor) -> Tensor:
        """Encode input embeddings.

        Parameters
        ----------
        x
            Input embeddings of shape (batch, input_dim).

        Returns
        -------
        Encoded features of shape (batch, hidden_dim).
        """
        return self.encoder(x)

    def project(self, h: Tensor) -> Tensor:
        """Project to niche embedding space.

        Parameters
        ----------
        h
            Encoded features.

        Returns
        -------
        Projected embeddings of shape (batch, output_dim).
        """
        return self.projection_head(h)

    def forward(
        self,
        x: Tensor,
        x_neighbor: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x
            Input embeddings of shape (batch, input_dim).
        x_neighbor
            Neighbor embeddings for contrastive learning.

        Returns
        -------
        Dictionary with niche_embedding and optionally contrastive outputs.
        """
        # Encode and project
        h = self.encode(x)
        z = self.project(h)

        outputs = {
            "niche_embedding": z,
            "hidden": h,
        }

        if x_neighbor is not None:
            # Encode neighbor
            h_neighbor = self.encode(x_neighbor)
            z_neighbor = self.project(h_neighbor)

            # Prediction
            p = self.predictor(z)
            p_neighbor = self.predictor(z_neighbor)

            outputs.update(
                {
                    "z_neighbor": z_neighbor,
                    "prediction": p,
                    "prediction_neighbor": p_neighbor,
                }
            )

        return outputs

    def contrastive_loss(
        self,
        z: Tensor,
        z_pos: Tensor,
        z_neg: Tensor | None = None,
    ) -> Tensor:
        """Compute contrastive loss.

        Parameters
        ----------
        z
            Anchor embeddings.
        z_pos
            Positive (neighbor) embeddings.
        z_neg
            Optional negative embeddings.

        Returns
        -------
        Contrastive loss value.
        """
        # Normalize
        z = nn.functional.normalize(z, dim=-1)
        z_pos = nn.functional.normalize(z_pos, dim=-1)

        # Positive similarity
        pos_sim = torch.sum(z * z_pos, dim=-1) / self.temperature

        if z_neg is not None:
            z_neg = nn.functional.normalize(z_neg, dim=-1)
            neg_sim = torch.mm(z, z_neg.t()) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
            loss = nn.functional.cross_entropy(logits, labels)
        else:
            # InfoNCE with in-batch negatives
            all_sim = torch.mm(z, z_pos.t()) / self.temperature
            labels = torch.arange(z.shape[0], device=z.device)
            loss = nn.functional.cross_entropy(all_sim, labels)

        return loss


class NicheClusteringHead(nn.Module):
    """Clustering head for discrete niche assignment.

    Parameters
    ----------
    input_dim
        Input embedding dimension.
    n_clusters
        Number of niche clusters.
    soft_assignment
        Whether to use soft cluster assignment.
    """

    def __init__(
        self,
        input_dim: int,
        n_clusters: int = 20,
        soft_assignment: bool = True,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.soft_assignment = soft_assignment

        # Cluster centroids
        self.centroids = nn.Parameter(torch.randn(n_clusters, input_dim))

        # Optional learnable temperature
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        """Assign cells to niches.

        Parameters
        ----------
        z
            Niche embeddings of shape (batch, input_dim).

        Returns
        -------
        Dictionary with assignments and distances.
        """
        # Compute distances to centroids
        z_norm = nn.functional.normalize(z, dim=-1)
        c_norm = nn.functional.normalize(self.centroids, dim=-1)

        distances = torch.cdist(z_norm, c_norm, p=2)

        temperature = torch.exp(self.log_temperature)

        if self.soft_assignment:
            # Soft assignment via softmax
            assignments = torch.softmax(-distances / temperature, dim=-1)
        else:
            # Hard assignment
            assignments = torch.zeros_like(distances)
            assignments.scatter_(1, distances.argmin(dim=-1, keepdim=True), 1)

        return {
            "assignments": assignments,
            "distances": distances,
            "cluster_ids": distances.argmin(dim=-1),
        }
