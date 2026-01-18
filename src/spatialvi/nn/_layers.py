"""Custom neural network layers for spatial models."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SpatialConv(nn.Module):
    """Spatial convolution layer using k-nearest neighbors.

    This layer performs convolution over spatial neighbors,
    aggregating information from nearby cells.

    Parameters
    ----------
    in_channels
        Number of input channels.
    out_channels
        Number of output channels.
    n_neighbors
        Number of neighbors to use.
    aggregation
        Aggregation method ("mean", "max", "sum", "attention").
    use_edge_features
        Whether to use edge features (distances).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_neighbors: int = 20,
        aggregation: Literal["mean", "max", "sum", "attention"] = "mean",
        use_edge_features: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_neighbors = n_neighbors
        self.aggregation = aggregation
        self.use_edge_features = use_edge_features

        # Feature transformation
        self.linear = nn.Linear(in_channels, out_channels)

        # Edge feature processing
        if use_edge_features:
            self.edge_mlp = nn.Sequential(
                nn.Linear(1, out_channels // 4),
                nn.ReLU(),
                nn.Linear(out_channels // 4, out_channels),
            )

        # Attention weights
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(out_channels * 2, out_channels),
                nn.LeakyReLU(0.2),
                nn.Linear(out_channels, 1),
            )

        # Output
        self.norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_distances: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Node features of shape (n_nodes, in_channels).
        neighbor_indices
            Neighbor indices of shape (n_nodes, n_neighbors).
        neighbor_distances
            Distances to neighbors of shape (n_nodes, n_neighbors).

        Returns
        -------
        Output features of shape (n_nodes, out_channels).
        """
        n_neighbors = neighbor_indices.shape[1]

        # Transform features
        h = self.linear(x)

        # Gather neighbor features
        neighbor_h = h[neighbor_indices.long()]  # (n_nodes, n_neighbors, out_channels)

        # Apply edge features
        if self.use_edge_features and neighbor_distances is not None:
            edge_feat = self.edge_mlp(neighbor_distances.unsqueeze(-1))
            neighbor_h = neighbor_h + edge_feat

        # Aggregate
        if self.aggregation == "mean":
            out = neighbor_h.mean(dim=1)
        elif self.aggregation == "max":
            out = neighbor_h.max(dim=1)[0]
        elif self.aggregation == "sum":
            out = neighbor_h.sum(dim=1)
        elif self.aggregation == "attention":
            # Compute attention scores
            h_expanded = h.unsqueeze(1).expand(-1, n_neighbors, -1)
            concat = torch.cat([h_expanded, neighbor_h], dim=-1)
            attn_scores = self.attention(concat).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = (attn_weights.unsqueeze(-1) * neighbor_h).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Residual + norm
        out = self.norm(h + out)

        return out


class GraphConv(nn.Module):
    """Graph convolution layer.

    Parameters
    ----------
    in_channels
        Number of input channels.
    out_channels
        Number of output channels.
    bias
        Whether to use bias.
    normalize
        Whether to normalize by neighbor count.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Node features of shape (n_nodes, in_channels).
        edge_index
            Edge indices of shape (2, n_edges).
        edge_weight
            Optional edge weights of shape (n_edges,).

        Returns
        -------
        Output features of shape (n_nodes, out_channels).
        """
        n_nodes = x.shape[0]
        src, dst = edge_index

        # Transform
        h = x @ self.weight

        # Message passing
        if edge_weight is not None:
            messages = h[src] * edge_weight.unsqueeze(-1)
        else:
            messages = h[src]

        # Aggregate
        out = torch.zeros(n_nodes, self.out_channels, device=x.device)
        out.index_add_(0, dst, messages)

        # Normalize
        if self.normalize:
            deg = torch.zeros(n_nodes, device=x.device)
            deg.index_add_(0, dst, torch.ones(src.shape[0], device=x.device))
            deg = deg.clamp(min=1)
            out = out / deg.unsqueeze(-1)

        if self.bias is not None:
            out = out + self.bias

        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for spatial coordinates.

    Encodes 2D or 3D spatial coordinates using sinusoidal
    or learnable positional encodings.

    Parameters
    ----------
    d_model
        Dimension of the positional encoding.
    n_dims
        Number of spatial dimensions (2 or 3).
    max_len
        Maximum length for each dimension.
    encoding_type
        Type of encoding ("sinusoidal" or "learned").
    """

    def __init__(
        self,
        d_model: int,
        n_dims: int = 2,
        max_len: int = 10000,
        encoding_type: Literal["sinusoidal", "learned"] = "sinusoidal",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_dims = n_dims
        self.encoding_type = encoding_type

        if encoding_type == "sinusoidal":
            # Create sinusoidal encoding
            self.register_buffer(
                "pe_weights",
                self._create_sinusoidal_weights(d_model, n_dims, max_len),
            )
        else:
            # Learnable MLP
            self.encoder = nn.Sequential(
                nn.Linear(n_dims, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

    def _create_sinusoidal_weights(
        self,
        d_model: int,
        n_dims: int,
        max_len: int,
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding weights."""
        d_per_dim = d_model // n_dims

        pe = torch.zeros(max_len, d_per_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_per_dim, 2).float() * (-math.log(10000.0) / d_per_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(
        self,
        coords: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        coords
            Spatial coordinates of shape (batch, n_dims).
        normalize
            Whether to normalize coordinates to [0, max_len).

        Returns
        -------
        Positional encodings of shape (batch, d_model).
        """
        if self.encoding_type == "learned":
            return self.encoder(coords)

        # Sinusoidal encoding
        if normalize:
            # Normalize to [0, max_len)
            coords_min = coords.min(dim=0, keepdim=True)[0]
            coords_max = coords.max(dim=0, keepdim=True)[0]
            coords = (coords - coords_min) / (coords_max - coords_min + 1e-8)
            coords = coords * (self.pe_weights.shape[0] - 1)

        coords = coords.long().clamp(0, self.pe_weights.shape[0] - 1)

        # Look up encodings for each dimension
        encodings = []
        for i in range(self.n_dims):
            encodings.append(self.pe_weights[coords[:, i]])

        return torch.cat(encodings, dim=-1)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.

    Modulates features based on conditioning information.

    Parameters
    ----------
    feature_dim
        Dimension of features to modulate.
    condition_dim
        Dimension of conditioning information.
    hidden_dim
        Hidden dimension for FiLM generators.
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or condition_dim

        self.gamma_gen = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.beta_gen = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Features to modulate of shape (..., feature_dim).
        condition
            Conditioning information of shape (..., condition_dim).

        Returns
        -------
        Modulated features of shape (..., feature_dim).
        """
        gamma = self.gamma_gen(condition)
        beta = self.beta_gen(condition)
        return gamma * x + beta


class MixedLinear(nn.Module):
    """Linear layer with mixture of experts.

    Parameters
    ----------
    in_features
        Number of input features.
    out_features
        Number of output features.
    n_experts
        Number of expert networks.
    bias
        Whether to use bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_experts: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.n_experts = n_experts

        # Expert weights
        self.experts = nn.Parameter(torch.empty(n_experts, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(n_experts, out_features))
        else:
            self.register_parameter("bias", None)

        # Gating network
        self.gate = nn.Linear(in_features, n_experts)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.n_experts):
            nn.init.xavier_uniform_(self.experts[i])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor of shape (batch, in_features).

        Returns
        -------
        Output tensor of shape (batch, out_features).
        """
        # Compute gating weights
        gate_scores = self.gate(x)
        gate_weights = F.softmax(gate_scores, dim=-1)  # (batch, n_experts)

        # Compute expert outputs
        # x: (batch, in_features)
        # experts: (n_experts, in_features, out_features)
        expert_outputs = torch.einsum("bi,eio->beo", x, self.experts)

        if self.bias is not None:
            expert_outputs = expert_outputs + self.bias.unsqueeze(0)

        # Weighted combination
        # gate_weights: (batch, n_experts)
        # expert_outputs: (batch, n_experts, out_features)
        output = torch.einsum("be,beo->bo", gate_weights, expert_outputs)

        return output
