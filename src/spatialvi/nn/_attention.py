"""Attention mechanisms for spatial models."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism that weights neighbors by distance.

    This attention mechanism uses spatial distances to modulate
    attention weights between cells.

    Parameters
    ----------
    embed_dim
        Embedding dimension.
    n_heads
        Number of attention heads.
    dropout
        Dropout rate.
    distance_decay
        Decay factor for distance weighting.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        distance_decay: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.distance_decay = distance_decay

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        distances: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        query
            Query tensor of shape (batch, seq_q, embed_dim).
        key
            Key tensor of shape (batch, seq_k, embed_dim).
        value
            Value tensor of shape (batch, seq_k, embed_dim).
        distances
            Spatial distances of shape (batch, seq_q, seq_k).
        mask
            Attention mask of shape (batch, seq_q, seq_k).

        Returns
        -------
        Tuple of (output, attention_weights).
        """
        batch_size = query.shape[0]

        # Project
        q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.n_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.n_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)  # (batch, n_heads, seq_q, head_dim)
        k = k.transpose(1, 2)  # (batch, n_heads, seq_k, head_dim)
        v = v.transpose(1, 2)  # (batch, n_heads, seq_k, head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply distance-based bias
        if distances is not None:
            distance_bias = -self.distance_decay * distances.unsqueeze(1)
            attn_scores = attn_scores + distance_bias

        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(output)

        return output, attn_weights.mean(dim=1)


class CrossAttention(nn.Module):
    """Cross-attention between two modalities or cell types.

    Useful for cell-cell interaction modeling where one cell type
    attends to another.

    Parameters
    ----------
    embed_dim
        Embedding dimension.
    n_heads
        Number of attention heads.
    dropout
        Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        query
            Query tensor of shape (batch, seq_q, embed_dim).
        context
            Context tensor of shape (batch, seq_c, embed_dim).
        context_mask
            Mask for context of shape (batch, seq_c).

        Returns
        -------
        Output tensor of shape (batch, seq_q, embed_dim).
        """
        # Cross-attention
        attn_out, _ = self.attention(
            query,
            context,
            context,
            key_padding_mask=context_mask,
        )
        x = self.norm1(query + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class NeighborAttention(nn.Module):
    """Attention mechanism for aggregating neighbor information.

    Specifically designed for spatial neighbor aggregation where
    each cell attends to its k-nearest neighbors.

    Parameters
    ----------
    input_dim
        Input feature dimension.
    hidden_dim
        Hidden dimension for attention computation.
    n_heads
        Number of attention heads.
    dropout
        Dropout rate.
    use_distance_weights
        Whether to use distance-based attention weights.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_distance_weights: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.use_distance_weights = use_distance_weights

        # Projections
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)

        # Distance embedding
        if use_distance_weights:
            self.distance_embed = nn.Sequential(
                nn.Linear(1, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
            )

        # Output
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        neighbor_x: torch.Tensor,
        neighbor_distances: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Cell features of shape (batch, input_dim).
        neighbor_x
            Neighbor features of shape (batch, n_neighbors, input_dim).
        neighbor_distances
            Distances to neighbors of shape (batch, n_neighbors).

        Returns
        -------
        Aggregated features of shape (batch, input_dim).
        """
        batch_size, n_neighbors, _ = neighbor_x.shape

        # Project
        query = self.query_proj(x).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.key_proj(neighbor_x)  # (batch, n_neighbors, hidden)
        values = self.value_proj(neighbor_x)  # (batch, n_neighbors, hidden)

        # Compute attention scores
        scale = math.sqrt(self.hidden_dim)
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / scale
        attn_scores = attn_scores.squeeze(1)  # (batch, n_neighbors)

        # Add distance-based bias
        if self.use_distance_weights and neighbor_distances is not None:
            dist_bias = self.distance_embed(neighbor_distances.unsqueeze(-1))
            dist_bias = dist_bias.squeeze(-1)  # (batch, n_neighbors)
            attn_scores = attn_scores + dist_bias

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate
        context = torch.matmul(attn_weights.unsqueeze(1), values)
        context = context.squeeze(1)  # (batch, hidden)

        # Output
        output = self.output_proj(context)
        output = self.dropout(output)

        # Residual connection
        output = self.norm(x + output)

        return output


class GATLayer(nn.Module):
    """Graph Attention Layer.

    Parameters
    ----------
    in_features
        Number of input features.
    out_features
        Number of output features.
    n_heads
        Number of attention heads.
    dropout
        Dropout rate.
    alpha
        Negative slope for LeakyReLU.
    concat
        Whether to concatenate heads or average.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat

        self.W = nn.Parameter(torch.empty(n_heads, in_features, out_features))
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_features, 1))

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Node features of shape (n_nodes, in_features).
        edge_index
            Edge indices of shape (2, n_edges).

        Returns
        -------
        Output features.
        """
        n_nodes = x.shape[0]
        src, dst = edge_index

        # Transform features for each head
        h = torch.einsum("ni,hio->nho", x, self.W)  # (n_nodes, n_heads, out_features)

        # Compute attention coefficients
        h_src = h[src]  # (n_edges, n_heads, out_features)
        h_dst = h[dst]  # (n_edges, n_heads, out_features)

        # Concatenate source and destination features
        edge_h = torch.cat([h_src, h_dst], dim=-1)  # (n_edges, n_heads, 2*out_features)
        attn_scores = torch.einsum("eho,ho->eh", edge_h, self.a.squeeze(-1))
        attn_scores = self.leaky_relu(attn_scores)

        # Softmax over neighbors
        attn_scores = attn_scores - attn_scores.max()
        attn_exp = attn_scores.exp()

        sum_exp = torch.zeros(n_nodes, self.n_heads, device=x.device)
        sum_exp.index_add_(0, dst, attn_exp)
        attn_weights = attn_exp / (sum_exp[dst] + 1e-8)
        attn_weights = self.dropout(attn_weights)

        # Aggregate
        out = torch.zeros(n_nodes, self.n_heads, self.out_features, device=x.device)
        weighted_h = attn_weights.unsqueeze(-1) * h_src
        out.index_add_(0, dst, weighted_h)

        if self.concat:
            return out.view(n_nodes, -1)  # (n_nodes, n_heads * out_features)
        else:
            return out.mean(dim=1)  # (n_nodes, out_features)
