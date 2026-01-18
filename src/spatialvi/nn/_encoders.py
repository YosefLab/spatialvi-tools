"""Encoder architectures for spatial models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from scvi.nn import FCLayers
from torch import nn
from torch.distributions import Normal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SpatialEncoder(nn.Module):
    """Encoder that incorporates spatial context.

    This encoder combines cell-intrinsic features with information
    from spatially neighboring cells.

    Parameters
    ----------
    n_input
        Number of input features.
    n_output
        Dimensionality of the output (latent space).
    n_cat_list
        List of number of categories for categorical variables.
    n_layers
        Number of hidden layers.
    n_hidden
        Number of nodes per hidden layer.
    dropout_rate
        Dropout rate.
    use_batch_norm
        Whether to use batch normalization.
    use_layer_norm
        Whether to use layer normalization.
    spatial_hidden
        Hidden dimension for spatial encoding.
    n_neighbors
        Number of neighbors to use.
    aggregation
        Neighbor aggregation method.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: list[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        spatial_hidden: int = 32,
        n_neighbors: int = 20,
        aggregation: Literal["mean", "attention", "gat"] = "mean",
    ):
        super().__init__()
        self.n_output = n_output
        self.aggregation = aggregation
        self.spatial_hidden = spatial_hidden

        # Main encoder for cell-intrinsic features
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # Spatial context encoder
        self.spatial_encoder = FCLayers(
            n_in=n_input,
            n_out=spatial_hidden,
            n_layers=1,
            n_hidden=spatial_hidden,
            dropout_rate=dropout_rate,
        )

        # Attention for neighbor aggregation
        if aggregation == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=spatial_hidden,
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True,
            )
        elif aggregation == "gat":
            self.gat_attention = nn.Sequential(
                nn.Linear(spatial_hidden * 2, spatial_hidden),
                nn.LeakyReLU(0.2),
                nn.Linear(spatial_hidden, 1),
            )

        # Combine intrinsic and spatial
        self.combiner = FCLayers(
            n_in=n_hidden + spatial_hidden,
            n_out=n_hidden,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # Output layers for mean and variance
        self.mean_layer = nn.Linear(n_hidden, n_output)
        self.var_layer = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        x: torch.Tensor,
        *cat_list: torch.Tensor,
    ) -> Normal:
        """Forward pass without spatial context.

        Parameters
        ----------
        x
            Input tensor.
        cat_list
            Categorical covariates.

        Returns
        -------
        Normal distribution for latent representation.
        """
        h = self.encoder(x, *cat_list)
        spatial_h = torch.zeros(x.shape[0], self.spatial_hidden, device=x.device)
        combined = torch.cat([h, spatial_h], dim=-1)
        h_final = self.combiner(combined)

        mean = self.mean_layer(h_final)
        var = torch.exp(self.var_layer(h_final)) + 1e-4

        return Normal(mean, torch.sqrt(var))

    def forward_spatial(
        self,
        x: torch.Tensor,
        cat_list: list[torch.Tensor] | None = None,
        spatial_coords: torch.Tensor | None = None,
        neighbor_indices: torch.Tensor | None = None,
        neighbor_expr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with spatial context.

        Parameters
        ----------
        x
            Input tensor of shape (n_cells, n_input).
        cat_list
            List of categorical covariates.
        spatial_coords
            Spatial coordinates (not used in current implementation).
        neighbor_indices
            Indices of neighboring cells.
        neighbor_expr
            Expression of neighboring cells.

        Returns
        -------
        Tuple of (mean, variance, sample) for latent representation.
        """
        cat_list = cat_list or []

        # Encode cell-intrinsic features
        h = self.encoder(x, *cat_list)

        # Encode spatial context
        spatial_h = self.spatial_encoder(x)

        if neighbor_indices is not None and neighbor_expr is not None:
            # Get neighbor embeddings
            neighbor_h = self.spatial_encoder(neighbor_expr.view(-1, neighbor_expr.shape[-1]))
            neighbor_h = neighbor_h.view(*neighbor_expr.shape[:2], -1)

            # Aggregate neighbors
            if self.aggregation == "mean":
                spatial_context = neighbor_h.mean(dim=1)
            elif self.aggregation == "attention":
                query = spatial_h.unsqueeze(1)
                attn_out, _ = self.attention(query, neighbor_h, neighbor_h)
                spatial_context = attn_out.squeeze(1)
            elif self.aggregation == "gat":
                # Graph attention
                n_neighbors = neighbor_h.shape[1]
                query_expanded = spatial_h.unsqueeze(1).expand(-1, n_neighbors, -1)
                concat = torch.cat([query_expanded, neighbor_h], dim=-1)
                attn_scores = self.gat_attention(concat).squeeze(-1)
                attn_weights = torch.softmax(attn_scores, dim=-1)
                spatial_context = (attn_weights.unsqueeze(-1) * neighbor_h).sum(dim=1)
            else:
                spatial_context = neighbor_h.mean(dim=1)

            spatial_h = spatial_h + spatial_context

        # Combine
        combined = torch.cat([h, spatial_h], dim=-1)
        h_final = self.combiner(combined)

        # Output
        mean = self.mean_layer(h_final)
        var = torch.exp(self.var_layer(h_final)) + 1e-4
        std = torch.sqrt(var)

        # Sample
        eps = torch.randn_like(mean)
        z = mean + std * eps

        return mean, var, z


class GraphEncoder(nn.Module):
    """Graph-based encoder using message passing.

    Parameters
    ----------
    n_input
        Number of input features.
    n_output
        Dimensionality of the output.
    n_hidden
        Number of hidden nodes.
    n_layers
        Number of graph conv layers.
    dropout_rate
        Dropout rate.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers

        # Initial projection
        self.input_proj = nn.Linear(n_input, n_hidden)

        # Graph convolution layers
        self.convs = nn.ModuleList([GraphConvLayer(n_hidden, n_hidden, dropout_rate) for _ in range(n_layers)])

        # Output layers
        self.mean_layer = nn.Linear(n_hidden, n_output)
        self.var_layer = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x
            Node features of shape (n_nodes, n_input).
        edge_index
            Edge indices of shape (2, n_edges).

        Returns
        -------
        Tuple of (mean, variance) for latent representation.
        """
        h = self.input_proj(x)

        for conv in self.convs:
            h = conv(h, edge_index)

        mean = self.mean_layer(h)
        var = torch.exp(self.var_layer(h)) + 1e-4

        return mean, var


class GraphConvLayer(nn.Module):
    """Simple graph convolution layer."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.linear = nn.Linear(n_input, n_output)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(n_output)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        # Simple aggregation (mean of neighbors)
        src, dst = edge_index
        n_nodes = x.shape[0]

        # Aggregate
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])

        # Count neighbors
        count = torch.zeros(n_nodes, device=x.device)
        count.index_add_(0, dst, torch.ones(src.shape[0], device=x.device))
        count = count.clamp(min=1)

        agg = agg / count.unsqueeze(-1)

        # Transform
        h = self.linear(agg)
        h = self.dropout(h)
        h = self.norm(h)
        h = torch.relu(h)

        return h + x  # Residual connection


class AttentionEncoder(nn.Module):
    """Transformer-based encoder with self-attention.

    Parameters
    ----------
    n_input
        Number of input features.
    n_output
        Dimensionality of the output.
    n_hidden
        Number of hidden nodes.
    n_heads
        Number of attention heads.
    n_layers
        Number of transformer layers.
    dropout_rate
        Dropout rate.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(n_input, n_hidden)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_hidden,
            nhead=n_heads,
            dim_feedforward=n_hidden * 4,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output layers
        self.mean_layer = nn.Linear(n_hidden, n_output)
        self.var_layer = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor of shape (batch, seq_len, n_input).
        mask
            Attention mask.

        Returns
        -------
        Tuple of (mean, variance) for latent representation.
        """
        h = self.input_proj(x)
        h = self.transformer(h, src_key_padding_mask=mask)

        # Pool over sequence
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            h = (h * (1 - mask_expanded)).sum(dim=1) / (1 - mask_expanded).sum(dim=1)
        else:
            h = h.mean(dim=1)

        mean = self.mean_layer(h)
        var = torch.exp(self.var_layer(h)) + 1e-4

        return mean, var
