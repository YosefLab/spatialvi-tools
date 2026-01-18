"""Tests for neural network components."""

import torch
import pytest


class TestEncoders:
    """Tests for encoder architectures."""

    
    def test_spatial_encoder(self, random_expression, n_latent, device):
        """Test spatial encoder forward pass."""
        from spatialvi.nn import SpatialEncoder

        batch_size, n_genes = random_expression.shape
        encoder = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=32,
            n_layers=1,
        ).to(device)

        x = random_expression.to(device)
        output = encoder(x)

        assert output.loc.shape == (batch_size, n_latent)
        assert output.scale.shape == (batch_size, n_latent)

    def test_spatial_encoder_with_neighbors(
        self, random_expression, random_neighbor_expression, n_latent, device
    ):
        """Test spatial encoder with neighbor context."""
        from spatialvi.nn import SpatialEncoder

        batch_size, n_genes = random_expression.shape
        encoder = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=32,
            n_layers=1,
            aggregation="mean",
        ).to(device)

        x = random_expression.to(device)
        neighbor_x = random_neighbor_expression.to(device)

        mean, var, z = encoder.forward_spatial(
            x,
            neighbor_expr=neighbor_x,
        )

        assert mean.shape == (batch_size, n_latent)
        assert var.shape == (batch_size, n_latent)
        assert z.shape == (batch_size, n_latent)

    def test_graph_encoder(self, n_genes, n_latent, device):
        """Test graph encoder forward pass."""
        from spatialvi.nn import GraphEncoder

        n_nodes = 100
        encoder = GraphEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=32,
            n_layers=2,
        ).to(device)

        x = torch.randn(n_nodes, n_genes).to(device)
        # Random edges
        edge_index = torch.randint(0, n_nodes, (2, 500)).to(device)

        mean, var = encoder(x, edge_index)

        assert mean.shape == (n_nodes, n_latent)
        assert var.shape == (n_nodes, n_latent)


class TestAttention:
    """Tests for attention mechanisms."""

    def test_spatial_attention(self, batch_size, embed_dim, n_neighbors, device):
        """Test spatial attention layer."""
        from spatialvi.nn import SpatialAttention

        attention = SpatialAttention(
            embed_dim=embed_dim,
            n_heads=4,
            dropout=0.1,
        ).to(device)

        query = torch.randn(batch_size, 1, embed_dim).to(device)
        key = torch.randn(batch_size, n_neighbors, embed_dim).to(device)
        value = key
        distances = torch.rand(batch_size, 1, n_neighbors).to(device)

        output, weights = attention(query, key, value, distances=distances)

        assert output.shape == (batch_size, 1, embed_dim)
        assert weights.shape[0] == batch_size

    def test_neighbor_attention(self, random_expression, random_neighbor_expression, device):
        """Test neighbor attention aggregation."""
        from spatialvi.nn import NeighborAttention

        batch_size, n_input = random_expression.shape
        attention = NeighborAttention(
            input_dim=n_input,
            hidden_dim=32,
            n_heads=4,
        ).to(device)

        x = random_expression.to(device)
        neighbor_x = random_neighbor_expression.to(device)

        output = attention(x, neighbor_x)

        assert output.shape == (batch_size, n_input)


class TestLayers:
    """Tests for custom layers."""

    def test_spatial_conv(self, n_genes, device):
        """Test spatial convolution layer."""
        from spatialvi.nn import SpatialConv

        n_nodes = 100
        n_neighbors = 10
        out_channels = 32

        conv = SpatialConv(
            in_channels=n_genes,
            out_channels=out_channels,
            n_neighbors=n_neighbors,
        ).to(device)

        x = torch.randn(n_nodes, n_genes).to(device)
        neighbor_indices = torch.randint(0, n_nodes, (n_nodes, n_neighbors)).to(device)
        neighbor_distances = torch.rand(n_nodes, n_neighbors).to(device)

        output = conv(x, neighbor_indices, neighbor_distances)

        assert output.shape == (n_nodes, out_channels)

    def test_positional_encoding(self, batch_size, device):
        """Test positional encoding."""
        from spatialvi.nn import PositionalEncoding

        d_model = 64
        pe = PositionalEncoding(d_model=d_model, n_dims=2).to(device)

        coords = torch.rand(batch_size, 2).to(device) * 100

        encoding = pe(coords, normalize=True)

        assert encoding.shape == (batch_size, d_model)
