"""Comprehensive tests for encoder architectures."""

import pytest
import torch

from spatialvi.nn import AttentionEncoder, GraphEncoder, SpatialEncoder


class TestSpatialEncoderInitialization:
    """Tests for SpatialEncoder initialization."""

    def test_basic_initialization(self, n_genes, n_latent, n_hidden):
        """Test basic encoder initialization."""
        encoder = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
        )

        assert encoder.n_output == n_latent

    def test_with_categorical_covariates(self, n_genes, n_latent, n_hidden):
        """Test initialization with categorical covariates."""
        encoder = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_cat_list=[2, 3],
            n_hidden=n_hidden,
        )

        assert encoder.encoder.n_cat_list == [2, 3]

    @pytest.mark.parametrize("aggregation", ["mean", "attention", "gat"])
    def test_aggregation_methods(self, n_genes, n_latent, n_hidden, aggregation):
        """Test different aggregation methods."""
        encoder = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
            aggregation=aggregation,
        )

        assert encoder.aggregation == aggregation

    def test_batch_norm_options(self, n_genes, n_latent, n_hidden):
        """Test batch normalization options."""
        encoder_bn = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
            use_batch_norm=True,
            use_layer_norm=False,
        )

        encoder_ln = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
            use_batch_norm=False,
            use_layer_norm=True,
        )

        assert encoder_bn is not None
        assert encoder_ln is not None


class TestSpatialEncoderForward:
    """Tests for SpatialEncoder forward pass."""

    def test_forward_without_spatial(self, batch_size, n_genes, n_latent, n_hidden, device):
        """Test forward pass without spatial context."""
        encoder = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
        ).to(device)

        x = torch.randn(batch_size, n_genes).to(device)

        output = encoder(x)

        # Output should be a Normal distribution
        assert hasattr(output, "loc")
        assert hasattr(output, "scale")
        assert output.loc.shape == (batch_size, n_latent)

    def test_forward_with_categorical(self, batch_size, n_genes, n_latent, n_hidden, device):
        """Test forward pass with categorical covariates."""
        encoder = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_cat_list=[2],
            n_hidden=n_hidden,
        ).to(device)

        x = torch.randn(batch_size, n_genes).to(device)
        batch_index = torch.randint(0, 2, (batch_size, 1)).to(device)

        output = encoder(x, batch_index)

        assert output.loc.shape == (batch_size, n_latent)


class TestSpatialEncoderForwardSpatial:
    """Tests for SpatialEncoder forward_spatial method."""

    def test_forward_spatial_mean_aggregation(self, batch_size, n_genes, n_latent, n_hidden, n_neighbors, device):
        """Test forward_spatial with mean aggregation."""
        encoder = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
            aggregation="mean",
        ).to(device)

        x = torch.randn(batch_size, n_genes).to(device)
        neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).to(device)
        neighbor_indices = torch.randint(0, batch_size, (batch_size, n_neighbors)).to(device)

        mean, var, z = encoder.forward_spatial(
            x,
            neighbor_expr=neighbor_expr,
            neighbor_indices=neighbor_indices,
        )

        assert mean.shape == (batch_size, n_latent)
        assert var.shape == (batch_size, n_latent)
        assert z.shape == (batch_size, n_latent)

    def test_forward_spatial_attention_aggregation(self, batch_size, n_genes, n_latent, n_hidden, n_neighbors, device):
        """Test forward_spatial with attention aggregation."""
        encoder = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
            spatial_hidden=32,
            aggregation="attention",
        ).to(device)

        x = torch.randn(batch_size, n_genes).to(device)
        neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).to(device)
        neighbor_indices = torch.randint(0, batch_size, (batch_size, n_neighbors)).to(device)

        mean, var, z = encoder.forward_spatial(
            x,
            neighbor_expr=neighbor_expr,
            neighbor_indices=neighbor_indices,
        )

        assert mean.shape == (batch_size, n_latent)
        assert z.shape == (batch_size, n_latent)

    def test_forward_spatial_gat_aggregation(self, batch_size, n_genes, n_latent, n_hidden, n_neighbors, device):
        """Test forward_spatial with GAT aggregation."""
        encoder = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
            spatial_hidden=32,
            aggregation="gat",
        ).to(device)

        x = torch.randn(batch_size, n_genes).to(device)
        neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).to(device)
        neighbor_indices = torch.randint(0, batch_size, (batch_size, n_neighbors)).to(device)

        mean, var, z = encoder.forward_spatial(
            x,
            neighbor_expr=neighbor_expr,
            neighbor_indices=neighbor_indices,
        )

        assert mean.shape == (batch_size, n_latent)

    def test_forward_spatial_without_neighbors(self, batch_size, n_genes, n_latent, n_hidden, device):
        """Test forward_spatial without neighbor information."""
        encoder = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
        ).to(device)

        x = torch.randn(batch_size, n_genes).to(device)

        mean, var, z = encoder.forward_spatial(x)

        assert mean.shape == (batch_size, n_latent)


class TestGraphEncoder:
    """Tests for GraphEncoder."""

    def test_initialization(self, n_genes, n_latent, n_hidden):
        """Test GraphEncoder initialization."""
        encoder = GraphEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
            n_layers=2,
        )

        assert encoder.n_layers == 2

    def test_forward(self, batch_size, n_genes, n_latent, n_hidden, edge_index, device):
        """Test GraphEncoder forward pass."""
        encoder = GraphEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
        ).to(device)

        x = torch.randn(batch_size, n_genes).to(device)
        edge_index = edge_index.to(device)

        mean, var = encoder(x, edge_index)

        assert mean.shape == (batch_size, n_latent)
        assert var.shape == (batch_size, n_latent)
        assert (var > 0).all()

    def test_multiple_layers(self, batch_size, n_genes, n_latent, n_hidden, edge_index, device):
        """Test GraphEncoder with multiple layers."""
        encoder = GraphEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
            n_layers=3,
        ).to(device)

        x = torch.randn(batch_size, n_genes).to(device)
        edge_index = edge_index.to(device)

        mean, var = encoder(x, edge_index)

        assert mean.shape == (batch_size, n_latent)


class TestAttentionEncoder:
    """Tests for AttentionEncoder."""

    def test_initialization(self, n_genes, n_latent, n_hidden):
        """Test AttentionEncoder initialization."""
        encoder = AttentionEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
            n_heads=4,
            n_layers=2,
        )

        assert encoder.transformer is not None

    def test_forward(self, batch_size, n_genes, n_latent, n_hidden, device):
        """Test AttentionEncoder forward pass."""
        seq_len = 10
        encoder = AttentionEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
        ).to(device)

        x = torch.randn(batch_size, seq_len, n_genes).to(device)

        mean, var = encoder(x)

        assert mean.shape == (batch_size, n_latent)
        assert var.shape == (batch_size, n_latent)

    def test_forward_with_mask(self, batch_size, n_genes, n_latent, n_hidden, device):
        """Test AttentionEncoder with attention mask."""
        seq_len = 10
        encoder = AttentionEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
        ).to(device)

        x = torch.randn(batch_size, seq_len, n_genes).to(device)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        mask[:, -3:] = True  # Mask last 3 positions

        mean, var = encoder(x, mask=mask)

        assert mean.shape == (batch_size, n_latent)


class TestEncoderGradients:
    """Tests for encoder gradient computation."""

    def test_spatial_encoder_gradients(self, batch_size, n_genes, n_latent, n_hidden, n_neighbors, device):
        """Test gradient flow through SpatialEncoder."""
        encoder = SpatialEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
        ).to(device)

        x = torch.randn(batch_size, n_genes, requires_grad=True).to(device)
        neighbor_expr = torch.randn(batch_size, n_neighbors, n_genes).to(device)
        neighbor_indices = torch.randint(0, batch_size, (batch_size, n_neighbors)).to(device)

        mean, var, z = encoder.forward_spatial(
            x,
            neighbor_expr=neighbor_expr,
            neighbor_indices=neighbor_indices,
        )

        loss = z.sum()
        loss.backward()

        assert x.grad is not None
        assert encoder.mean_layer.weight.grad is not None

    def test_graph_encoder_gradients(self, batch_size, n_genes, n_latent, n_hidden, edge_index, device):
        """Test gradient flow through GraphEncoder."""
        encoder = GraphEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
        ).to(device)

        x = torch.randn(batch_size, n_genes, requires_grad=True).to(device)
        edge_index = edge_index.to(device)

        mean, var = encoder(x, edge_index)

        loss = mean.sum()
        loss.backward()

        assert x.grad is not None

    def test_attention_encoder_gradients(self, batch_size, n_genes, n_latent, n_hidden, device):
        """Test gradient flow through AttentionEncoder."""
        seq_len = 5
        encoder = AttentionEncoder(
            n_input=n_genes,
            n_output=n_latent,
            n_hidden=n_hidden,
        ).to(device)

        x = torch.randn(batch_size, seq_len, n_genes, requires_grad=True).to(device)

        mean, var = encoder(x)

        loss = mean.sum()
        loss.backward()

        assert x.grad is not None
