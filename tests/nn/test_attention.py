"""Tests for attention mechanisms."""

import pytest
import torch

from spatialvi.nn import SpatialAttention, CrossAttention, NeighborAttention, GATLayer


class TestSpatialAttention:
    """Tests for SpatialAttention module."""

    @pytest.fixture
    def attention(self, embed_dim):
        """Create a SpatialAttention instance."""
        return SpatialAttention(embed_dim=embed_dim, n_heads=4, dropout=0.1)

    def test_initialization(self, attention, embed_dim):
        """Test that SpatialAttention initializes correctly."""
        assert attention.embed_dim == embed_dim
        assert attention.n_heads == 4
        assert attention.head_dim == embed_dim // 4

    def test_forward_shape(self, attention, batch_size, embed_dim):
        """Test forward pass output shapes."""
        seq_len = 16
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)

        output, attn_weights = attention(query, key, value)

        assert output.shape == (batch_size, seq_len, embed_dim)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)

    def test_forward_with_distances(self, attention, batch_size, embed_dim):
        """Test forward pass with distance matrix."""
        seq_len = 16
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        distances = torch.rand(batch_size, seq_len, seq_len)

        output, attn_weights = attention(query, key, value, distances=distances)

        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_forward_with_mask(self, attention, batch_size, embed_dim):
        """Test forward pass with attention mask."""
        seq_len = 16
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool)
        mask[:, :, -1] = True  # Mask last position

        output, attn_weights = attention(query, key, value, mask=mask)

        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_embed_dim_divisibility(self):
        """Test that embed_dim must be divisible by n_heads."""
        with pytest.raises(AssertionError):
            SpatialAttention(embed_dim=65, n_heads=4)


class TestCrossAttention:
    """Tests for CrossAttention module."""

    @pytest.fixture
    def cross_attention(self, embed_dim):
        """Create a CrossAttention instance."""
        return CrossAttention(embed_dim=embed_dim, n_heads=4, dropout=0.1)

    def test_initialization(self, cross_attention, embed_dim):
        """Test that CrossAttention initializes correctly."""
        assert cross_attention.attention.embed_dim == embed_dim

    def test_forward_shape(self, cross_attention, batch_size, embed_dim):
        """Test forward pass output shapes."""
        seq_q, seq_c = 16, 32
        query = torch.randn(batch_size, seq_q, embed_dim)
        context = torch.randn(batch_size, seq_c, embed_dim)

        output = cross_attention(query, context)

        assert output.shape == (batch_size, seq_q, embed_dim)

    def test_forward_with_mask(self, cross_attention, batch_size, embed_dim):
        """Test forward pass with context mask."""
        seq_q, seq_c = 16, 32
        query = torch.randn(batch_size, seq_q, embed_dim)
        context = torch.randn(batch_size, seq_c, embed_dim)
        context_mask = torch.zeros(batch_size, seq_c, dtype=torch.bool)
        context_mask[:, -5:] = True  # Mask last 5 positions

        output = cross_attention(query, context, context_mask=context_mask)

        assert output.shape == (batch_size, seq_q, embed_dim)


class TestNeighborAttention:
    """Tests for NeighborAttention module."""

    @pytest.fixture
    def neighbor_attention(self):
        """Create a NeighborAttention instance."""
        return NeighborAttention(
            input_dim=64,
            hidden_dim=64,
            n_heads=4,
            dropout=0.1,
            use_distance_weights=True,
        )

    def test_initialization(self, neighbor_attention):
        """Test that NeighborAttention initializes correctly."""
        assert neighbor_attention.input_dim == 64
        assert neighbor_attention.hidden_dim == 64

    def test_forward_shape(self, neighbor_attention, batch_size, n_neighbors):
        """Test forward pass output shapes."""
        input_dim = 64
        x = torch.randn(batch_size, input_dim)
        neighbor_x = torch.randn(batch_size, n_neighbors, input_dim)
        neighbor_distances = torch.rand(batch_size, n_neighbors)

        output = neighbor_attention(x, neighbor_x, neighbor_distances)

        assert output.shape == (batch_size, input_dim)

    def test_forward_without_distances(self, batch_size, n_neighbors):
        """Test forward pass without distance weights."""
        attention = NeighborAttention(
            input_dim=64,
            hidden_dim=64,
            use_distance_weights=False,
        )
        x = torch.randn(batch_size, 64)
        neighbor_x = torch.randn(batch_size, n_neighbors, 64)

        output = attention(x, neighbor_x)

        assert output.shape == (batch_size, 64)


class TestGATLayer:
    """Tests for GATLayer module."""

    @pytest.fixture
    def gat_layer(self):
        """Create a GATLayer instance."""
        return GATLayer(
            in_features=64,
            out_features=32,
            n_heads=4,
            dropout=0.1,
            concat=True,
        )

    def test_initialization(self, gat_layer):
        """Test that GATLayer initializes correctly."""
        assert gat_layer.in_features == 64
        assert gat_layer.out_features == 32
        assert gat_layer.n_heads == 4

    def test_forward_shape_concat(self, gat_layer):
        """Test forward pass output shapes with concatenation."""
        n_nodes = 100
        n_edges = 500
        x = torch.randn(n_nodes, 64)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))

        output = gat_layer(x, edge_index)

        # With concat=True, output should be n_heads * out_features
        assert output.shape == (n_nodes, 4 * 32)

    def test_forward_shape_mean(self):
        """Test forward pass output shapes without concatenation."""
        gat_layer = GATLayer(
            in_features=64,
            out_features=32,
            n_heads=4,
            concat=False,
        )
        n_nodes = 100
        n_edges = 500
        x = torch.randn(n_nodes, 64)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))

        output = gat_layer(x, edge_index)

        # With concat=False, output should be out_features
        assert output.shape == (n_nodes, 32)
