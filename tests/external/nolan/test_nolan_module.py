"""Comprehensive tests for Nolan module and utilities."""

import numpy as np
import pytest
import torch


class TestNolanModuleInitialization:
    """Tests for NolanModule initialization."""

    def test_basic_initialization(self):
        """Test basic module initialization."""
        from spatialvi.external.nolan import NolanModule

        module = NolanModule(
            input_dim=64,
            hidden_dim=32,
            output_dim=10,
        )

        assert module.input_dim == 64
        assert module.output_dim == 10

    def test_with_custom_layers(self):
        """Test initialization with custom layers."""
        from spatialvi.external.nolan import NolanModule

        module = NolanModule(
            input_dim=64,
            hidden_dim=32,
            output_dim=10,
            n_layers=3,
        )

        assert module is not None

    def test_different_architectures(self):
        """Test different architecture configurations."""
        from spatialvi.external.nolan import NolanModule

        for n_layers in [1, 2, 3]:
            module = NolanModule(
                input_dim=64,
                hidden_dim=32,
                output_dim=10,
                n_layers=n_layers,
            )
            assert module is not None


class TestNolanModuleForward:
    """Tests for NolanModule forward pass."""

    def test_forward_pass(self):
        """Test forward pass returns expected outputs."""
        from spatialvi.external.nolan import NolanModule

        module = NolanModule(
            input_dim=64,
            hidden_dim=32,
            output_dim=10,
        )

        batch_size = 16
        x = torch.randn(batch_size, 64)

        outputs = module(x)

        assert "niche_embedding" in outputs
        assert "hidden" in outputs
        assert outputs["niche_embedding"].shape == (batch_size, 10)

    def test_forward_with_neighbor(self):
        """Test forward pass with neighbor input."""
        from spatialvi.external.nolan import NolanModule

        module = NolanModule(
            input_dim=64,
            hidden_dim=32,
            output_dim=10,
        )

        batch_size = 16
        x = torch.randn(batch_size, 64)
        x_neighbor = torch.randn(batch_size, 64)

        outputs = module(x, x_neighbor=x_neighbor)

        assert "niche_embedding" in outputs
        assert "z_neighbor" in outputs
        assert "prediction" in outputs

    def test_niche_embedding_shape(self):
        """Test niche embedding output shape."""
        from spatialvi.external.nolan import NolanModule

        output_dim = 15
        module = NolanModule(
            input_dim=64,
            hidden_dim=32,
            output_dim=output_dim,
        )

        batch_size = 16
        x = torch.randn(batch_size, 64)

        outputs = module(x)

        assert outputs["niche_embedding"].shape == (batch_size, output_dim)


class TestNicheClusteringHead:
    """Tests for NicheClusteringHead class."""

    def test_initialization(self):
        """Test NicheClusteringHead initialization."""
        from spatialvi.external.nolan import NicheClusteringHead

        head = NicheClusteringHead(
            input_dim=32,
            n_clusters=20,
        )

        assert head.n_clusters == 20

    def test_forward(self):
        """Test NicheClusteringHead forward."""
        from spatialvi.external.nolan import NicheClusteringHead

        head = NicheClusteringHead(
            input_dim=32,
            n_clusters=20,
        )

        batch_size = 16
        x = torch.randn(batch_size, 32)

        outputs = head(x)

        assert "assignments" in outputs
        assert "distances" in outputs
        assert "cluster_ids" in outputs
        assert outputs["assignments"].shape == (batch_size, 20)
        # Should be probabilities (softmax)
        assert torch.allclose(
            outputs["assignments"].sum(dim=1),
            torch.ones(batch_size),
            atol=1e-5
        )

    def test_hard_assignment(self):
        """Test hard cluster assignment."""
        from spatialvi.external.nolan import NicheClusteringHead

        head = NicheClusteringHead(
            input_dim=32,
            n_clusters=20,
            soft_assignment=False,
        )

        batch_size = 16
        x = torch.randn(batch_size, 32)

        outputs = head(x)

        # Hard assignment should be one-hot
        assert torch.all(
            outputs["assignments"].sum(dim=1) == torch.ones(batch_size)
        )


class TestNolanContrastiveLoss:
    """Tests for Nolan contrastive loss."""

    def test_contrastive_loss(self):
        """Test contrastive loss computation."""
        from spatialvi.external.nolan import NolanModule

        module = NolanModule(
            input_dim=64,
            hidden_dim=32,
            output_dim=10,
        )

        batch_size = 16
        z = torch.randn(batch_size, 10)
        z_pos = torch.randn(batch_size, 10)

        loss = module.contrastive_loss(z, z_pos)

        assert loss.numel() == 1
        assert torch.isfinite(loss)

    def test_contrastive_loss_with_negatives(self):
        """Test contrastive loss with explicit negatives."""
        from spatialvi.external.nolan import NolanModule

        module = NolanModule(
            input_dim=64,
            hidden_dim=32,
            output_dim=10,
        )

        batch_size = 16
        z = torch.randn(batch_size, 10)
        z_pos = torch.randn(batch_size, 10)
        z_neg = torch.randn(batch_size * 2, 10)

        loss = module.contrastive_loss(z, z_pos, z_neg)

        assert loss.numel() == 1
        assert torch.isfinite(loss)


class TestNolanUtils:
    """Tests for Nolan utility functions."""

    def test_compute_grid_size(self, small_spatial_adata):
        """Test grid size computation."""
        from spatialvi.external.nolan import compute_grid_size

        adata = small_spatial_adata.copy()

        radius, mean_count, max_count = compute_grid_size(
            adata, spatial_key="spatial", expected_num_cells=10
        )

        assert isinstance(radius, (float, np.floating))
        assert radius > 0
        assert mean_count > 0
        assert max_count > 0

    def test_sample_spatial_crops(self, small_spatial_adata):
        """Test spatial crop sampling."""
        from spatialvi.external.nolan import sample_spatial_crops

        adata = small_spatial_adata.copy()
        coords = adata.obsm["spatial"]
        if hasattr(coords, "values"):
            coords = coords.values

        crops = sample_spatial_crops(
            coords=coords,
            crop_radius=50.0,
            n_crops=5,
        )

        assert len(crops) == 5
        for crop_indices in crops:
            assert len(crop_indices) > 0

    def test_create_niche_graph(self, small_spatial_adata):
        """Test niche graph creation."""
        from spatialvi.external.nolan import create_niche_graph

        adata = small_spatial_adata.copy()
        adata.obs["niche_cluster"] = np.random.choice(
            ["N1", "N2", "N3"], adata.n_obs
        )

        graph = create_niche_graph(
            adata,
            niche_key="niche_cluster",
            spatial_key="spatial",
        )

        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) == 3  # 3 niche clusters

    def test_evaluate_niche_clustering(self, small_spatial_adata):
        """Test niche clustering evaluation."""
        from spatialvi.external.nolan import evaluate_niche_clustering

        adata = small_spatial_adata.copy()
        adata.obs["niche_cluster"] = np.random.choice(
            ["N1", "N2", "N3"], adata.n_obs
        )
        adata.obs["cell_type"] = np.random.choice(
            ["A", "B", "C"], adata.n_obs
        )

        metrics = evaluate_niche_clustering(
            adata,
            niche_key="niche_cluster",
            labels_key="cell_type",
        )

        assert "spatial_silhouette" in metrics
        assert "ari_with_labels" in metrics
        assert "nmi_with_labels" in metrics


class TestNolanModelWrapper:
    """Tests for Nolan model wrapper."""

    def test_import(self):
        """Test that Nolan wrapper can be imported."""
        from spatialvi.external.nolan import Nolan

        assert Nolan is not None

    def test_import_module(self):
        """Test module import."""
        from spatialvi.external.nolan import NolanModule, NicheClusteringHead

        assert NolanModule is not None
        assert NicheClusteringHead is not None

    def test_import_utils(self):
        """Test utils import."""
        from spatialvi.external.nolan import (
            compute_grid_size,
            create_niche_graph,
            evaluate_niche_clustering,
            sample_spatial_crops,
        )

        assert compute_grid_size is not None
        assert create_niche_graph is not None
        assert evaluate_niche_clustering is not None
        assert sample_spatial_crops is not None
