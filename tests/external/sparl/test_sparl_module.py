"""Comprehensive tests for SPARL module and utilities."""

import numpy as np
import pytest
import torch


class TestSPARLModuleInitialization:
    """Tests for SPARLModule initialization."""

    def test_basic_initialization(self):
        """Test basic module initialization."""
        from spatialvi.external.sparl import SPARLModule

        module = SPARLModule(
            n_proteins=50,
            n_hidden=64,
            n_latent=16,
        )

        assert module.n_proteins == 50
        assert module.n_latent == 16

    def test_with_spatial(self):
        """Test initialization with spatial mode."""
        from spatialvi.external.sparl import SPARLModule

        module = SPARLModule(
            n_proteins=50,
            n_hidden=64,
            n_latent=16,
            use_spatial=True,
            spatial_dim=32,
        )

        assert module.use_spatial is True
        assert hasattr(module, "spatial_encoder")

    def test_without_spatial(self):
        """Test initialization without spatial mode."""
        from spatialvi.external.sparl import SPARLModule

        module = SPARLModule(
            n_proteins=50,
            n_hidden=64,
            n_latent=16,
            use_spatial=False,
        )

        assert module.use_spatial is False

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_different_layer_depths(self, n_layers):
        """Test different layer configurations."""
        from spatialvi.external.sparl import SPARLModule

        module = SPARLModule(
            n_proteins=50,
            n_hidden=64,
            n_latent=16,
            n_layers=n_layers,
        )

        assert module is not None


class TestSPARLModuleForward:
    """Tests for SPARLModule forward pass."""

    def test_forward_pass(self):
        """Test forward pass returns expected outputs."""
        from spatialvi.external.sparl import SPARLModule

        module = SPARLModule(
            n_proteins=50,
            n_hidden=64,
            n_latent=16,
        )

        batch_size = 32
        x = torch.randn(batch_size, 50).abs()
        tensors = {"X": x}

        inference_outputs, generative_outputs, loss = module(
            tensors, compute_loss=True
        )

        assert "z" in inference_outputs
        assert "qz" in inference_outputs
        assert "qz_m" in inference_outputs
        assert "qz_v" in inference_outputs
        assert "px" in generative_outputs
        assert loss.loss.numel() == 1
        assert torch.isfinite(loss.loss)

    def test_forward_with_spatial(self):
        """Test forward pass with spatial coordinates."""
        from spatialvi.external.sparl import SPARLModule

        module = SPARLModule(
            n_proteins=50,
            n_hidden=64,
            n_latent=16,
            use_spatial=True,
        )

        batch_size = 32
        x = torch.randn(batch_size, 50).abs()
        spatial = torch.randn(batch_size, 2)
        tensors = {"X": x, "spatial": spatial}

        inference_outputs, generative_outputs, loss = module(
            tensors, compute_loss=True
        )

        assert "z" in inference_outputs
        assert torch.isfinite(loss.loss)

    def test_latent_shape(self):
        """Test latent representation shape."""
        from spatialvi.external.sparl import SPARLModule

        n_latent = 20
        module = SPARLModule(
            n_proteins=50,
            n_hidden=64,
            n_latent=n_latent,
        )

        batch_size = 32
        x = torch.randn(batch_size, 50).abs()
        tensors = {"X": x}

        inference_outputs, _, _ = module(tensors, compute_loss=True)

        assert inference_outputs["z"].shape == (batch_size, n_latent)
        assert inference_outputs["qz_m"].shape == (batch_size, n_latent)
        assert inference_outputs["qz_v"].shape == (batch_size, n_latent)


class TestSPARLModuleInference:
    """Tests for SPARLModule inference method."""

    def test_inference_deterministic(self):
        """Test inference in eval mode is more deterministic."""
        from spatialvi.external.sparl import SPARLModule

        module = SPARLModule(
            n_proteins=50,
            n_hidden=64,
            n_latent=16,
        )
        module.eval()

        x = torch.randn(16, 50).abs()

        # Mean should be deterministic
        out1 = module.inference(x=x)
        out2 = module.inference(x=x)

        assert torch.allclose(out1["qz_m"], out2["qz_m"])


class TestSPARLModuleGenerative:
    """Tests for SPARLModule generative method."""

    def test_generative_output(self):
        """Test generative output."""
        from spatialvi.external.sparl import SPARLModule

        module = SPARLModule(
            n_proteins=50,
            n_hidden=64,
            n_latent=16,
        )

        batch_size = 32
        z = torch.randn(batch_size, 16)

        outputs = module.generative(z=z)

        assert "px" in outputs
        assert "px_m" in outputs
        assert "px_v" in outputs
        assert outputs["px_m"].shape == (batch_size, 50)


class TestSPARLUtils:
    """Tests for SPARL utility functions."""

    def test_normalize_protein_expression_arcsinh(self):
        """Test arcsinh normalization."""
        from spatialvi.external.sparl import normalize_protein_expression

        X = np.random.rand(100, 50) * 100

        normalized = normalize_protein_expression(X, method="arcsinh", cofactor=5.0)

        assert normalized.shape == X.shape
        assert normalized.max() < X.max()  # Arcsinh compresses values

    def test_normalize_protein_expression_log1p(self):
        """Test log1p normalization."""
        from spatialvi.external.sparl import normalize_protein_expression

        X = np.random.rand(100, 50) * 100

        normalized = normalize_protein_expression(X, method="log1p")

        assert normalized.shape == X.shape
        assert np.allclose(normalized, np.log1p(X))

    def test_normalize_protein_expression_zscore(self):
        """Test zscore normalization."""
        from spatialvi.external.sparl import normalize_protein_expression

        X = np.random.rand(100, 50) * 100

        normalized = normalize_protein_expression(X, method="zscore")

        assert normalized.shape == X.shape
        # Z-scored columns should have mean ~0, std ~1
        assert np.allclose(normalized.mean(axis=0), 0, atol=1e-5)
        assert np.allclose(normalized.std(axis=0), 1, atol=1e-1)

    def test_compute_protein_coexpression_pearson(self):
        """Test Pearson coexpression computation."""
        from spatialvi.external.sparl import compute_protein_coexpression

        X = np.random.rand(100, 20)

        coexpr = compute_protein_coexpression(X, method="pearson")

        assert coexpr.shape == (20, 20)
        assert np.allclose(np.diag(coexpr), 1.0)
        assert np.allclose(coexpr, coexpr.T)

    def test_compute_protein_coexpression_spearman(self):
        """Test Spearman coexpression computation."""
        from spatialvi.external.sparl import compute_protein_coexpression

        X = np.random.rand(100, 10)

        coexpr = compute_protein_coexpression(X, method="spearman")

        assert coexpr.shape == (10, 10)
        assert np.allclose(np.diag(coexpr), 1.0)

    def test_detect_protein_communities_spectral(self):
        """Test protein community detection with spectral clustering."""
        from spatialvi.external.sparl import (
            compute_protein_coexpression,
            detect_protein_communities,
        )

        X = np.random.rand(100, 15)
        coexpr = compute_protein_coexpression(X)

        communities = detect_protein_communities(
            coexpr, n_communities=3
        )

        assert len(communities) == 15
        assert len(np.unique(communities)) == 3

    def test_spatial_protein_enrichment(self, small_spatial_adata):
        """Test spatial protein enrichment computation."""
        from spatialvi.external.sparl import spatial_protein_enrichment

        adata = small_spatial_adata.copy()
        adata.obs["region"] = np.random.choice(["R1", "R2"], adata.n_obs)

        enrichment = spatial_protein_enrichment(
            adata,
            labels_key="region",
            spatial_key="spatial",
            n_neighbors=5,
        )

        assert enrichment.shape[0] == 2  # 2 regions
        assert enrichment.shape[1] == adata.n_vars

    def test_identify_marker_proteins(self, small_spatial_adata):
        """Test marker protein identification."""
        from spatialvi.external.sparl import identify_marker_proteins

        adata = small_spatial_adata.copy()
        adata.obs["cluster"] = np.random.choice(
            ["C1", "C2", "C3"], adata.n_obs
        )

        markers = identify_marker_proteins(
            adata, labels_key="cluster", n_markers=5
        )

        assert isinstance(markers, dict)
        assert len(markers) == 3
        for proteins in markers.values():
            assert len(proteins) <= 5

    def test_compute_neighborhood_composition(self, small_spatial_adata):
        """Test neighborhood composition computation."""
        from spatialvi.external.sparl import compute_neighborhood_composition

        adata = small_spatial_adata.copy()

        composition = compute_neighborhood_composition(
            adata, spatial_key="spatial", n_neighbors=5
        )

        assert composition.shape == adata.X.shape
