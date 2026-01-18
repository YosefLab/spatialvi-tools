"""Comprehensive tests for VIVS model, module, and utilities."""

import numpy as np
import pytest
import torch


class TestVIVSModuleInitialization:
    """Tests for VIVSModule initialization."""

    def test_basic_initialization(self):
        """Test basic module initialization."""
        from spatialvi.external.vivs import VIVSModule

        module = VIVSModule(
            n_genes=100,
            n_scales=5,
            n_neighbors_base=20,
        )

        assert module.n_genes == 100
        assert module.n_scales == 5

    def test_custom_scales(self):
        """Test initialization with custom scales."""
        from spatialvi.external.vivs import VIVSModule

        module = VIVSModule(
            n_genes=100,
            n_scales=3,
            n_neighbors_base=10,
        )

        assert module.n_scales == 3
        assert len(module.scales) == 3

    def test_gpu_option(self):
        """Test initialization with GPU option."""
        from spatialvi.external.vivs import VIVSModule

        module = VIVSModule(
            n_genes=100,
            use_gpu=False,
        )

        assert module.device == torch.device("cpu")


class TestVIVSModuleForward:
    """Tests for VIVSModule forward pass."""

    def test_forward_pass(self):
        """Test forward pass returns expected outputs."""
        from spatialvi.external.vivs import VIVSModule

        n_genes = 50
        n_cells = 100
        n_neighbors = 10
        module = VIVSModule(n_genes=n_genes)

        X = torch.randn(n_cells, n_genes).abs()
        neighbor_indices = torch.randint(0, n_cells, (n_cells, n_neighbors))

        results = module(X, neighbor_indices, n_permutations=10)

        assert "z_scores" in results
        assert "local_variance" in results
        assert results["z_scores"].shape == (n_genes,)

    def test_compute_local_variance(self):
        """Test local variance computation."""
        from spatialvi.external.vivs import VIVSModule

        n_genes = 20
        n_cells = 50
        n_neighbors = 5
        module = VIVSModule(n_genes=n_genes)

        X = torch.randn(n_cells, n_genes).abs()
        neighbor_indices = torch.randint(0, n_cells, (n_cells, n_neighbors))

        local_var = module.compute_local_variance(X, neighbor_indices)

        assert local_var.shape == (n_genes,)
        assert (local_var >= 0).all()


class TestMultiScaleVIVS:
    """Tests for MultiScaleVIVS class."""

    def test_initialization(self):
        """Test MultiScaleVIVS initialization."""
        from spatialvi.external.vivs import MultiScaleVIVS

        module = MultiScaleVIVS(
            n_genes=100,
            scales=[10, 20, 50],
        )

        assert module.scales == [10, 20, 50]
        assert module.n_scales == 3

    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        from spatialvi.external.vivs import MultiScaleVIVS

        for agg in ["max", "mean", "weighted"]:
            module = MultiScaleVIVS(
                n_genes=50,
                aggregation=agg,
            )
            assert module.aggregation == agg

    def test_aggregate_scores(self):
        """Test score aggregation."""
        from spatialvi.external.vivs import MultiScaleVIVS

        module = MultiScaleVIVS(n_genes=20, aggregation="max")
        scale_scores = torch.randn(20, 3)

        aggregated = module.aggregate_scores(scale_scores)

        assert aggregated.shape == (20,)


class TestVIVSUtils:
    """Tests for VIVS utility functions."""

    def test_compute_multiscale_neighbors(self, small_spatial_adata):
        """Test multiscale neighbor computation."""
        from spatialvi.external.vivs import compute_multiscale_neighbors

        adata = small_spatial_adata.copy()
        scales = [5, 10, 15]

        neighbor_indices = compute_multiscale_neighbors(
            adata, scales=scales, spatial_key="spatial"
        )

        assert len(neighbor_indices) == len(scales)
        for i, idx_array in enumerate(neighbor_indices):
            assert idx_array.shape[0] == adata.n_obs
            assert idx_array.shape[1] == scales[i]

    def test_compute_fdr(self):
        """Test FDR computation."""
        from spatialvi.external.vivs import compute_fdr

        pvalues = np.array([0.01, 0.02, 0.05, 0.1, 0.5])
        fdr_values, significant = compute_fdr(pvalues, alpha=0.1)

        assert len(fdr_values) == len(pvalues)
        assert len(significant) == len(pvalues)
        assert all(0 <= q <= 1 for q in fdr_values)

    def test_z_to_pvalue(self):
        """Test z-score to p-value conversion."""
        from spatialvi.external.vivs import z_to_pvalue

        z_scores = np.array([0.0, 1.96, -1.96, 3.0])
        pvalues = z_to_pvalue(z_scores)

        assert len(pvalues) == len(z_scores)
        assert all(0 <= p <= 1 for p in pvalues)
        # z=0 should give p~1
        assert pvalues[0] > 0.9
        # z=1.96 should give p~0.05
        assert 0.04 < pvalues[1] < 0.06

    def test_rank_genes_by_spatial_variance(self, small_spatial_adata):
        """Test spatial variance gene ranking."""
        from spatialvi.external.vivs import rank_genes_by_spatial_variance

        adata = small_spatial_adata.copy()
        # Add mock scores to var
        adata.var["vivs_importance"] = np.random.rand(adata.n_vars)

        ranked = rank_genes_by_spatial_variance(
            adata, scores_key="vivs_importance", n_top=10
        )

        assert isinstance(ranked, list)
        assert len(ranked) == 10


class TestVIVSModelWrapper:
    """Tests for VIVS model wrapper."""

    def test_import(self):
        """Test that VIVS wrapper can be imported."""
        from spatialvi.external.vivs import VIVS

        assert VIVS is not None

    def test_initialization(self, small_spatial_adata):
        """Test VIVS model initialization."""
        from spatialvi.external.vivs import VIVS

        adata = small_spatial_adata.copy()
        model = VIVS(adata, spatial_key="spatial")

        assert model.adata is adata
        assert model.spatial_key == "spatial"
