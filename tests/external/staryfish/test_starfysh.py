"""Comprehensive tests for Starfysh model, module, and utilities."""

import numpy as np
import pytest
import torch


class TestStarfyshModuleInitialization:
    """Tests for StarfyshModule initialization."""

    def test_basic_initialization(self):
        """Test basic module initialization."""
        from spatialvi.external.starfysh import StarfyshModule

        module = StarfyshModule(
            n_genes=100,
            n_cell_types=5,
            n_hidden=64,
        )

        assert module.n_genes == 100
        assert module.n_cell_types == 5

    def test_with_factors(self):
        """Test initialization with custom factors."""
        from spatialvi.external.starfysh import StarfyshModule

        module = StarfyshModule(
            n_genes=100,
            n_cell_types=5,
            n_factors=10,
            n_hidden=64,
        )

        assert module.n_factors == 10

    def test_with_histology(self):
        """Test initialization with histology mode."""
        from spatialvi.external.starfysh import StarfyshModule

        module = StarfyshModule(
            n_genes=100,
            n_cell_types=5,
            n_hidden=64,
            use_histology=True,
            histology_dim=128,
        )

        assert module.use_histology is True


class TestStarfyshModuleForward:
    """Tests for StarfyshModule forward pass."""

    def test_forward_pass(self):
        """Test forward pass returns expected outputs."""
        from spatialvi.external.starfysh import StarfyshModule

        module = StarfyshModule(
            n_genes=100,
            n_cell_types=5,
            n_hidden=64,
        )

        batch_size = 32
        x = torch.randn(batch_size, 100).abs()
        tensors = {"X": x}

        inference_outputs, generative_outputs, loss = module(
            tensors, compute_loss=True
        )

        assert "proportions" in inference_outputs
        assert "concentration" in inference_outputs
        assert "px_rate" in generative_outputs
        assert loss.loss.numel() == 1
        assert torch.isfinite(loss.loss)

    def test_proportions_sum_to_one(self):
        """Test that predicted proportions sum to 1."""
        from spatialvi.external.starfysh import StarfyshModule

        module = StarfyshModule(
            n_genes=100,
            n_cell_types=5,
            n_hidden=64,
        )

        batch_size = 32
        x = torch.randn(batch_size, 100).abs()
        tensors = {"X": x}

        inference_outputs, _, _ = module(tensors, compute_loss=True)
        proportions = inference_outputs["proportions"]

        # Proportions should sum to ~1
        sums = proportions.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestStarfyshUtils:
    """Tests for Starfysh utility functions."""

    def test_compute_reference_signatures(self, small_spatial_adata):
        """Test reference signature computation."""
        from spatialvi.external.starfysh import compute_reference_signatures

        adata = small_spatial_adata.copy()
        adata.obs["cell_type"] = np.random.choice(
            ["A", "B", "C"], adata.n_obs
        )
        adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

        signatures, cell_types = compute_reference_signatures(
            adata, cell_type_key="cell_type"
        )

        assert signatures.shape[0] == 3  # 3 cell types
        assert signatures.shape[1] == adata.n_vars
        assert len(cell_types) == 3

    def test_find_marker_genes(self, small_spatial_adata):
        """Test marker gene identification."""
        from spatialvi.external.starfysh import find_marker_genes

        adata = small_spatial_adata.copy()
        adata.obs["cell_type"] = np.random.choice(
            ["A", "B", "C"], adata.n_obs
        )

        markers = find_marker_genes(
            adata, cell_type_key="cell_type", n_markers=5
        )

        assert isinstance(markers, dict)
        assert len(markers) == 3
        for genes in markers.values():
            assert len(genes) <= 5

    def test_validate_input_data(self, small_spatial_adata):
        """Test input data validation."""
        from spatialvi.external.starfysh import validate_input_data

        adata_spatial = small_spatial_adata.copy()
        adata_ref = small_spatial_adata.copy()
        adata_ref.obs["cell_type"] = np.random.choice(
            ["A", "B", "C"], adata_ref.n_obs
        )
        adata_ref.obs["cell_type"] = adata_ref.obs["cell_type"].astype("category")

        common_genes, cell_types = validate_input_data(
            adata_spatial, adata_ref, cell_type_key="cell_type"
        )

        assert len(common_genes) > 0
        assert len(cell_types) == 3

    def test_proportions_to_counts(self):
        """Test proportion to count conversion."""
        from spatialvi.external.starfysh import proportions_to_counts

        proportions = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])
        total_counts = np.array([100, 200])

        counts = proportions_to_counts(proportions, total_counts)

        assert counts.shape == proportions.shape
        # Should be approximately correct
        np.testing.assert_array_almost_equal(
            counts, np.array([[50, 30, 20], [20, 140, 40]]), decimal=0
        )

    def test_evaluate_deconvolution(self):
        """Test deconvolution evaluation."""
        from spatialvi.external.starfysh import evaluate_deconvolution

        predicted = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])
        true = np.array([[0.4, 0.4, 0.2], [0.2, 0.6, 0.2]])

        metrics = evaluate_deconvolution(predicted, true)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "mean_correlation" in metrics
        assert "cosine_similarity" in metrics
        assert metrics["rmse"] >= 0


class TestStarfyshModelWrapper:
    """Tests for Starfysh model wrapper."""

    def test_import(self):
        """Test that Starfysh wrapper can be imported."""
        from spatialvi.external.starfysh import Starfysh

        assert Starfysh is not None

    def test_initialization(self, small_spatial_adata):
        """Test Starfysh model initialization."""
        from spatialvi.external.starfysh import Starfysh

        adata_spatial = small_spatial_adata.copy()
        adata_ref = small_spatial_adata.copy()
        adata_ref.obs["cell_type"] = np.random.choice(
            ["A", "B", "C"], adata_ref.n_obs
        )
        adata_ref.obs["cell_type"] = adata_ref.obs["cell_type"].astype("category")

        model = Starfysh(
            adata_spatial=adata_spatial,
            adata_ref=adata_ref,
            cell_type_key="cell_type"
        )

        assert model is not None
