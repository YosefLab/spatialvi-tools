"""Comprehensive tests for DestVI utilities."""

import numpy as np
import pandas as pd
import pytest


class TestDestVIUtils:
    """Tests for DestVI utility functions."""

    def test_compute_cell_type_abundance(self):
        """Test cell type abundance computation."""
        from spatialvi.external.destvi import compute_cell_type_abundance

        proportions = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.3, 0.3, 0.4],
        ])

        abundance = compute_cell_type_abundance(proportions, normalize=True)

        assert len(abundance) == 3
        assert np.isclose(abundance.sum(), 1.0)

    def test_compute_cell_type_abundance_unnormalized(self):
        """Test unnormalized abundance computation."""
        from spatialvi.external.destvi import compute_cell_type_abundance

        proportions = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
        ])

        abundance = compute_cell_type_abundance(proportions, normalize=False)

        assert len(abundance) == 3
        assert np.isclose(abundance.sum(), 2.0)  # Sum of rows

    def test_compute_cell_type_abundance_dataframe(self):
        """Test abundance computation with DataFrame input."""
        from spatialvi.external.destvi import compute_cell_type_abundance

        proportions = pd.DataFrame({
            "TypeA": [0.5, 0.2],
            "TypeB": [0.3, 0.6],
            "TypeC": [0.2, 0.2],
        })

        abundance = compute_cell_type_abundance(proportions)

        assert len(abundance) == 3

    def test_compute_spatial_autocorrelation(self, small_spatial_adata):
        """Test spatial autocorrelation (Moran's I) computation."""
        from spatialvi.external.destvi import compute_spatial_autocorrelation

        adata = small_spatial_adata.copy()
        proportions = np.random.dirichlet([1, 1, 1], size=adata.n_obs)

        morans_i = compute_spatial_autocorrelation(
            adata, proportions, spatial_key="spatial", n_neighbors=5
        )

        assert isinstance(morans_i, dict)
        assert len(morans_i) == 3
        for ct, value in morans_i.items():
            assert -1 <= value <= 1  # Moran's I range

    def test_identify_dominant_cell_type(self):
        """Test dominant cell type identification."""
        from spatialvi.external.destvi import identify_dominant_cell_type

        proportions = np.array([
            [0.6, 0.2, 0.2],  # Dominant: 0
            [0.2, 0.5, 0.3],  # Dominant: 1
            [0.3, 0.3, 0.4],  # Dominant: 2
            [0.33, 0.33, 0.34],  # No dominant (below threshold)
        ])

        dominant = identify_dominant_cell_type(proportions, threshold=0.4)

        assert dominant[0] == 0
        assert dominant[1] == 1
        assert dominant[2] == 2
        assert dominant[3] == -1  # No dominant

    def test_compute_colocalization_pearson(self):
        """Test colocalization with Pearson correlation."""
        from spatialvi.external.destvi import compute_colocalization

        proportions = np.random.rand(100, 4)

        coloc = compute_colocalization(proportions, method="pearson")

        assert coloc.shape == (4, 4)
        # Diagonal should be 1
        assert np.allclose(np.diag(coloc), 1.0)
        # Should be symmetric
        assert np.allclose(coloc, coloc.T)

    def test_compute_colocalization_spearman(self):
        """Test colocalization with Spearman correlation."""
        from spatialvi.external.destvi import compute_colocalization

        proportions = np.random.rand(100, 3)

        coloc = compute_colocalization(proportions, method="spearman")

        assert coloc.shape == (3, 3)
        assert np.allclose(np.diag(coloc), 1.0)

    def test_compute_niche_enrichment(self, small_spatial_adata):
        """Test niche enrichment computation."""
        from spatialvi.external.destvi import compute_niche_enrichment

        adata = small_spatial_adata.copy()
        adata.obs["region"] = np.random.choice(["R1", "R2"], adata.n_obs)
        proportions = pd.DataFrame(
            np.random.dirichlet([1, 1, 1], size=adata.n_obs),
            columns=["TypeA", "TypeB", "TypeC"],
            index=adata.obs_names,
        )

        enrichment = compute_niche_enrichment(
            adata, proportions, region_key="region"
        )

        assert enrichment.shape[0] == 2  # 2 regions
        assert enrichment.shape[1] == 3  # 3 cell types
        assert "TypeA" in enrichment.columns

    def test_validate_reference_overlap(self, small_spatial_adata):
        """Test reference overlap validation."""
        from spatialvi.external.destvi import validate_reference_overlap

        sc_adata = small_spatial_adata.copy()
        st_adata = small_spatial_adata.copy()

        # Add some unique genes to each
        sc_adata.var_names = [f"gene_{i}" for i in range(sc_adata.n_vars)]
        st_adata.var_names = [f"gene_{i}" for i in range(st_adata.n_vars)]

        result = validate_reference_overlap(sc_adata, st_adata, min_genes=10)

        assert "n_shared" in result
        assert "n_sc_only" in result
        assert "n_st_only" in result
        assert "is_valid" in result
        assert "shared_genes" in result

    def test_validate_reference_overlap_insufficient(self, small_spatial_adata):
        """Test validation with insufficient overlap."""
        from spatialvi.external.destvi import validate_reference_overlap

        sc_adata = small_spatial_adata.copy()
        st_adata = small_spatial_adata.copy()

        # Make genes completely different
        sc_adata.var_names = [f"sc_gene_{i}" for i in range(sc_adata.n_vars)]
        st_adata.var_names = [f"st_gene_{i}" for i in range(st_adata.n_vars)]

        result = validate_reference_overlap(sc_adata, st_adata, min_genes=10)

        assert result["n_shared"] == 0
        assert result["is_valid"] is False


class TestDestVIModelWrapper:
    """Tests for DestVI model wrapper."""

    def test_import(self):
        """Test that DestVI wrapper can be imported."""
        from spatialvi.external.destvi import DestVI, CondSCVI

        assert DestVI is not None
        assert CondSCVI is not None

    def test_utils_import(self):
        """Test that all utils can be imported."""
        from spatialvi.external.destvi import (
            compute_cell_type_abundance,
            compute_colocalization,
            compute_niche_enrichment,
            compute_spatial_autocorrelation,
            identify_dominant_cell_type,
            validate_reference_overlap,
        )

        assert compute_cell_type_abundance is not None
        assert compute_colocalization is not None
