"""Comprehensive tests for data utilities."""

import numpy as np
import pytest


class TestSyntheticData:
    """Tests for synthetic data generation."""

    def test_synthetic_spatial(self):
        """Test synthetic spatial data generation."""
        from spatialvi.data import synthetic_spatial

        adata = synthetic_spatial(n_cells=100, n_genes=50, n_cell_types=3, seed=42)

        assert adata.n_obs == 100
        assert adata.n_vars == 50
        assert "spatial" in adata.obsm
        assert adata.obsm["spatial"].shape == (100, 2)
        assert "cell_type" in adata.obs.columns

    def test_synthetic_spatial_reproducibility(self):
        """Test that seed makes generation reproducible."""
        from spatialvi.data import synthetic_spatial

        adata1 = synthetic_spatial(n_cells=50, n_genes=20, seed=42)
        adata2 = synthetic_spatial(n_cells=50, n_genes=20, seed=42)

        np.testing.assert_array_equal(adata1.X.toarray(), adata2.X.toarray())
        np.testing.assert_array_equal(adata1.obsm["spatial"], adata2.obsm["spatial"])

    def test_synthetic_spatial_cell_types(self):
        """Test that correct number of cell types are generated."""
        from spatialvi.data import synthetic_spatial

        for n_ct in [2, 5, 10]:
            adata = synthetic_spatial(n_cells=100, n_genes=50, n_cell_types=n_ct)
            unique_types = adata.obs["cell_type"].nunique()
            assert unique_types <= n_ct

    def test_synthetic_scrna(self):
        """Test synthetic scRNA-seq data generation."""
        from spatialvi.data import synthetic_scrna

        adata = synthetic_scrna(n_cells=200, n_genes=50, n_cell_types=3, seed=42)

        assert adata.n_obs == 200
        assert adata.n_vars == 50
        assert "cell_type" in adata.obs.columns
        assert "spatial" not in adata.obsm

    def test_synthetic_scrna_expression_values(self):
        """Test that expression values are non-negative counts."""
        from spatialvi.data import synthetic_scrna

        adata = synthetic_scrna(n_cells=100, n_genes=50)

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        assert (X >= 0).all()
        assert np.allclose(X, X.astype(int))  # Should be integers


class TestPreprocessingNeighbors:
    """Tests for spatial neighbor computation."""

    def test_compute_spatial_neighbors(self, small_spatial_adata):
        """Test spatial neighbor computation."""
        from spatialvi.data import compute_spatial_neighbors

        adata = small_spatial_adata.copy()
        compute_spatial_neighbors(adata, n_neighbors=10)

        assert "nn_index" in adata.obsm
        assert "nn_dist" in adata.obsm
        assert adata.obsm["nn_index"].shape[1] == 10
        assert adata.obsm["nn_dist"].shape[1] == 10

    def test_compute_spatial_neighbors_different_k(self, small_spatial_adata):
        """Test neighbor computation with different k values."""
        from spatialvi.data import compute_spatial_neighbors

        for k in [5, 10, 20]:
            adata = small_spatial_adata.copy()
            compute_spatial_neighbors(adata, n_neighbors=k)
            assert adata.obsm["nn_index"].shape[1] == k

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
    def test_compute_spatial_neighbors_metrics(self, small_spatial_adata, metric):
        """Test different distance metrics."""
        from spatialvi.data import compute_spatial_neighbors

        adata = small_spatial_adata.copy()
        compute_spatial_neighbors(adata, n_neighbors=10, metric=metric)

        assert "nn_index" in adata.obsm
        assert adata.uns["spatial"]["neighbor_metric"] == metric

    def test_compute_spatial_neighbors_custom_keys(self, small_spatial_adata):
        """Test custom output keys."""
        from spatialvi.data import compute_spatial_neighbors

        adata = small_spatial_adata.copy()
        compute_spatial_neighbors(
            adata,
            n_neighbors=10,
            index_key="custom_index",
            dist_key="custom_dist",
        )

        assert "custom_index" in adata.obsm
        assert "custom_dist" in adata.obsm

    def test_compute_spatial_neighbors_distances_sorted(self, small_spatial_adata):
        """Test that distances are sorted (nearest first)."""
        from spatialvi.data import compute_spatial_neighbors

        adata = small_spatial_adata.copy()
        compute_spatial_neighbors(adata, n_neighbors=10)

        distances = adata.obsm["nn_dist"]
        # Each row should be sorted
        for i in range(distances.shape[0]):
            assert (np.diff(distances[i]) >= 0).all()


class TestPreprocessingNicheComposition:
    """Tests for niche composition computation."""

    def test_compute_niche_composition(self, spatial_adata_with_neighbors):
        """Test niche composition computation."""
        from spatialvi.data import compute_niche_composition

        adata = spatial_adata_with_neighbors.copy()
        adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
        compute_niche_composition(adata, labels_key="cell_type")

        assert "niche_composition" in adata.obsm
        # Composition should sum to 1
        row_sums = adata.obsm["niche_composition"].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0)

    def test_niche_composition_values(self, spatial_adata_with_neighbors):
        """Test that composition values are valid probabilities."""
        from spatialvi.data import compute_niche_composition

        adata = spatial_adata_with_neighbors.copy()
        adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
        compute_niche_composition(adata, labels_key="cell_type")

        comp = adata.obsm["niche_composition"]
        assert (comp >= 0).all()
        assert (comp <= 1).all()

    def test_niche_composition_custom_key(self, spatial_adata_with_neighbors):
        """Test custom composition key."""
        from spatialvi.data import compute_niche_composition

        adata = spatial_adata_with_neighbors.copy()
        adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
        compute_niche_composition(adata, labels_key="cell_type", composition_key="custom_comp")

        assert "custom_comp" in adata.obsm

    def test_niche_composition_without_neighbors_raises(self, small_spatial_adata):
        """Test that error is raised without neighbors."""
        from spatialvi.data import compute_niche_composition

        adata = small_spatial_adata.copy()
        with pytest.raises(ValueError, match="Neighbor indices not found"):
            compute_niche_composition(adata, labels_key="cell_type")


class TestPreprocessingNormalization:
    """Tests for spatial normalization."""

    def test_normalize_spatial_minmax(self, small_spatial_adata):
        """Test min-max normalization."""
        from spatialvi.data import normalize_spatial

        adata = small_spatial_adata.copy()
        normalize_spatial(adata, method="minmax")

        coords = adata.obsm["spatial"]
        assert coords.min() >= 0
        assert coords.max() <= 1

    def test_normalize_spatial_zscore(self, small_spatial_adata):
        """Test z-score normalization."""
        from spatialvi.data import normalize_spatial

        adata = small_spatial_adata.copy()
        normalize_spatial(adata, method="zscore")

        coords = adata.obsm["spatial"]
        # Mean should be ~0, std should be ~1
        np.testing.assert_array_almost_equal(coords.mean(axis=0), 0, decimal=5)
        np.testing.assert_array_almost_equal(coords.std(axis=0), 1, decimal=5)

    def test_normalize_spatial_center(self, small_spatial_adata):
        """Test center normalization."""
        from spatialvi.data import normalize_spatial

        adata = small_spatial_adata.copy()
        normalize_spatial(adata, method="center")

        coords = adata.obsm["spatial"]
        # Mean should be ~0
        np.testing.assert_array_almost_equal(coords.mean(axis=0), 0, decimal=5)

    def test_normalize_spatial_invalid_method(self, small_spatial_adata):
        """Test invalid normalization method raises error."""
        from spatialvi.data import normalize_spatial

        adata = small_spatial_adata.copy()
        with pytest.raises(ValueError, match="Unknown normalization"):
            normalize_spatial(adata, method="invalid")


class TestPreprocessingFilter:
    """Tests for spatial filtering."""

    def test_filter_by_spatial_density(self, small_spatial_adata):
        """Test density filtering."""
        from spatialvi.data import filter_by_spatial_density

        adata = small_spatial_adata.copy()
        original_n = adata.n_obs
        filter_by_spatial_density(adata, min_density=0.2)

        assert adata.n_obs <= original_n
        assert "spatial_density" in adata.obs.columns

    def test_filter_by_spatial_density_max(self, small_spatial_adata):
        """Test max density filtering."""
        from spatialvi.data import filter_by_spatial_density

        adata = small_spatial_adata.copy()
        original_n = adata.n_obs
        filter_by_spatial_density(adata, max_density=0.8)

        assert adata.n_obs <= original_n


class TestPreprocessingNoise:
    """Tests for spatial noise addition."""

    def test_add_spatial_noise(self, small_spatial_adata):
        """Test adding spatial noise."""
        from spatialvi.data import add_spatial_noise

        adata = small_spatial_adata.copy()
        original_coords = adata.obsm["spatial"].copy()
        add_spatial_noise(adata, noise_scale=0.01, seed=42)

        new_coords = adata.obsm["spatial"]
        # Coords should be different
        assert not np.allclose(original_coords, new_coords)
        # But not too different
        assert np.allclose(original_coords, new_coords, atol=5)

    def test_add_spatial_noise_reproducibility(self, small_spatial_adata):
        """Test noise reproducibility with seed."""
        from spatialvi.data import add_spatial_noise

        adata1 = small_spatial_adata.copy()
        adata2 = small_spatial_adata.copy()
        add_spatial_noise(adata1, seed=42)
        add_spatial_noise(adata2, seed=42)

        np.testing.assert_array_equal(adata1.obsm["spatial"], adata2.obsm["spatial"])


class TestNeighborExpression:
    """Tests for neighbor expression aggregation."""

    def test_get_neighbor_expression_mean(self, spatial_adata_with_neighbors):
        """Test mean aggregation of neighbor expression."""
        from spatialvi.data import get_neighbor_expression

        adata = spatial_adata_with_neighbors.copy()
        neighbor_expr = get_neighbor_expression(adata, aggregation="mean")

        assert neighbor_expr.shape == (adata.n_obs, adata.n_vars)
        assert (neighbor_expr >= 0).all()

    @pytest.mark.parametrize("aggregation", ["mean", "sum", "max"])
    def test_get_neighbor_expression_aggregations(self, spatial_adata_with_neighbors, aggregation):
        """Test different aggregation methods."""
        from spatialvi.data import get_neighbor_expression

        adata = spatial_adata_with_neighbors.copy()
        neighbor_expr = get_neighbor_expression(adata, aggregation=aggregation)

        assert neighbor_expr.shape == (adata.n_obs, adata.n_vars)

    def test_get_neighbor_expression_without_neighbors_raises(self, small_spatial_adata):
        """Test error without neighbor computation."""
        from spatialvi.data import get_neighbor_expression

        adata = small_spatial_adata.copy()
        with pytest.raises(ValueError, match="Neighbor indices not found"):
            get_neighbor_expression(adata)


class TestFields:
    """Tests for data field classes."""

    def test_spatial_coordinates_field(self, small_spatial_adata):
        """Test spatial coordinates field validation."""
        from spatialvi.data._fields import SpatialCoordinatesField

        field = SpatialCoordinatesField("spatial", "spatial")
        # Should not raise
        field.validate_field(small_spatial_adata)

    def test_neighbor_index_field(self, spatial_adata_with_neighbors):
        """Test neighbor index field validation."""
        from spatialvi.data._fields import NeighborIndexField

        field = NeighborIndexField("nn_index", "nn_index")
        # Should not raise
        field.validate_field(spatial_adata_with_neighbors)
