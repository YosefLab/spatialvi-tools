"""Tests for utility functions."""

import numpy as np
import pytest


class TestMetrics:
    """Tests for spatial metrics."""

    def test_spatial_autocorrelation(self, spatial_adata_with_neighbors):
        """Test spatial autocorrelation computation."""
        from spatialvi.utils import spatial_autocorrelation

        adata = spatial_adata_with_neighbors.copy()
        gene_names = adata.var_names[:5].tolist()

        results = spatial_autocorrelation(
            adata,
            var_names=gene_names,
            method="moran",
        )

        assert len(results) == 5
        for gene in gene_names:
            assert gene in results
            assert -1 <= results[gene] <= 1

    def test_morans_i(self):
        """Test Moran's I computation."""
        from spatialvi.utils import compute_morans_i

        n = 100
        values = np.random.randn(n)
        weights = np.eye(n)
        weights[np.triu_indices(n, 1)] = np.random.random((n * (n - 1)) // 2) > 0.9
        weights = weights + weights.T
        weights = weights / weights.sum(axis=1, keepdims=True)

        moran_i = compute_morans_i(values, weights)
        assert -1 <= moran_i <= 1

    def test_silhouette_spatial(self, spatial_adata_with_neighbors):
        """Test silhouette score with spatial coordinates."""
        from spatialvi.utils import silhouette_spatial

        adata = spatial_adata_with_neighbors.copy()
        score = silhouette_spatial(adata, labels_key="cell_type")

        assert -1 <= score <= 1


class TestVisualization:
    """Tests for visualization functions."""

    
    def test_plot_spatial(self, small_spatial_adata):
        """Test spatial plot."""
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend

        from spatialvi.utils import plot_spatial

        ax = plot_spatial(small_spatial_adata, color="cell_type", show=False)
        assert ax is not None

    def test_plot_proportions(self, small_spatial_adata):
        """Test proportions plot."""
        import matplotlib
        matplotlib.use("Agg")

        from spatialvi.utils import plot_proportions

        # Add fake proportions
        n_types = 3
        adata = small_spatial_adata.copy()
        adata.obsm["proportions"] = np.random.dirichlet(
            np.ones(n_types), size=adata.n_obs
        )
        adata.uns["cell_type_names"] = [f"Type_{i}" for i in range(n_types)]

        fig = plot_proportions(adata, show=False)
        assert fig is not None

    def test_plot_interactions(self):
        """Test interactions plot."""
        import matplotlib
        matplotlib.use("Agg")

        from spatialvi.utils import plot_interactions

        n_types = 4
        interaction_matrix = np.random.randn(n_types, n_types)
        cell_types = [f"Type_{i}" for i in range(n_types)]

        ax = plot_interactions(interaction_matrix, cell_types, show=False)
        assert ax is not None
