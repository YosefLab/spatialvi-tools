"""Comprehensive tests for scVIVA utilities."""

import numpy as np
import pytest


class TestScVIVAUtils:
    """Tests for scVIVA utility functions."""

    def test_compute_niche_composition(self, small_spatial_adata):
        """Test niche composition computation."""
        from spatialvi.external.scviva import compute_niche_composition

        adata = small_spatial_adata.copy()
        adata.obs["cell_type"] = np.random.choice(
            ["TypeA", "TypeB", "TypeC"], adata.n_obs
        )

        composition = compute_niche_composition(
            adata,
            labels_key="cell_type",
            spatial_key="spatial",
            n_neighbors=5,
        )

        assert composition.shape[0] == adata.n_obs
        assert composition.shape[1] == 3  # 3 cell types
        # Each row should sum to ~1
        row_sums = composition.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_identify_niche_clusters_kmeans(self, small_spatial_adata):
        """Test niche clustering with kmeans."""
        from spatialvi.external.scviva import (
            compute_niche_composition,
            identify_niche_clusters,
        )

        adata = small_spatial_adata.copy()
        adata.obs["cell_type"] = np.random.choice(
            ["TypeA", "TypeB"], adata.n_obs
        )

        composition = compute_niche_composition(
            adata, labels_key="cell_type", n_neighbors=5
        )

        clusters = identify_niche_clusters(
            composition, n_clusters=3, method="kmeans"
        )

        assert len(clusters) == adata.n_obs
        assert len(np.unique(clusters)) == 3

    def test_compute_niche_heterogeneity(self, small_spatial_adata):
        """Test niche heterogeneity computation."""
        from spatialvi.external.scviva import compute_niche_heterogeneity

        adata = small_spatial_adata.copy()
        adata.obs["niche"] = np.random.choice(["N1", "N2"], adata.n_obs)

        heterogeneity = compute_niche_heterogeneity(adata, niche_key="niche")

        assert isinstance(heterogeneity, dict)
        assert "N1" in heterogeneity
        assert "N2" in heterogeneity
        assert all(h >= 0 for h in heterogeneity.values())

    def test_compute_niche_interaction_strength(self, small_spatial_adata):
        """Test niche interaction strength computation."""
        from spatialvi.external.scviva import compute_niche_interaction_strength

        adata = small_spatial_adata.copy()
        adata.obs["cell_type"] = np.random.choice(
            ["A", "B", "C"], adata.n_obs
        )

        interactions = compute_niche_interaction_strength(
            adata,
            labels_key="cell_type",
            spatial_key="spatial",
            n_neighbors=5,
        )

        assert interactions.shape == (3, 3)

    def test_identify_boundary_cells(self, small_spatial_adata):
        """Test boundary cell identification."""
        from spatialvi.external.scviva import identify_boundary_cells

        adata = small_spatial_adata.copy()
        adata.obs["cell_type"] = np.random.choice(
            ["A", "B"], adata.n_obs
        )

        boundary = identify_boundary_cells(
            adata,
            labels_key="cell_type",
            spatial_key="spatial",
            n_neighbors=5,
            threshold=0.3,
        )

        assert len(boundary) == adata.n_obs
        assert boundary.dtype == bool

    def test_compute_niche_differential_genes(self, small_spatial_adata):
        """Test niche differential gene finding."""
        from spatialvi.external.scviva import compute_niche_differential_genes

        adata = small_spatial_adata.copy()
        adata.obs["niche"] = np.random.choice(["N1", "N2"], adata.n_obs)

        markers = compute_niche_differential_genes(
            adata, niche_key="niche", n_genes=5
        )

        assert isinstance(markers, dict)
        assert "N1" in markers
        assert "N2" in markers
        for genes in markers.values():
            assert len(genes) <= 5

    def test_compute_spatial_entropy(self, small_spatial_adata):
        """Test spatial entropy computation."""
        from spatialvi.external.scviva import compute_spatial_entropy

        adata = small_spatial_adata.copy()
        adata.obs["cell_type"] = np.random.choice(
            ["A", "B", "C"], adata.n_obs
        )

        entropy = compute_spatial_entropy(
            adata,
            labels_key="cell_type",
            spatial_key="spatial",
            n_neighbors=5,
        )

        assert len(entropy) == adata.n_obs
        assert all(e >= 0 for e in entropy)  # Entropy is non-negative

    def test_visualize_niche_embedding_umap(self, small_spatial_adata):
        """Test niche embedding visualization with UMAP."""
        from spatialvi.external.scviva import visualize_niche_embedding

        adata = small_spatial_adata.copy()
        niche_effects = np.random.randn(adata.n_obs, 10)

        embedding = visualize_niche_embedding(
            adata, niche_effects, method="umap"
        )

        assert embedding.shape == (adata.n_obs, 2)

    def test_visualize_niche_embedding_pca(self, small_spatial_adata):
        """Test niche embedding visualization with PCA."""
        from spatialvi.external.scviva import visualize_niche_embedding

        adata = small_spatial_adata.copy()
        niche_effects = np.random.randn(adata.n_obs, 10)

        embedding = visualize_niche_embedding(
            adata, niche_effects, method="pca"
        )

        assert embedding.shape == (adata.n_obs, 2)


class TestScVIVAModelWrapper:
    """Tests for scVIVA model wrapper."""

    def test_import(self):
        """Test that scVIVA wrapper can be imported."""
        from spatialvi.external.scviva import scVIVA

        assert scVIVA is not None

    def test_utils_import(self):
        """Test that all utils can be imported."""
        from spatialvi.external.scviva import (
            compute_niche_composition,
            compute_niche_differential_genes,
            compute_niche_heterogeneity,
            compute_niche_interaction_strength,
            compute_spatial_entropy,
            identify_boundary_cells,
            identify_niche_clusters,
            visualize_niche_embedding,
        )

        assert compute_niche_composition is not None
        assert identify_niche_clusters is not None
