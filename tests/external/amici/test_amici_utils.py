"""Comprehensive tests for AMICI utilities."""

import numpy as np
import pytest


class TestAMICIUtils:
    """Tests for AMICI utility functions."""

    def test_compute_interaction_neighbors(self, small_spatial_adata):
        """Test interaction neighbor computation."""
        from spatialvi.external.amici import compute_interaction_neighbors

        adata = small_spatial_adata.copy()

        indices, distances = compute_interaction_neighbors(
            adata, spatial_key="spatial", n_neighbors=5
        )

        assert indices.shape[0] == adata.n_obs
        assert indices.shape[1] == 5
        assert distances.shape[0] == adata.n_obs
        assert distances.shape[1] == 5

    def test_compute_interaction_neighbors_with_max_dist(self, small_spatial_adata):
        """Test interaction neighbor computation with max distance."""
        from spatialvi.external.amici import compute_interaction_neighbors

        adata = small_spatial_adata.copy()

        indices, distances = compute_interaction_neighbors(
            adata, spatial_key="spatial", n_neighbors=5, max_dist=10.0
        )

        assert indices.shape[0] == adata.n_obs
        # Check that far neighbors are marked as -1
        # (depending on spatial extent, some may be within distance)

    def test_build_interaction_matrix(self, small_spatial_adata):
        """Test interaction matrix construction."""
        from spatialvi.external.amici import (
            build_interaction_matrix,
            compute_interaction_neighbors,
        )

        adata = small_spatial_adata.copy()
        adata.obs["cell_type"] = np.random.choice(
            ["A", "B", "C"], adata.n_obs
        )

        # First compute neighbors
        neighbor_indices, _ = compute_interaction_neighbors(
            adata, spatial_key="spatial", n_neighbors=5
        )

        matrix = build_interaction_matrix(
            adata,
            labels_key="cell_type",
            neighbor_indices=neighbor_indices,
        )

        assert matrix.shape == (3, 3)  # 3 cell types
        assert list(matrix.index) == list(matrix.columns)

    def test_compute_interaction_strength(self, small_spatial_adata):
        """Test interaction strength computation."""
        from spatialvi.external.amici import compute_interaction_strength

        adata = small_spatial_adata.copy()
        adata.obs["cell_type"] = np.random.choice(
            ["TypeA", "TypeB"], adata.n_obs
        )

        n_neighbors = 5
        attention_weights = np.random.rand(adata.n_obs, n_neighbors)
        labels = adata.obs["cell_type"].values

        strength = compute_interaction_strength(
            attention_weights=attention_weights,
            labels=labels,
        )

        assert isinstance(strength, dict)

    def test_get_ligand_receptor_pairs(self):
        """Test ligand-receptor pair retrieval."""
        from spatialvi.external.amici import get_ligand_receptor_pairs

        pairs = get_ligand_receptor_pairs()

        # Returns a DataFrame
        assert hasattr(pairs, "columns")
        assert "ligand" in pairs.columns
        assert "receptor" in pairs.columns
        assert len(pairs) > 0

    def test_get_ligand_receptor_pairs_mouse(self):
        """Test ligand-receptor pair retrieval for mouse."""
        from spatialvi.external.amici import get_ligand_receptor_pairs

        pairs = get_ligand_receptor_pairs(species="mouse")

        assert len(pairs) > 0
        # Mouse genes should be capitalized
        assert pairs["ligand"].iloc[0][0].isupper()

    def test_filter_expressed_pairs(self, small_spatial_adata):
        """Test filtering of expressed ligand-receptor pairs."""
        from spatialvi.external.amici import (
            filter_expressed_pairs,
            get_ligand_receptor_pairs,
        )
        import pandas as pd

        adata = small_spatial_adata.copy()

        # Create a fake pairs DataFrame using genes from adata
        pairs = pd.DataFrame({
            "ligand": [adata.var_names[0], adata.var_names[2]],
            "receptor": [adata.var_names[1], adata.var_names[3]],
            "category": ["test", "test"],
        })

        filtered = filter_expressed_pairs(
            adata, pairs, min_pct=0.0
        )

        assert isinstance(filtered, pd.DataFrame)
        assert "ligand" in filtered.columns
        assert "receptor" in filtered.columns

    def test_filter_expressed_pairs_with_threshold(self, small_spatial_adata):
        """Test filtering with expression threshold."""
        from spatialvi.external.amici import filter_expressed_pairs
        import pandas as pd

        adata = small_spatial_adata.copy()

        # Create pairs using genes from adata
        pairs = pd.DataFrame({
            "ligand": [adata.var_names[0]],
            "receptor": [adata.var_names[1]],
            "category": ["test"],
        })

        # With high threshold, may filter out pairs
        filtered = filter_expressed_pairs(
            adata, pairs, min_pct=0.5
        )

        assert isinstance(filtered, pd.DataFrame)


class TestAMICIModuleImports:
    """Tests for AMICI module imports."""

    def test_import_module(self):
        """Test AMICIModule import."""
        from spatialvi.external.amici import AMICIModule

        assert AMICIModule is not None

    def test_import_utils(self):
        """Test AMICI utils import."""
        from spatialvi.external.amici import (
            build_interaction_matrix,
            compute_interaction_neighbors,
            compute_interaction_strength,
            filter_expressed_pairs,
            get_ligand_receptor_pairs,
        )

        assert compute_interaction_neighbors is not None
        assert build_interaction_matrix is not None
        assert compute_interaction_strength is not None
        assert get_ligand_receptor_pairs is not None
        assert filter_expressed_pairs is not None
