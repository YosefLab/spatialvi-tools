"""Tests for DestVI module classes."""


class TestDestVIModuleImports:
    """Tests for DestVI module imports."""

    def test_import_mrdeconv(self):
        """Test MRDeconv import."""
        from spatialvi.external.destvi import MRDeconv

        assert MRDeconv is not None

    def test_import_destvi_module(self):
        """Test DestVIModule alias import."""
        from spatialvi.external.destvi import DestVIModule

        assert DestVIModule is not None

    def test_destvi_module_is_mrdeconv(self):
        """Test DestVIModule is subclass of MRDeconv."""
        from spatialvi.external.destvi import DestVIModule, MRDeconv

        assert issubclass(DestVIModule, MRDeconv)


class TestMRDeconvStructure:
    """Tests for MRDeconv module structure."""

    def test_mrdeconv_has_expected_methods(self):
        """Test MRDeconv has expected methods."""
        from spatialvi.external.destvi import MRDeconv

        assert hasattr(MRDeconv, "inference")
        assert hasattr(MRDeconv, "generative")
        assert hasattr(MRDeconv, "loss")
        assert hasattr(MRDeconv, "get_proportions")
        assert hasattr(MRDeconv, "_get_inference_input")
        assert hasattr(MRDeconv, "_get_generative_input")


class TestDestVIUtilsImport:
    """Tests for DestVI utility imports."""

    def test_import_utils(self):
        """Test utility function imports."""
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
        assert compute_niche_enrichment is not None
        assert compute_spatial_autocorrelation is not None
        assert identify_dominant_cell_type is not None
        assert validate_reference_overlap is not None


class TestDestVIUtilFunctions:
    """Tests for DestVI utility functions."""

    def test_compute_cell_type_abundance(self):
        """Test cell type abundance computation."""
        import numpy as np

        from spatialvi.external.destvi import compute_cell_type_abundance

        proportions = np.random.rand(100, 5)
        proportions = proportions / proportions.sum(axis=1, keepdims=True)

        abundance = compute_cell_type_abundance(proportions)

        assert abundance.shape == (5,)
        assert np.isclose(abundance.sum(), 1.0)

    def test_identify_dominant_cell_type(self):
        """Test dominant cell type identification."""
        import numpy as np

        from spatialvi.external.destvi import identify_dominant_cell_type

        proportions = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.1, 0.8],
            ]
        )

        dominant = identify_dominant_cell_type(proportions, threshold=0.5)

        assert dominant.shape == (3,)
        assert dominant[0] == 0
        assert dominant[1] == 1
        assert dominant[2] == 2

    def test_compute_colocalization(self):
        """Test colocalization computation."""
        import numpy as np

        from spatialvi.external.destvi import compute_colocalization

        proportions = np.random.rand(100, 5)
        proportions = proportions / proportions.sum(axis=1, keepdims=True)

        coloc = compute_colocalization(proportions, method="pearson")

        assert coloc.shape == (5, 5)
        assert np.allclose(np.diag(coloc), 1.0)
        assert np.allclose(coloc, coloc.T)
