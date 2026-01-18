"""Tests for ResolVI module classes."""


class TestRESOLVAEModelImport:
    """Tests for RESOLVAE imports."""

    def test_import_resolvae(self):
        """Test RESOLVAE import."""
        from spatialvi.external.resolvi import RESOLVAE

        assert RESOLVAE is not None

    def test_import_resolvae_model(self):
        """Test RESOLVAEModel import."""
        from spatialvi.external.resolvi import RESOLVAEModel

        assert RESOLVAEModel is not None

    def test_import_resolvae_guide(self):
        """Test RESOLVAEGuide import."""
        from spatialvi.external.resolvi import RESOLVAEGuide

        assert RESOLVAEGuide is not None


class TestRESOLVAEModelStructure:
    """Tests for RESOLVAE model structure."""

    def test_resolvae_model_attributes(self):
        """Test RESOLVAEModel has expected attributes."""
        from spatialvi.external.resolvi import RESOLVAEModel

        # Check class has expected methods
        assert hasattr(RESOLVAEModel, "forward")
        assert hasattr(RESOLVAEModel, "model_unconditioned")
        assert hasattr(RESOLVAEModel, "model_corrected")
        assert hasattr(RESOLVAEModel, "_get_fn_args_from_batch")

    def test_resolvae_guide_attributes(self):
        """Test RESOLVAEGuide has expected attributes."""
        from spatialvi.external.resolvi import RESOLVAEGuide

        # Check class has expected methods
        assert hasattr(RESOLVAEGuide, "forward")

    def test_resolvae_attributes(self):
        """Test RESOLVAE has expected attributes."""
        from spatialvi.external.resolvi import RESOLVAE

        # Check class has expected properties
        assert hasattr(RESOLVAE, "model")
        assert hasattr(RESOLVAE, "guide")
        assert hasattr(RESOLVAE, "model_corrected")
        assert hasattr(RESOLVAE, "model_unconditioned")


class TestRESOLVAEProperties:
    """Tests for RESOLVAE module properties."""

    def test_pyro_module_name(self):
        """Test that the Pyro module name is defined."""
        from spatialvi.external.resolvi._module import _RESOLVAE_PYRO_MODULE_NAME

        assert _RESOLVAE_PYRO_MODULE_NAME == "resolvae"

    def test_pyro_available_check(self):
        """Test that PYRO_AVAILABLE flag exists."""
        from spatialvi.external.resolvi._module import PYRO_AVAILABLE

        assert isinstance(PYRO_AVAILABLE, bool)


class TestResolVIUtilsImport:
    """Tests for ResolVI utility imports."""

    def test_import_utils(self):
        """Test utility function imports."""
        from spatialvi.external.resolvi import (
            compare_denoising_quality,
            compute_background_signal,
            compute_segmentation_confidence,
            compute_signal_to_noise,
            compute_spatial_smoothness,
            filter_low_quality_cells,
            identify_contaminated_cells,
        )

        assert compute_background_signal is not None
        assert compute_segmentation_confidence is not None
        assert identify_contaminated_cells is not None
        assert compute_signal_to_noise is not None
        assert filter_low_quality_cells is not None
        assert compute_spatial_smoothness is not None
        assert compare_denoising_quality is not None
