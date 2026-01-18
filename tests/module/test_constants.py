"""Tests for module constants."""


class TestModuleKeys:
    """Tests for MODULE_KEYS constants."""

    def test_module_keys_import(self):
        """Test MODULE_KEYS can be imported."""
        from spatialvi.module import MODULE_KEYS

        assert MODULE_KEYS is not None

    def test_latent_keys(self):
        """Test latent variable keys are defined."""
        from spatialvi.module import MODULE_KEYS

        assert hasattr(MODULE_KEYS, "Z_KEY")
        assert hasattr(MODULE_KEYS, "QZ_KEY")
        assert hasattr(MODULE_KEYS, "QZM_KEY")
        assert hasattr(MODULE_KEYS, "QZV_KEY")

    def test_library_keys(self):
        """Test library size keys are defined."""
        from spatialvi.module import MODULE_KEYS

        assert hasattr(MODULE_KEYS, "LIBRARY_KEY")
        assert hasattr(MODULE_KEYS, "QL_KEY")
        assert hasattr(MODULE_KEYS, "QLM_KEY")
        assert hasattr(MODULE_KEYS, "QLV_KEY")

    def test_generative_keys(self):
        """Test generative output keys are defined."""
        from spatialvi.module import MODULE_KEYS

        assert hasattr(MODULE_KEYS, "PX_KEY")
        assert hasattr(MODULE_KEYS, "PX_SCALE_KEY")
        assert hasattr(MODULE_KEYS, "PX_RATE_KEY")
        assert hasattr(MODULE_KEYS, "PX_R_KEY")

    def test_spatial_keys(self):
        """Test spatial-specific keys are defined."""
        from spatialvi.module import MODULE_KEYS

        assert hasattr(MODULE_KEYS, "SPATIAL_LATENT_KEY")
        assert hasattr(MODULE_KEYS, "NEIGHBOR_Z_KEY")
        assert hasattr(MODULE_KEYS, "SPATIAL_CONTEXT_KEY")
        assert hasattr(MODULE_KEYS, "NICHE_KEY")

    def test_deconvolution_keys(self):
        """Test deconvolution keys are defined."""
        from spatialvi.module import MODULE_KEYS

        assert hasattr(MODULE_KEYS, "PROPORTIONS_KEY")
        assert hasattr(MODULE_KEYS, "CELLTYPE_SCALE_KEY")

    def test_attention_keys(self):
        """Test attention keys are defined."""
        from spatialvi.module import MODULE_KEYS

        assert hasattr(MODULE_KEYS, "ATTENTION_WEIGHTS_KEY")
        assert hasattr(MODULE_KEYS, "NEIGHBOR_ATTENTION_KEY")


class TestSpatialModuleKeys:
    """Tests for SPATIAL_MODULE_KEYS constants."""

    def test_spatial_module_keys_import(self):
        """Test SPATIAL_MODULE_KEYS can be imported."""
        from spatialvi.module import SPATIAL_MODULE_KEYS

        assert SPATIAL_MODULE_KEYS is not None

    def test_coordinate_keys(self):
        """Test spatial coordinate keys are defined."""
        from spatialvi.module import SPATIAL_MODULE_KEYS

        assert hasattr(SPATIAL_MODULE_KEYS, "COORDS_KEY")
        assert hasattr(SPATIAL_MODULE_KEYS, "NEIGHBOR_INDEX_KEY")
        assert hasattr(SPATIAL_MODULE_KEYS, "NEIGHBOR_DIST_KEY")

    def test_niche_keys(self):
        """Test niche composition keys are defined."""
        from spatialvi.module import SPATIAL_MODULE_KEYS

        assert hasattr(SPATIAL_MODULE_KEYS, "NICHE_COMPOSITION_KEY")
        assert hasattr(SPATIAL_MODULE_KEYS, "NICHE_COMPOSITION_PRED_KEY")

    def test_interaction_keys(self):
        """Test cell-cell interaction keys are defined."""
        from spatialvi.module import SPATIAL_MODULE_KEYS

        assert hasattr(SPATIAL_MODULE_KEYS, "INTERACTION_KEY")
        assert hasattr(SPATIAL_MODULE_KEYS, "LIGAND_KEY")
        assert hasattr(SPATIAL_MODULE_KEYS, "RECEPTOR_KEY")

    def test_spatial_loss_keys(self):
        """Test spatial loss keys are defined."""
        from spatialvi.module import SPATIAL_MODULE_KEYS

        assert hasattr(SPATIAL_MODULE_KEYS, "SPATIAL_LOSS_KEY")
        assert hasattr(SPATIAL_MODULE_KEYS, "NEIGHBOR_LOSS_KEY")


class TestLossKeys:
    """Tests for LOSS_KEYS constants."""

    def test_loss_keys_import(self):
        """Test LOSS_KEYS can be imported."""
        from spatialvi.module import LOSS_KEYS

        assert LOSS_KEYS is not None

    def test_vae_loss_keys(self):
        """Test standard VAE loss keys are defined."""
        from spatialvi.module import LOSS_KEYS

        assert hasattr(LOSS_KEYS, "RECONSTRUCTION_LOSS")
        assert hasattr(LOSS_KEYS, "KL_LOCAL")
        assert hasattr(LOSS_KEYS, "KL_GLOBAL")

    def test_spatial_loss_keys(self):
        """Test spatial-specific loss keys are defined."""
        from spatialvi.module import LOSS_KEYS

        assert hasattr(LOSS_KEYS, "SPATIAL_LOSS")
        assert hasattr(LOSS_KEYS, "NEIGHBOR_LOSS")
        assert hasattr(LOSS_KEYS, "NICHE_LOSS")

    def test_classification_loss_keys(self):
        """Test classification loss keys are defined."""
        from spatialvi.module import LOSS_KEYS

        assert hasattr(LOSS_KEYS, "CLASSIFICATION_LOSS")
        assert hasattr(LOSS_KEYS, "CELLTYPE_LOSS")

    def test_regularization_keys(self):
        """Test regularization keys are defined."""
        from spatialvi.module import LOSS_KEYS

        assert hasattr(LOSS_KEYS, "L1_REG")
        assert hasattr(LOSS_KEYS, "L2_REG")


class TestConstantValues:
    """Test that constants have correct string values."""

    def test_z_key_value(self):
        """Test Z_KEY has correct value."""
        from spatialvi.module import MODULE_KEYS

        assert MODULE_KEYS.Z_KEY == "z"

    def test_qz_key_value(self):
        """Test QZ_KEY has correct value."""
        from spatialvi.module import MODULE_KEYS

        assert MODULE_KEYS.QZ_KEY == "qz"

    def test_coords_key_value(self):
        """Test COORDS_KEY has correct value."""
        from spatialvi.module import SPATIAL_MODULE_KEYS

        assert SPATIAL_MODULE_KEYS.COORDS_KEY == "spatial_coords"

    def test_reconstruction_loss_value(self):
        """Test RECONSTRUCTION_LOSS has correct value."""
        from spatialvi.module import LOSS_KEYS

        assert LOSS_KEYS.RECONSTRUCTION_LOSS == "reconstruction_loss"


class TestAllConstantsExported:
    """Test all constants are properly exported."""

    def test_module_exports_all_constants(self):
        """Test module __all__ includes all constant classes."""
        from spatialvi import module

        assert "MODULE_KEYS" in module.__all__
        assert "SPATIAL_MODULE_KEYS" in module.__all__
        assert "LOSS_KEYS" in module.__all__
