"""Tests for scVIVA module classes."""

import torch


class TestScVIVAImports:
    """Tests for scVIVA imports."""

    def test_import_niche_vae(self):
        """Test nicheVAE import."""
        from spatialvi.external.scviva import nicheVAE

        assert nicheVAE is not None

    def test_import_niche_loss_output(self):
        """Test NicheLossOutput import."""
        from spatialvi.external.scviva import NicheLossOutput

        assert NicheLossOutput is not None

    def test_import_constants(self):
        """Test constants import."""
        from spatialvi.external.scviva import SCVIVA_MODULE_KEYS, SCVIVA_REGISTRY_KEYS

        assert SCVIVA_MODULE_KEYS is not None
        assert SCVIVA_REGISTRY_KEYS is not None


class TestScVIVAComponents:
    """Tests for scVIVA component classes."""

    def test_import_encoder(self):
        """Test Encoder import."""
        from spatialvi.external.scviva import Encoder

        assert Encoder is not None

    def test_import_dirichlet_decoder(self):
        """Test DirichletDecoder import."""
        from spatialvi.external.scviva import DirichletDecoder

        assert DirichletDecoder is not None

    def test_import_niche_decoder(self):
        """Test NicheDecoder import."""
        from spatialvi.external.scviva import NicheDecoder

        assert NicheDecoder is not None

    def test_encoder_initialization(self):
        """Test Encoder initialization."""
        from spatialvi.external.scviva import Encoder

        encoder = Encoder(
            n_input=100,
            n_output=10,
            n_hidden=64,
            n_layers=2,
            dropout_rate=0.1,
        )

        assert encoder is not None
        assert hasattr(encoder, "encoder")
        assert hasattr(encoder, "dist_encoder")

    def test_encoder_forward(self):
        """Test Encoder forward pass."""
        from spatialvi.external.scviva import Encoder

        encoder = Encoder(
            n_input=100,
            n_output=10,
            n_hidden=64,
            n_layers=1,
            return_dist=False,
        )

        x = torch.randn(32, 100)
        q_m, q_v, z = encoder(x)

        assert q_m.shape == (32, 10)
        assert q_v.shape == (32, 10)
        assert z.shape == (32, 10)

    def test_encoder_return_dist(self):
        """Test Encoder with return_dist=True."""
        from spatialvi.external.scviva import Encoder

        encoder = Encoder(
            n_input=100,
            n_output=10,
            n_hidden=64,
            n_layers=1,
            return_dist=True,
        )

        x = torch.randn(32, 100)
        dist, z = encoder(x)

        assert hasattr(dist, "loc")
        assert hasattr(dist, "scale")
        assert z.shape == (32, 10)

    def test_niche_decoder_initialization(self):
        """Test NicheDecoder initialization."""
        from spatialvi.external.scviva import NicheDecoder

        decoder = NicheDecoder(
            n_input=10,
            n_output=50,
            n_niche_components=5,
            n_hidden=64,
            n_layers=1,
        )

        assert decoder is not None
        assert decoder.n_niche_components == 5
        assert decoder.n_output == 50

    def test_niche_decoder_forward(self):
        """Test NicheDecoder forward pass."""
        from spatialvi.external.scviva import NicheDecoder

        decoder = NicheDecoder(
            n_input=10,
            n_output=50,
            n_niche_components=5,
            n_hidden=64,
            n_layers=1,
        )

        z = torch.randn(32, 10)
        p_m, p_v = decoder(z)

        assert p_m.shape == (32, 5, 50)
        assert p_v.shape == (32, 5, 50)
        assert (p_v > 0).all()  # Variance should be positive


class TestScVIVAConstants:
    """Tests for scVIVA constants."""

    def test_registry_keys(self):
        """Test registry keys are defined."""
        from spatialvi.external.scviva import SCVIVA_REGISTRY_KEYS

        assert hasattr(SCVIVA_REGISTRY_KEYS, "SAMPLE_KEY")
        assert hasattr(SCVIVA_REGISTRY_KEYS, "NICHE_COMPOSITION_KEY")
        assert hasattr(SCVIVA_REGISTRY_KEYS, "Z1_MEAN_CT_KEY")
        assert hasattr(SCVIVA_REGISTRY_KEYS, "CELL_COORDINATES_KEY")

    def test_module_keys(self):
        """Test module keys are defined."""
        from spatialvi.external.scviva import SCVIVA_MODULE_KEYS

        assert hasattr(SCVIVA_MODULE_KEYS, "NICHE_MEAN")
        assert hasattr(SCVIVA_MODULE_KEYS, "NICHE_VARIANCE")
        assert hasattr(SCVIVA_MODULE_KEYS, "P_NICHE_COMPOSITION")
        assert hasattr(SCVIVA_MODULE_KEYS, "P_NICHE_EXPRESSION")


class TestScVIVAUtilsImport:
    """Tests for scVIVA utility imports."""

    def test_import_utils(self):
        """Test utility function imports."""
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
        assert compute_niche_differential_genes is not None
        assert compute_niche_heterogeneity is not None
        assert compute_niche_interaction_strength is not None
        assert compute_spatial_entropy is not None
        assert identify_boundary_cells is not None
        assert identify_niche_clusters is not None
        assert visualize_niche_embedding is not None
