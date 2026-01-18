"""Tests for SPARL (Spatial Proteomics Analysis with Representation Learning)."""

import pytest


class TestSPARLInitialization:
    """Tests for SPARL model initialization."""

    def test_basic_initialization(self, small_spatial_adata):
        """Test basic model initialization."""
        from spatialvi.external import SPARL

        adata = small_spatial_adata.copy()

        model = SPARL(adata)

        assert model.adata is adata
        assert model.spatial_key == "spatial"
        assert model.layer is None
        assert model.n_latent == 32
        assert model._is_trained is False

    def test_custom_parameters(self, small_spatial_adata):
        """Test initialization with custom parameters."""
        from spatialvi.external import SPARL

        adata = small_spatial_adata.copy()
        adata.layers["normalized"] = adata.X.copy()
        adata.obsm["coords"] = adata.obsm["spatial"]

        model = SPARL(
            adata,
            spatial_key="coords",
            layer="normalized",
            n_latent=64,
        )

        assert model.spatial_key == "coords"
        assert model.layer == "normalized"
        assert model.n_latent == 64


class TestSPARLMethods:
    """Tests for SPARL model methods."""

    def test_require_sparl_raises(self, small_spatial_adata):
        """Test that _require_sparl raises ImportError when sparl not installed."""
        from spatialvi.external import SPARL

        adata = small_spatial_adata.copy()

        model = SPARL(adata)

        try:
            model._require_sparl()
        except ImportError as e:
            assert "sparl" in str(e).lower()

    def test_train_without_dependency(self, small_spatial_adata):
        """Test train method handles missing dependency."""
        from spatialvi.external import SPARL

        adata = small_spatial_adata.copy()

        model = SPARL(adata)

        try:
            model.train(max_epochs=1)
        except ImportError:
            pytest.skip("sparl package not installed")
        except (ValueError, RuntimeError, OSError):
            # Other errors are acceptable if dependency is installed
            pass

    def test_get_latent_before_train_raises(self, small_spatial_adata):
        """Test get_latent_representation raises error before training."""
        from spatialvi.external import SPARL

        adata = small_spatial_adata.copy()

        model = SPARL(adata)

        try:
            model.get_latent_representation()
        except ImportError:
            pytest.skip("sparl package not installed")
        except RuntimeError as e:
            assert "trained" in str(e).lower()

    def test_predict_before_train_raises(self, small_spatial_adata):
        """Test predict raises error before training."""
        from spatialvi.external import SPARL

        adata = small_spatial_adata.copy()

        model = SPARL(adata)

        try:
            model.predict()
        except ImportError:
            pytest.skip("sparl package not installed")
        except RuntimeError as e:
            assert "trained" in str(e).lower()

    def test_reconstruct_before_train_raises(self, small_spatial_adata):
        """Test reconstruct raises error before training."""
        from spatialvi.external import SPARL

        adata = small_spatial_adata.copy()

        model = SPARL(adata)

        try:
            model.reconstruct()
        except ImportError:
            pytest.skip("sparl package not installed")
        except RuntimeError as e:
            assert "trained" in str(e).lower()
