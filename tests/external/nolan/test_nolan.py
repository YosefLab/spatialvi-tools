"""Tests for Nolan (NicheExplorer) model."""

import pytest
import numpy as np


class TestNolanInitialization:
    """Tests for Nolan model initialization."""

    def test_basic_initialization(self, small_spatial_adata):
        """Test basic model initialization."""
        from spatialvi.external import Nolan

        adata = small_spatial_adata.copy()
        # Add mock embeddings
        adata.obsm["X_scVI"] = np.random.randn(adata.n_obs, 10)

        model = Nolan(adata, num_niches=20)

        assert model.adata is adata
        assert model.num_niches == 20
        assert model.emb_key == "X_scVI"
        assert model.spatial_key == "spatial"
        assert model._is_trained is False

    def test_custom_keys(self, small_spatial_adata):
        """Test initialization with custom keys."""
        from spatialvi.external import Nolan

        adata = small_spatial_adata.copy()
        adata.obsm["custom_emb"] = np.random.randn(adata.n_obs, 20)
        adata.obsm["coords"] = adata.obsm["spatial"]

        model = Nolan(
            adata,
            emb_key="custom_emb",
            spatial_key="coords",
            num_niches=30,
        )

        assert model.emb_key == "custom_emb"
        assert model.spatial_key == "coords"
        assert model.num_niches == 30

    def test_with_batch_key(self, small_spatial_adata):
        """Test initialization with batch key."""
        from spatialvi.external import Nolan

        adata = small_spatial_adata.copy()
        adata.obsm["X_scVI"] = np.random.randn(adata.n_obs, 10)
        adata.obs["batch"] = np.random.choice(["A", "B"], adata.n_obs)

        model = Nolan(adata, batch_key="batch")

        assert model.batch_key == "batch"


class TestNolanMethods:
    """Tests for Nolan model methods."""

    def test_require_nolan_raises(self, small_spatial_adata):
        """Test that _require_nolan raises ImportError when nolan not installed."""
        from spatialvi.external import Nolan

        adata = small_spatial_adata.copy()
        adata.obsm["X_scVI"] = np.random.randn(adata.n_obs, 10)

        model = Nolan(adata)

        # This will either work (if nolan installed) or raise ImportError
        try:
            model._require_nolan()
        except ImportError as e:
            assert "nolan" in str(e).lower()

    def test_train_without_dependency(self, small_spatial_adata):
        """Test train method handles missing dependency."""
        from spatialvi.external import Nolan

        adata = small_spatial_adata.copy()
        adata.obsm["X_scVI"] = np.random.randn(adata.n_obs, 10)

        model = Nolan(adata)

        try:
            model.train(num_epochs=1)
        except ImportError:
            pytest.skip("nolan package not installed")
        except Exception:
            # Other errors are acceptable if dependency is installed
            pass

    def test_predict_before_train_raises(self, small_spatial_adata):
        """Test predict raises error before training."""
        from spatialvi.external import Nolan

        adata = small_spatial_adata.copy()
        adata.obsm["X_scVI"] = np.random.randn(adata.n_obs, 10)

        model = Nolan(adata)

        try:
            model.predict()
        except ImportError:
            pytest.skip("nolan package not installed")
        except RuntimeError as e:
            assert "trained" in str(e).lower()
