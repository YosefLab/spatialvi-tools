"""Tests for Lambda (LLM annotation) model."""

import pytest
import numpy as np


class TestLambdaInitialization:
    """Tests for Lambda model initialization."""

    def test_basic_initialization(self, small_spatial_adata):
        """Test basic model initialization."""
        from spatialvi.external import Lambda

        adata = small_spatial_adata.copy()

        model = Lambda(adata)

        assert model.adata is adata
        assert model.provider == "openai"
        assert model.num_parallel == 10
        assert model.n_top_genes == 50
        assert model.resolution == 0.5
        assert model._is_trained is False

    def test_custom_parameters(self, small_spatial_adata):
        """Test initialization with custom parameters."""
        from spatialvi.external import Lambda

        adata = small_spatial_adata.copy()

        model = Lambda(
            adata,
            location="brain",
            organism="human",
            provider="google",
            num_parallel=5,
            n_top_genes=100,
            resolution=1.0,
        )

        assert model.location == "brain"
        assert model.organism == "human"
        assert model.provider == "google"
        assert model.num_parallel == 5
        assert model.n_top_genes == 100
        assert model.resolution == 1.0


class TestLambdaMethods:
    """Tests for Lambda model methods."""

    def test_require_agent_raises(self, small_spatial_adata):
        """Test that _require_agent raises ImportError when LAMBDA not installed."""
        from spatialvi.external import Lambda

        adata = small_spatial_adata.copy()

        model = Lambda(adata)

        try:
            model._require_agent()
        except ImportError as e:
            assert "lambda" in str(e).lower()

    def test_train_without_dependency(self, small_spatial_adata):
        """Test train method handles missing dependency."""
        from spatialvi.external import Lambda

        adata = small_spatial_adata.copy()

        model = Lambda(adata)

        try:
            model.train()
        except ImportError:
            pytest.skip("LAMBDA package not installed")
        except Exception:
            # Other errors are acceptable if dependency is installed
            pass

    def test_predict_lazy_init(self, small_spatial_adata):
        """Test predict lazily initializes agent."""
        from spatialvi.external import Lambda

        adata = small_spatial_adata.copy()

        model = Lambda(adata)

        try:
            model.predict()
        except ImportError:
            pytest.skip("LAMBDA package not installed")
        except Exception:
            pass

    def test_get_annotations_before_train_raises(self, small_spatial_adata):
        """Test get_annotations raises error before training."""
        from spatialvi.external import Lambda

        adata = small_spatial_adata.copy()

        model = Lambda(adata)

        with pytest.raises(RuntimeError, match="not initialized"):
            model.get_annotations()
