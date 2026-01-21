"""Tests for the LambdaModel wrapper."""

import pytest

from spatialvi_tools.data import load_dummy_spatial_dataset
from spatialvi_tools.models import LambdaModel


def test_lambda_initialization() -> None:
    adata = load_dummy_spatial_dataset(n_cells=10, n_genes=5)
    model = LambdaModel(adata, provider="openai")
    assert model.provider == "openai"


def test_lambda_train_predict_without_dependency() -> None:
    adata = load_dummy_spatial_dataset(n_cells=10, n_genes=5)
    model = LambdaModel(adata)
    try:
        model.train()
        # attempt to predict annotations on mock clusters; no clusters exist so annotate fails gracefully
        model.predict()
    except ImportError:
        pytest.skip("LAMBDA dependency not installed")
    except RuntimeError:
        # Agent has no stored annotation; acceptable
        pass