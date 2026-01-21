"""Tests for the NolanModel wrapper."""

import pytest

from spatialvi_tools.data import load_dummy_spatial_dataset
from spatialvi_tools.models import NolanModel


def test_nolan_initialization() -> None:
    adata = load_dummy_spatial_dataset(n_cells=10, n_genes=5)
    model = NolanModel(adata, num_niches=5)
    # attributes are stored correctly
    assert model.num_niches == 5
    assert model.emb_key == "X_scVI"
    assert model.spatial_key == "spatial"


def test_nolan_train_without_dependency() -> None:
    adata = load_dummy_spatial_dataset(n_cells=10, n_genes=5)
    model = NolanModel(adata)
    # If nolan is not installed, training should raise ImportError
    try:
        model.train(num_epochs=1)
    except ImportError:
        pytest.skip("nolan dependency not installed")
    else:
        # If nolan is installed, ensure model is trained and embeddings can be generated
        adata_out = model.predict()
        assert "X_nolan" in adata_out.obsm