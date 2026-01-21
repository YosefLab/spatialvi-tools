"""Tests for the VIVSModel wrapper."""

import pytest

import numpy as np

from spatialvi_tools.data import load_dummy_spatial_dataset
from spatialvi_tools.models import VIVSModel


def test_vivs_initialization() -> None:
    adata = load_dummy_spatial_dataset(n_cells=10, n_genes=5)
    # create a fake response matrix in obsm
    adata.obsm["Y"] = np.random.randn(10, 2)
    model = VIVSModel(adata, feature_obsm_key="Y")
    assert model.feature_obsm_key == "Y"


def test_vivs_train_predict_without_dependency() -> None:
    adata = load_dummy_spatial_dataset(n_cells=10, n_genes=5)
    adata.obsm["Y"] = np.random.randn(10, 1)
    model = VIVSModel(adata, feature_obsm_key="Y")
    try:
        model.train()
    except ImportError:
        pytest.skip("vivs dependency not installed")
    else:
        # If vivs is installed, we can compute importance scores
        result = model.predict(n_clusters_list=[2])
        assert result is not None
        assert "vivs_hier_importance" in model.adata.uns