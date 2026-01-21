"""Tests for dummy dataset generation."""

import numpy as np

from spatialvi_tools.data import load_dummy_spatial_dataset


def test_dummy_dataset_shapes() -> None:
    adata = load_dummy_spatial_dataset(n_cells=10, n_genes=5, n_dims=2)
    # check shape of count matrix
    assert adata.X.shape == (10, 5)
    # check presence of spatial coordinates
    assert "spatial" in adata.obsm
    assert adata.obsm["spatial"].shape == (10, 2)
    # check presence of dummy embedding
    assert "X_scVI" in adata.obsm
    assert adata.obsm["X_scVI"].shape[0] == 10