import numpy as np
import pytest
from anndata import AnnData

from spatialvi.utils._spatial import get_spatial_coords, validate_spatial_coords


def test_get_spatial_coords_2d():
    adata = AnnData(X=np.random.rand(10, 5))
    adata.obsm["spatial"] = np.random.rand(10, 2)
    coords = get_spatial_coords(adata, key="spatial")
    assert coords.shape == (10, 2)


def test_get_spatial_coords_missing_key():
    adata = AnnData(X=np.random.rand(10, 5))
    with pytest.raises(KeyError):
        get_spatial_coords(adata, key="missing")


def test_validate_spatial_coords_valid():
    coords = np.random.rand(20, 2)
    validate_spatial_coords(coords)  # should not raise


def test_validate_spatial_coords_invalid():
    coords = np.random.rand(20, 5)
    with pytest.raises(ValueError):
        validate_spatial_coords(coords)
