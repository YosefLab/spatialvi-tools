# tests/base/test_fields.py
import numpy as np
import pytest
from anndata import AnnData

from scviva.data._fields import NeighborhoodGraphField, SpatialCoordsField


def _make_adata(coords_2d=True):
    n = 50
    adata = AnnData(X=np.random.rand(n, 20))
    if coords_2d:
        adata.obsm["spatial"] = np.random.rand(n, 2)
    else:
        adata.obsm["spatial"] = np.random.rand(n, 3)
    return adata


def test_spatial_coords_field_2d():
    adata = _make_adata(coords_2d=True)
    field = SpatialCoordsField(obsm_key="spatial")
    data = field.get_field_data(adata)
    assert data.shape == (50, 2)


def test_spatial_coords_field_3d():
    adata = _make_adata(coords_2d=False)
    field = SpatialCoordsField(obsm_key="spatial")
    data = field.get_field_data(adata)
    assert data.shape == (50, 3)


def test_spatial_coords_field_invalid_dim():
    adata = _make_adata()
    adata.obsm["spatial"] = np.random.rand(50, 5)  # invalid
    field = SpatialCoordsField(obsm_key="spatial")
    with pytest.raises(ValueError, match="2D or 3D"):
        field.get_field_data(adata)


def test_neighborhood_graph_field_converts_sparse():
    from scipy.sparse import csr_matrix

    adata = _make_adata()
    sparse_idx = csr_matrix(np.random.randint(0, 50, size=(50, 6)))
    adata.obsm["index_neighbor"] = sparse_idx
    field = NeighborhoodGraphField(obsm_key="index_neighbor")
    data = field.get_field_data(adata)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 6)
