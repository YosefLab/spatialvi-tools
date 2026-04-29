# tests/base/test_neighborhood_mixin.py
import numpy as np
import pytest
from anndata import AnnData
from scvi.model.base import UnsupervisedTrainingMixin

from scviva.model.base._neighborhood_mixin import SpatialNeighborhoodMixin
from scviva.model.base._spatial_base import SpatialBaseModel


def _make_coords_adata(n=100, n_genes=20):
    adata = AnnData(X=np.abs(np.random.rand(n, n_genes)))
    adata.layers["counts"] = np.abs(np.random.poisson(3, size=(n, n_genes)))
    adata.obsm["spatial"] = np.random.rand(n, 2)
    return adata


def test_compute_neighbors_squidpy_adds_obsm():
    pytest.importorskip("squidpy")
    adata = _make_coords_adata()

    class _M(SpatialNeighborhoodMixin, SpatialBaseModel, UnsupervisedTrainingMixin):
        def train(self, *args, **kwargs):
            pass

        @classmethod
        def setup_anndata(cls, adata, **kwargs):
            from scvi.data import AnnDataManager
            from scvi.data.fields import LayerField

            mgr = AnnDataManager(fields=[LayerField("X", "counts", is_count_data=True)])
            mgr.register_fields(adata)
            cls.register_manager(mgr)

    _M.setup_anndata(adata)
    model = _M(adata)
    model.compute_neighbors(adata, coord_type="generic", n_neighs=6, backend="squidpy")
    assert "index_neighbor" in adata.obsm
    assert "distance_neighbor" in adata.obsm
    assert adata.obsm["index_neighbor"].shape == (100, 6)
    # Guard against silent all-zero failure (wrong squidpy obsp key suffix)
    assert adata.obsm["index_neighbor"].sum() > 0, (
        "index_neighbor is all zeros — squidpy obsp key suffix may have changed. "
        "Check sq.gr.spatial_neighbors key_added convention for installed squidpy version."
    )


def test_compute_neighbors_invalid_backend():
    adata = _make_coords_adata()
    mixin = SpatialNeighborhoodMixin()
    with pytest.raises(ValueError, match="backend"):
        mixin.compute_neighbors(adata, backend="unknown")
