# tests/base/test_spatial_base.py
import numpy as np
import pytest
from anndata import AnnData
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from scvi.model.base import UnsupervisedTrainingMixin

from scviva.data._fields import SpatialCoordsField
from scviva.model.base._spatial_base import SpatialBaseModel


# Minimal concrete subclass — SpatialBaseModel cannot be instantiated directly
class _MinimalSpatialModel(SpatialBaseModel, UnsupervisedTrainingMixin):
    def train(self, *args, **kwargs):
        pass

    @classmethod
    def setup_anndata(cls, adata, layer=None, spatial_key="spatial", **kwargs):
        fields = [
            LayerField("X", layer, is_count_data=True),
            SpatialCoordsField(obsm_key=spatial_key),
        ]
        manager = AnnDataManager(fields=fields)
        manager.register_fields(adata, **kwargs)
        cls.register_manager(manager)


def _make_spatial_adata(n=80, n_genes=20):
    adata = AnnData(X=np.abs(np.random.rand(n, n_genes)))
    adata.layers["counts"] = np.abs(np.random.poisson(3, size=(n, n_genes)))
    adata.obsm["spatial"] = np.random.rand(n, 2)
    return adata


@pytest.fixture(scope="module")
def minimal_model():
    adata = _make_spatial_adata()
    _MinimalSpatialModel.setup_anndata(adata, layer="counts", spatial_key="spatial")
    return _MinimalSpatialModel(adata)


def test_setup_anndata_registers_spatial(minimal_model):
    mgr = minimal_model.adata_manager
    assert "spatial" in [f.attr_key for f in mgr.fields]


def test_setup_spatialdata_requires_spatialdata(minimal_model):
    """setup_spatialdata must raise ImportError if spatialdata is not installed
    or TypeError if given a non-SpatialData object."""
    with pytest.raises((ImportError, TypeError)):
        _MinimalSpatialModel.setup_spatialdata(object(), table_key="table", region="cells")


def test_get_latent_cpu_not_implemented_on_base(minimal_model):
    """SpatialBaseModel.get_latent_representation calls super() which raises
    NotImplementedError on the minimal model (no module.inference)."""
    with pytest.raises((NotImplementedError, AttributeError)):
        minimal_model.get_latent_representation()
