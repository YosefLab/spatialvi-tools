"""Shared pytest fixtures for scviva test suite."""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from scvi.model.base import UnsupervisedTrainingMixin

from scviva.data._fields import SpatialCoordsField
from scviva.model.base._spatial_base import SpatialBaseModel


class _MinimalSpatialModel(SpatialBaseModel, UnsupervisedTrainingMixin):
    """Minimal concrete model for testing SpatialBaseModel methods in isolation.

    Cannot be trained (no real module.inference). Used only for testing
    field registration, setup_spatialdata, and plotting helpers.
    """

    @classmethod
    def setup_anndata(cls, adata, layer=None, spatial_key="spatial", **kwargs):
        fields = [
            LayerField("X", layer, is_count_data=True),
            SpatialCoordsField(obsm_key=spatial_key),
        ]
        manager = AnnDataManager(fields=fields)
        manager.register_fields(adata, **kwargs)
        cls.register_manager(manager)


def make_spatial_adata(n: int = 80, n_genes: int = 20) -> AnnData:
    """Create a minimal AnnData with counts layer and spatial coordinates."""
    adata = AnnData(X=np.abs(np.random.rand(n, n_genes)))
    adata.layers["counts"] = np.abs(np.random.poisson(3, size=(n, n_genes)))
    adata.obsm["spatial"] = np.random.rand(n, 2)
    return adata


@pytest.fixture(scope="module")
def minimal_spatial_adata():
    return make_spatial_adata()


@pytest.fixture(scope="module")
def minimal_model(minimal_spatial_adata):
    _MinimalSpatialModel.setup_anndata(
        minimal_spatial_adata, layer="counts", spatial_key="spatial"
    )
    return _MinimalSpatialModel(minimal_spatial_adata)
