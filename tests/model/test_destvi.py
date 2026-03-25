"""Tests for DestVI model."""

import numpy as np
import pytest
from scvi.data import synthetic_iid
from scvi.model import CondSCVI

from spatialvi.model._destvi import DestVI


@pytest.fixture(scope="session")
def destvi_data():
    sc_adata = synthetic_iid(n_labels=4, n_genes=50)
    sc_adata.layers["counts"] = sc_adata.X.copy()
    st_adata = synthetic_iid(n_labels=4, n_genes=50, n_batches=1)
    st_adata.layers["counts"] = st_adata.X.copy()
    st_adata.obsm["spatial"] = np.random.rand(st_adata.n_obs, 2)
    return sc_adata, st_adata


def test_destvi_from_rna_model_train(destvi_data):
    sc_adata, st_adata = destvi_data
    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="labels")
    sc_model = CondSCVI(sc_adata, weight_obs=False)
    sc_model.train(max_epochs=2, accelerator="cpu")
    DestVI.setup_anndata(st_adata, layer="counts")
    st_model = DestVI.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=2, accelerator="cpu")
    assert st_model.is_trained


def test_destvi_get_proportions_df(destvi_data):
    sc_adata, st_adata = destvi_data
    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="labels")
    sc_model = CondSCVI(sc_adata, weight_obs=False)
    sc_model.train(max_epochs=2, accelerator="cpu")
    DestVI.setup_anndata(st_adata, layer="counts")
    st_model = DestVI.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=2, accelerator="cpu")
    df = st_model.get_proportions_df(st_adata)
    assert df.shape[1] == 4  # n_labels
    np.testing.assert_allclose(df.sum(axis=1).values, 1.0, atol=1e-4)


def test_condscvi_not_re_exported():
    """CondSCVI must NOT be exported from spatialvi — users import from scvi.model directly."""
    import spatialvi

    with pytest.raises(AttributeError):
        _ = spatialvi.CondSCVI
