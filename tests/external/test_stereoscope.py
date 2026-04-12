"""Tests for Stereoscope external models."""

from __future__ import annotations

import numpy as np
import pytest
from scvi.data import synthetic_iid

from spatialvi.external.stereoscope._model import RNAStereoscope, SpatialStereoscope


@pytest.fixture(scope="module")
def stereo_data():
    np.random.seed(0)
    sc_adata = synthetic_iid(n_labels=4, n_genes=50, sparse_format=None)
    sc_adata.layers["counts"] = sc_adata.X.copy()

    st_adata = synthetic_iid(n_genes=50, sparse_format=None)
    st_adata.layers["counts"] = st_adata.X.copy()
    st_adata.var_names = sc_adata.var_names
    st_adata.obsm["spatial"] = np.random.rand(st_adata.n_obs, 2)

    return sc_adata, st_adata


def _train_rna_model(sc_adata):
    RNAStereoscope.setup_anndata(sc_adata, labels_key="labels", layer="counts")
    sc_model = RNAStereoscope(sc_adata)
    sc_model.train(max_epochs=2, accelerator="cpu")
    return sc_model


def test_rna_stereoscope_train(stereo_data):
    sc_adata, _ = stereo_data
    sc_model = _train_rna_model(sc_adata)
    assert sc_model.is_trained


def test_spatial_stereoscope_from_rna_model(stereo_data):
    sc_adata, st_adata = stereo_data
    sc_model = _train_rna_model(sc_adata)

    SpatialStereoscope.setup_anndata(st_adata, layer="counts")
    st_model = SpatialStereoscope.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=2, accelerator="cpu")
    assert st_model.is_trained


def test_spatial_stereoscope_get_proportions(stereo_data):
    sc_adata, st_adata = stereo_data
    sc_model = _train_rna_model(sc_adata)

    SpatialStereoscope.setup_anndata(st_adata, layer="counts")
    st_model = SpatialStereoscope.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=2, accelerator="cpu")

    props = st_model.get_proportions()
    import pandas as pd

    assert isinstance(props, pd.DataFrame)
    assert props.shape == (st_adata.n_obs, 4)  # n_labels
    np.testing.assert_allclose(props.sum(axis=1).values, 1.0, atol=1e-5)


def test_spatial_stereoscope_get_proportions_df_via_mixin(stereo_data):
    """SpatialDeconvolutionMixin.get_proportions_df wraps get_proportions correctly."""
    sc_adata, st_adata = stereo_data
    sc_model = _train_rna_model(sc_adata)

    SpatialStereoscope.setup_anndata(st_adata, layer="counts")
    st_model = SpatialStereoscope.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=2, accelerator="cpu")

    df = st_model.get_proportions_df()
    assert df.shape[1] == 4
    np.testing.assert_allclose(df.sum(axis=1).values, 1.0, atol=1e-5)


def test_spatial_stereoscope_get_scale_for_ct(stereo_data):
    sc_adata, st_adata = stereo_data
    sc_model = _train_rna_model(sc_adata)

    SpatialStereoscope.setup_anndata(st_adata, layer="counts")
    st_model = SpatialStereoscope.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=2, accelerator="cpu")

    cell_types = np.array([st_model.cell_type_mapping[0]])
    expr = st_model.get_scale_for_ct(cell_types)
    assert expr.shape == (1, st_adata.n_vars)


def test_spatial_stereoscope_spatialdata_methods():
    """setup_spatialdata and from_spatialdata must be inherited from SpatialBaseModel."""
    assert hasattr(SpatialStereoscope, "setup_spatialdata")
    assert hasattr(SpatialStereoscope, "from_spatialdata")
    assert hasattr(RNAStereoscope, "setup_spatialdata")


def test_rna_stereoscope_ct_weight(stereo_data):
    """ct_weight kwarg must be read from model_kwargs['ct_weight'], not ['ct_prop']."""
    sc_adata, _ = stereo_data
    ct_weight = np.array([1.0, 2.0, 1.0, 0.5], dtype=np.float32)
    RNAStereoscope.setup_anndata(sc_adata, labels_key="labels", layer="counts")
    # This must not raise KeyError
    sc_model = RNAStereoscope(sc_adata, ct_weight=ct_weight)

    np.testing.assert_allclose(sc_model.module.ct_weight.numpy(), ct_weight)
    sc_model.train(max_epochs=2, accelerator="cpu")
    assert sc_model.is_trained


def test_stereoscope_external_import():
    """Must be accessible from spatialvi.external namespace."""
    from spatialvi.external import RNAStereoscope as R
    from spatialvi.external import SpatialStereoscope as S

    assert R is RNAStereoscope
    assert S is SpatialStereoscope
