"""Regression tests: spatialvi.external.SpatialStereoscope vs scvi.external.SpatialStereoscope.

Each test runs the same operations with the same seed and asserts outputs match.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scvi.data import synthetic_iid
from scvi.external import RNAStereoscope as ScviRNA
from scvi.external import SpatialStereoscope as ScviSpatial

from spatialvi.external.stereoscope._model import RNAStereoscope as SpatialRNA
from spatialvi.external.stereoscope._model import SpatialStereoscope as SpatialSpatial

SEED = 42
N_EPOCHS = 2
N_LABELS = 4
N_GENES = 50


def _make_data(seed=SEED):
    np.random.seed(seed)
    sc_adata = synthetic_iid(n_labels=N_LABELS, n_genes=N_GENES, sparse_format=None)
    sc_adata.layers["counts"] = sc_adata.X.copy()
    st_adata = synthetic_iid(n_genes=N_GENES, sparse_format=None)
    st_adata.layers["counts"] = st_adata.X.copy()
    st_adata.var_names = sc_adata.var_names
    return sc_adata, st_adata


def _train_scvi(sc_adata, st_adata, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ScviRNA.setup_anndata(sc_adata, labels_key="labels", layer="counts")
    sc_model = ScviRNA(sc_adata)
    sc_model.train(max_epochs=N_EPOCHS, accelerator="cpu")

    ScviSpatial.setup_anndata(st_adata, layer="counts")
    st_model = ScviSpatial.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=N_EPOCHS, accelerator="cpu")
    return sc_model, st_model


def _train_spatialvi(sc_adata, st_adata, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    SpatialRNA.setup_anndata(sc_adata, labels_key="labels", layer="counts")
    sc_model = SpatialRNA(sc_adata)
    sc_model.train(max_epochs=N_EPOCHS, accelerator="cpu")

    SpatialSpatial.setup_anndata(st_adata, layer="counts")
    st_model = SpatialSpatial.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=N_EPOCHS, accelerator="cpu")
    return sc_model, st_model


@pytest.fixture(scope="module")
def both_models():
    sc_adata, st_adata = _make_data()
    scvi_rna, scvi_st = _train_scvi(sc_adata.copy(), st_adata.copy())
    spatialvi_rna, spatialvi_st = _train_spatialvi(sc_adata.copy(), st_adata.copy())
    return scvi_st, spatialvi_st, st_adata


def test_stereoscope_both_trained(both_models):
    scvi_st, spatialvi_st, _ = both_models
    assert scvi_st.is_trained
    assert spatialvi_st.is_trained


def test_stereoscope_proportions_shape_match(both_models):
    scvi_st, spatialvi_st, st_adata = both_models

    scvi_props = scvi_st.get_proportions()
    spatialvi_props = spatialvi_st.get_proportions()

    assert scvi_props.shape == spatialvi_props.shape
    assert scvi_props.shape == (st_adata.n_obs, N_LABELS)


def test_stereoscope_proportions_sum_to_one(both_models):
    _, spatialvi_st, st_adata = both_models
    props = spatialvi_st.get_proportions()
    np.testing.assert_allclose(props.sum(axis=1).values, 1.0, atol=1e-5)


def test_stereoscope_column_names_match(both_models):
    scvi_st, spatialvi_st, _ = both_models
    scvi_props = scvi_st.get_proportions()
    spatialvi_props = spatialvi_st.get_proportions()
    assert list(scvi_props.columns) == list(spatialvi_props.columns)


def test_stereoscope_get_proportions_df_mixin(both_models):
    """get_proportions_df from SpatialDeconvolutionMixin must match get_proportions output."""
    _, spatialvi_st, _ = both_models
    df_via_mixin = spatialvi_st.get_proportions_df()
    df_direct = spatialvi_st.get_proportions()

    assert df_via_mixin.shape == df_direct.shape
    np.testing.assert_allclose(df_via_mixin.values, df_direct.values, atol=1e-8)


def test_stereoscope_module_same_structure():
    """Both RNADeconv and SpatialDeconv must have same public interface."""
    from scvi.external.stereoscope._module import RNADeconv as ScviRNADeconv
    from scvi.external.stereoscope._module import SpatialDeconv as ScviSpatialDeconv

    from spatialvi.external.stereoscope._module import RNADeconv as SpatialRNADeconv
    from spatialvi.external.stereoscope._module import SpatialDeconv as SpatialSpatialDeconv

    for ScviCls, SpatialCls in [
        (ScviRNADeconv, SpatialRNADeconv),
        (ScviSpatialDeconv, SpatialSpatialDeconv),
    ]:
        scvi_methods = {m for m in dir(ScviCls) if not m.startswith("__")}
        spatialvi_methods = {m for m in dir(SpatialCls) if not m.startswith("__")}
        missing = scvi_methods - spatialvi_methods
        assert not missing, f"{ScviCls.__name__}: methods missing in spatialvi: {missing}"
