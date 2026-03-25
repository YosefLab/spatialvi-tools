"""Regression tests: spatialvi.DestVI vs scvi.model.DestVI.

Each test runs the exact same operations on the same data with the same random seed
on both implementations and asserts outputs are identical (within float tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scvi.data import synthetic_iid
from scvi.model import CondSCVI
from scvi.model import DestVI as ScviDestVI

from spatialvi.model import DestVI as SpatialDestVI

SEED = 42
N_EPOCHS = 2
N_LATENT = 2
N_LABELS = 5


@pytest.fixture(scope="module")
def sc_adata():
    """Single-cell reference dataset."""
    np.random.seed(SEED)
    adata = synthetic_iid(n_labels=N_LABELS, sparse_format=None)
    adata.layers["counts"] = adata.X.copy()
    return adata


@pytest.fixture(scope="module")
def st_adata(sc_adata):
    """Spatial (spot) dataset with same genes."""
    np.random.seed(SEED + 1)
    adata = synthetic_iid(n_labels=N_LABELS, sparse_format=None)
    adata.layers["counts"] = adata.X.copy()
    # Match vars with sc_adata
    adata = adata[:, sc_adata.var_names].copy()
    return adata


@pytest.fixture(scope="module")
def condscvi_model(sc_adata):
    """Shared trained CondSCVI reference model."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="labels")
    model = CondSCVI(sc_adata, n_latent=N_LATENT, prior="mog")
    model.train(max_epochs=N_EPOCHS)
    return model


def _train_scvi_destvi(st_adata, condscvi_model, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ScviDestVI.setup_anndata(st_adata, layer="counts")
    model = ScviDestVI.from_rna_model(st_adata, condscvi_model)
    model.train(max_epochs=N_EPOCHS)
    return model


def _train_spatial_destvi(st_adata, condscvi_model, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    SpatialDestVI.setup_anndata(st_adata, layer="counts")
    model = SpatialDestVI.from_rna_model(st_adata, condscvi_model)
    model.train(max_epochs=N_EPOCHS)
    return model


def test_destvi_trains(st_adata, condscvi_model):
    """Both implementations must train without errors."""
    scvi_model = _train_scvi_destvi(st_adata, condscvi_model)
    spatial_model = _train_spatial_destvi(st_adata, condscvi_model)
    assert scvi_model.is_trained
    assert spatial_model.is_trained


def test_destvi_proportions_match(st_adata, condscvi_model):
    """Cell type proportions must match between implementations."""
    scvi_model = _train_scvi_destvi(st_adata, condscvi_model)
    spatial_model = _train_spatial_destvi(st_adata, condscvi_model)

    torch.manual_seed(SEED)
    props_scvi = scvi_model.get_proportions()
    torch.manual_seed(SEED)
    props_spatial = spatial_model.get_proportions()

    assert props_scvi.shape == props_spatial.shape, (
        f"Proportions shape mismatch: scvi={props_scvi.shape}, spatialvi={props_spatial.shape}"
    )
    np.testing.assert_allclose(
        props_scvi,
        props_spatial,
        atol=1e-5,
        err_msg="Cell type proportions differ between scvi and spatialvi DestVI",
    )


def test_destvi_proportions_df(st_adata, condscvi_model):
    """spatialvi-specific get_proportions_df must return correct shape and columns."""
    spatial_model = _train_spatial_destvi(st_adata, condscvi_model)
    df = spatial_model.get_proportions_df()

    assert df.shape[0] == st_adata.n_obs, "DataFrame rows must equal number of spots"
    assert df.shape[1] == N_LABELS, f"Expected {N_LABELS} cell types, got {df.shape[1]}"
    np.testing.assert_allclose(
        df.values.sum(axis=1),
        np.ones(st_adata.n_obs),
        atol=1e-4,
        err_msg="Proportions must sum to 1 per spot",
    )


def test_destvi_elbo_matches(st_adata, condscvi_model):
    """Training ELBO history must match between implementations."""
    scvi_model = _train_scvi_destvi(st_adata, condscvi_model)
    spatial_model = _train_spatial_destvi(st_adata, condscvi_model)

    elbo_scvi = np.array(scvi_model.history_["elbo_train"].values, dtype=float)
    elbo_spatial = np.array(spatial_model.history_["elbo_train"].values, dtype=float)

    assert len(elbo_scvi) == len(elbo_spatial)
    np.testing.assert_allclose(
        elbo_scvi,
        elbo_spatial,
        atol=1e-4,
        err_msg="Training ELBO differs between scvi and spatialvi DestVI",
    )


def test_destvi_save_load_matches(st_adata, condscvi_model, tmp_path):
    """Save/load round-trip must produce identical proportions."""
    spatial_model = _train_spatial_destvi(st_adata, condscvi_model)
    props_before = spatial_model.get_proportions()

    save_path = str(tmp_path / "destvi_model")
    spatial_model.save(save_path, save_anndata=True, overwrite=True)
    loaded = SpatialDestVI.load(save_path)
    props_after = loaded.get_proportions()

    np.testing.assert_array_equal(
        props_before.values,
        props_after.values,
        err_msg="Proportions changed after save/load",
    )
