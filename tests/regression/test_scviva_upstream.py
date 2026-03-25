"""Regression tests: spatialvi.SCVIVA vs scvi.external.scviva.SCVIVA.

Each test runs the exact same operations on the same data with the same random seed
on both implementations and asserts outputs are identical (within float tolerance).
"""

from __future__ import annotations

import numpy as np
import pytest
import scvi as scvi_pkg
import torch
from scvi.data import synthetic_iid
from scvi.external.scviva import SCVIVA as ScviSCVIVA

from spatialvi.model import SCVIVA as SpatialSCVIVA

SEED = 42
N_LATENT_INTRINSIC = 20
K_NN = 5
N_EPOCHS = 2
LABELS_KEY = "labels"

SETUP_KWARGS = {
    "sample_key": "batch",
    "labels_key": LABELS_KEY,
    "cell_coordinates_key": "coordinates",
    "expression_embedding_key": "qz1_m",
    "expression_embedding_niche_key": "qz1_m_niche_ct",
    "niche_composition_key": "neighborhood_composition",
    "niche_indexes_key": "niche_indexes",
    "niche_distances_key": "niche_distances",
}

MODEL_KWARGS = {
    "prior_mixture": False,
    "semisupervised": True,
    "linear_classifier": True,
}

TRAIN_KWARGS = {
    "max_epochs": N_EPOCHS,
    "train_size": 1.0,
    "validation_size": 0.0,
    "early_stopping": False,
    "accelerator": "cpu",
}


@pytest.fixture(scope="module")
def adata():
    np.random.seed(SEED)
    adata = synthetic_iid(
        batch_size=128,
        n_genes=50,
        n_proteins=0,
        n_regions=0,
        n_batches=2,
        n_labels=3,
        dropout_ratio=0.5,
        generate_coordinates=True,
        sparse_format=None,
        return_mudata=False,
    )
    adata.obsm["qz1_m"] = np.random.normal(size=(adata.shape[0], N_LATENT_INTRINSIC))
    adata.layers["counts"] = adata.X.copy()
    return adata


def _train_scvi(adata, seed=SEED):
    scvi_pkg.settings.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    ScviSCVIVA.preprocessing_anndata(adata, k_nn=K_NN, **SETUP_KWARGS)
    ScviSCVIVA.setup_anndata(adata, layer="counts", batch_key="batch", **SETUP_KWARGS)
    model = ScviSCVIVA(adata, **MODEL_KWARGS)
    model.train(**TRAIN_KWARGS)
    return model


def _train_spatialvi(adata, seed=SEED):
    scvi_pkg.settings.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    SpatialSCVIVA.preprocessing_anndata(adata, k_nn=K_NN, **SETUP_KWARGS)
    SpatialSCVIVA.setup_anndata(adata, layer="counts", batch_key="batch", **SETUP_KWARGS)
    model = SpatialSCVIVA(adata, **MODEL_KWARGS)
    model.train(**TRAIN_KWARGS)
    return model


def test_scviva_trains(adata):
    """Both implementations must train without errors."""
    scvi_model = _train_scvi(adata)
    spatial_model = _train_spatialvi(adata)
    assert scvi_model.is_trained
    assert spatial_model.is_trained


def test_scviva_latent_representation_matches(adata):
    """Latent representations must be identical given same seed and data."""
    scvi_model = _train_scvi(adata)
    spatial_model = _train_spatialvi(adata)

    latent_scvi = scvi_model.get_latent_representation()
    latent_spatial = spatial_model.get_latent_representation()

    assert latent_scvi.shape == latent_spatial.shape, (
        f"Shape mismatch: scvi={latent_scvi.shape}, spatialvi={latent_spatial.shape}"
    )
    np.testing.assert_allclose(
        latent_scvi,
        latent_spatial,
        atol=1e-5,
        err_msg="Latent representations differ between scvi and spatialvi SCVIVA",
    )


def test_scviva_elbo_matches(adata):
    """Training ELBO history must match between implementations."""
    scvi_model = _train_scvi(adata)
    spatial_model = _train_spatialvi(adata)

    elbo_scvi = np.array(scvi_model.history["elbo_train"].values, dtype=float)
    elbo_spatial = np.array(spatial_model.history["elbo_train"].values, dtype=float)

    assert len(elbo_scvi) == len(elbo_spatial)
    np.testing.assert_allclose(
        elbo_scvi,
        elbo_spatial,
        atol=1e-4,
        err_msg="Training ELBO differs between scvi and spatialvi SCVIVA",
    )


def test_scviva_predict_neighborhood_matches(adata):
    """Predicted neighborhood compositions must match between implementations."""
    scvi_model = _train_scvi(adata)
    spatial_model = _train_spatialvi(adata)

    torch.manual_seed(SEED)
    alpha_scvi = scvi_model.predict_neighborhood()
    torch.manual_seed(SEED)
    alpha_spatial = spatial_model.predict_neighborhood()

    assert alpha_scvi.shape == alpha_spatial.shape
    np.testing.assert_allclose(
        alpha_scvi,
        alpha_spatial,
        atol=1e-5,
        err_msg="Neighborhood predictions differ between scvi and spatialvi SCVIVA",
    )


def test_scviva_save_load_matches(adata, tmp_path):
    """Save/load round-trip must produce identical latent representations."""
    spatial_model = _train_spatialvi(adata)
    latent_before = spatial_model.get_latent_representation()

    save_path = str(tmp_path / "scviva_model")
    spatial_model.save(save_path, save_anndata=True, overwrite=True)
    loaded = SpatialSCVIVA.load(save_path)
    latent_after = loaded.get_latent_representation()

    np.testing.assert_array_equal(
        latent_before,
        latent_after,
        err_msg="Latent representation changed after save/load",
    )


def test_scviva_composition_error_matches(adata):
    """Composition and niche errors must have the same shape in both implementations."""
    scvi_model = _train_scvi(adata)
    spatial_model = _train_spatialvi(adata)

    err_scvi = scvi_model.get_composition_error(return_mean=False)
    err_spatial = spatial_model.get_composition_error(return_mean=False)
    assert err_scvi.shape == err_spatial.shape

    niche_scvi = scvi_model.get_niche_error(return_mean=False)
    niche_spatial = spatial_model.get_niche_error(return_mean=False)
    assert niche_scvi.shape == niche_spatial.shape


@pytest.mark.parametrize("dispersion", ["gene", "gene-batch", "gene-label", "gene-cell"])
def test_scviva_dispersion_matches(adata, dispersion):
    """Both implementations must train successfully with each dispersion mode."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ScviSCVIVA.preprocessing_anndata(adata, k_nn=K_NN, **SETUP_KWARGS)
    ScviSCVIVA.setup_anndata(adata, layer="counts", batch_key="batch", **SETUP_KWARGS)
    scvi_model = ScviSCVIVA(adata, dispersion=dispersion)
    scvi_model.train(max_epochs=1)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    SpatialSCVIVA.preprocessing_anndata(adata, k_nn=K_NN, **SETUP_KWARGS)
    SpatialSCVIVA.setup_anndata(adata, layer="counts", batch_key="batch", **SETUP_KWARGS)
    spatial_model = SpatialSCVIVA(adata, dispersion=dispersion)
    spatial_model.train(max_epochs=1)

    lat_scvi = scvi_model.get_latent_representation()
    lat_spatial = spatial_model.get_latent_representation()
    assert lat_scvi.shape == lat_spatial.shape
