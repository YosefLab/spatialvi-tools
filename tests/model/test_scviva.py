import numpy as np
import pytest
from scvi.data import synthetic_iid

from scviva.model._scviva import SCVIVA

N_LATENT = 10
N_EPOCHS = 1
K_NN = 5

# Default key names used by SCVIVA (matching the actual defaults)
setup_kwargs = {
    "sample_key": "batch",
    "labels_key": "labels",
    "cell_coordinates_key": "spatial",
    "expression_embedding_key": "X_scVI",
    "expression_embedding_niche_key": "niche_activation",
    "niche_composition_key": "niche_composition",
    "niche_indexes_key": "niche_indexes",
    "niche_distances_key": "niche_distances",
}


@pytest.fixture(scope="session")
def scviva_adata():
    adata = synthetic_iid(
        batch_size=128,
        n_genes=50,
        n_batches=2,
        n_labels=3,
        dropout_ratio=0.3,
    )
    n = adata.n_obs
    # Add required obsm keys with default names
    adata.obsm["spatial"] = np.random.rand(n, 2)
    adata.obsm["X_scVI"] = np.random.normal(size=(n, 20))
    # Add integer counts layer
    raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    adata.layers["counts"] = np.abs(raw).astype(int)
    return adata


def test_scviva_train(scviva_adata):
    SCVIVA.preprocessing_anndata(scviva_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(scviva_adata, layer="counts", batch_key="batch", **setup_kwargs)
    model = SCVIVA(scviva_adata, prior_mixture=False)
    model.train(max_epochs=N_EPOCHS)
    assert model.is_trained


def test_scviva_get_latent_cpu(scviva_adata):
    SCVIVA.preprocessing_anndata(scviva_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(scviva_adata, layer="counts", batch_key="batch", **setup_kwargs)
    model = SCVIVA(scviva_adata)
    model.train(max_epochs=N_EPOCHS, accelerator="cpu")
    latent = model.get_latent_representation(backend="cpu")
    assert latent.shape[0] == scviva_adata.n_obs


def test_scviva_compute_neighbors(scviva_adata):
    pytest.importorskip("squidpy")
    SCVIVA.preprocessing_anndata(scviva_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(scviva_adata, layer="counts", **setup_kwargs)
    model = SCVIVA(scviva_adata)
    model.compute_neighbors(scviva_adata, spatial_key="spatial", n_neighs=5)
    assert "index_neighbor" in scviva_adata.obsm


def _trained_model(adata):
    SCVIVA.preprocessing_anndata(adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(adata, layer="counts", batch_key="batch", **setup_kwargs)
    model = SCVIVA(adata, prior_mixture=False)
    model.train(max_epochs=N_EPOCHS, accelerator="cpu")
    return model


def test_scviva_normalized_expression(scviva_adata):
    model = _trained_model(scviva_adata)
    expr = model.get_normalized_expression(return_numpy=True)
    assert expr.shape == (scviva_adata.n_obs, scviva_adata.n_vars)
    assert not np.isnan(expr).any()


def test_scviva_importance_weighting_path(scviva_adata):
    """SCVIVA has no standalone importance method; importance is a weights kwarg."""
    model = _trained_model(scviva_adata)
    # Importance-sampling expression is not a separate method on PyTorch models;
    # it is obtained via get_normalized_expression(weights="importance").
    assert not hasattr(model, "get_normalized_expression_importance")
    expr = model.get_normalized_expression(n_samples=3, weights="importance", return_numpy=True)
    assert expr.shape == (scviva_adata.n_obs, scviva_adata.n_vars)
    assert not np.isnan(expr).any()


def test_scviva_neighbor_abundance(scviva_adata):
    model = _trained_model(scviva_adata)
    df = model.get_neighbor_abundance()
    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == scviva_adata.n_obs
    assert not df.isnull().any().any()


def test_scviva_differential_expression(scviva_adata):
    model = _trained_model(scviva_adata)
    import pandas as pd

    de = model.differential_expression(
        groupby="labels",
        group1="label_0",
        group2="label_1",
        batch_correction=False,
        niche_mode=False,
        fdr_target=1,
        delta=0.5,
    )
    assert isinstance(de, pd.DataFrame)
    assert de.shape[0] == scviva_adata.n_vars


def test_scviva_differential_abundance(scviva_adata):
    import pandas as pd

    model = _trained_model(scviva_adata)
    # differential_abundance stores results in adata.obsm["da_log_probs"] (returns None)
    model.differential_abundance(sample_key="batch")
    assert "da_log_probs" in scviva_adata.obsm
    assert isinstance(scviva_adata.obsm["da_log_probs"], pd.DataFrame)
