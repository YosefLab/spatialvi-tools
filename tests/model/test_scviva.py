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


def test_scviva_save_load(scviva_adata, tmp_path):
    """Save/load round-trip preserves history, latent, and downstream API outputs."""
    SCVIVA.preprocessing_anndata(scviva_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(scviva_adata, layer="counts", batch_key="batch", **setup_kwargs)
    model = SCVIVA(scviva_adata, prior_mixture=False)
    model.train(
        max_epochs=N_EPOCHS, enable_progress_bar=False, train_size=0.8, validation_size=0.2
    )

    hist_elbo = model.history["elbo_train"].copy()
    latent = model.get_latent_representation()
    assert latent.shape == (scviva_adata.n_obs, model.module.n_latent)

    save_path = str(tmp_path / "test_scviva")
    model.save(save_path, save_anndata=True, overwrite=True)
    model2 = SCVIVA.load(save_path)

    np.testing.assert_array_equal(model2.history_["elbo_train"], hist_elbo)
    latent2 = model2.get_latent_representation()
    assert np.allclose(latent, latent2, atol=1e-5)

    model2.get_elbo(indices=model2.validation_indices)
    model2.get_composition_error(return_mean=False, indices=model2.validation_indices)
    model2.get_niche_error(return_mean=False, indices=model2.validation_indices)
    model2.get_normalized_expression()

    pred_alpha = model2.predict_neighborhood()
    assert pred_alpha.shape == (scviva_adata.n_obs, model2.n_labels)
    assert np.allclose(pred_alpha.sum(), scviva_adata.n_obs, atol=1e-4)

    pred_eta = model2.predict_niche_activation()
    n_latent_intrinsic = scviva_adata.obsm["X_scVI"].shape[1]
    assert pred_eta.shape == (scviva_adata.n_obs, model2.n_labels, n_latent_intrinsic)


@pytest.mark.optional
def test_scviva_differential(scviva_adata):
    """differential_expression returns DifferentialExpressionResults with a fitted GPC."""
    from sklearn.gaussian_process import GaussianProcessClassifier

    from scviva.model.utils._scviva_de import DifferentialExpressionResults

    SCVIVA.preprocessing_anndata(scviva_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(scviva_adata, layer="counts", batch_key="batch", **setup_kwargs)
    model = SCVIVA(scviva_adata, prior_mixture=False)
    model.train(max_epochs=N_EPOCHS, enable_progress_bar=False, logger=False)

    # Two-group niche-unaware DE
    model.differential_expression(
        groupby="labels",
        group1="label_0",
        group2="label_1",
        batch_correction=False,
        niche_mode=False,
        fdr_target=1,
        delta=0.5,
    )

    # One-group vs rest with kNN
    model.differential_expression(
        groupby="labels",
        group1="label_0",
        batch_correction=False,
        radius=None,
        k_nn=K_NN,
        fdr_target=1,
        delta=0.5,
    )

    # Full DE with list fdr_target / delta → DifferentialExpressionResults
    de_results = model.differential_expression(
        groupby="labels",
        group1="label_0",
        group2="label_1",
        batch_correction=False,
        radius=None,
        k_nn=K_NN,
        fdr_target=[1, 1, 1, 1],
        delta=[0.5, 0.5, 0.5, 0.5],
    )

    assert isinstance(de_results, DifferentialExpressionResults)
    assert isinstance(de_results.gpc, GaussianProcessClassifier)
    assert hasattr(de_results.gpc, "log_marginal_likelihood_value_")

    import matplotlib.pyplot as plt

    _orig_show = plt.show
    plt.show = lambda: None
    try:
        de_results.plot(show_plot=False)
    finally:
        plt.show = _orig_show

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    try:
        de_results.plot(path_to_save=path, show_plot=False)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        os.remove(path)


@pytest.fixture(scope="module")
def split_adata(scviva_adata):
    """Split adata into ref (left) / query (right) by median x-coordinate."""
    SCVIVA.preprocessing_anndata(scviva_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(scviva_adata, layer="counts", batch_key="batch", **setup_kwargs)
    median_x = np.median(scviva_adata.obsm["spatial"][:, 0])
    ref = scviva_adata[scviva_adata.obsm["spatial"][:, 0] <= median_x].copy()
    query = scviva_adata[scviva_adata.obsm["spatial"][:, 0] > median_x].copy()
    return ref, query


@pytest.mark.optional
def test_scviva_scarches_less_features(split_adata):
    """scArches query workflow succeeds when query has fewer genes than reference."""
    ref_adata, query_adata = split_adata

    SCVIVA.preprocessing_anndata(ref_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(ref_adata, layer="counts", batch_key="batch", **setup_kwargs)
    model = SCVIVA(ref_adata, prior_mixture=False)
    model.train(max_epochs=N_EPOCHS, enable_progress_bar=False, logger=False)
    assert model.is_trained

    # Drop 5 genes from query
    query_adata = query_adata[
        :, np.random.default_rng(42).permutation(query_adata.var_names)[:-5]
    ].copy()
    query_adata.obs["labels"] = (
        query_adata.obs["labels"].astype(str).replace("label_2", "label_0").astype("category")
    )

    model.preprocessing_query_anndata(
        query_adata, reference_model=model, k_nn=K_NN, **setup_kwargs
    )
    query_model = model.load_query_data(query_adata, reference_model=model)
    query_model.train(max_epochs=N_EPOCHS, enable_progress_bar=False, logger=False)

    pred = query_model.predict_neighborhood(query_adata)
    assert pred.shape == (query_adata.n_obs, query_model.n_labels)
    query_adata.obsm["X_scVIVA"] = query_model.get_latent_representation(query_adata)
    assert query_model.get_latent_representation(adata=ref_adata).shape[0] == ref_adata.n_obs
    assert query_model.get_latent_representation(adata=query_adata).shape[0] == query_adata.n_obs


@pytest.mark.optional
def test_scviva_scarches_same_features(split_adata):
    """scArches query workflow succeeds when query has the same gene set as reference."""
    ref_adata, query_adata = split_adata

    SCVIVA.preprocessing_anndata(ref_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(ref_adata, layer="counts", batch_key="batch", **setup_kwargs)
    model = SCVIVA(ref_adata, prior_mixture=False)
    model.train(max_epochs=N_EPOCHS, enable_progress_bar=False, logger=False)

    # Same genes, just shuffled
    query_adata = query_adata[
        :, np.random.default_rng(7).permutation(query_adata.var_names)
    ].copy()

    model.preprocessing_query_anndata(
        query_adata, reference_model=model, k_nn=K_NN, **setup_kwargs
    )
    query_model = model.load_query_data(query_adata, reference_model=model)
    query_model.train(max_epochs=N_EPOCHS, enable_progress_bar=False, logger=False)

    pred = query_model.predict_neighborhood(query_adata)
    assert pred.shape == (query_adata.n_obs, query_model.n_labels)
    assert query_model.get_latent_representation(adata=ref_adata).shape[0] == ref_adata.n_obs
    assert query_model.get_latent_representation(adata=query_adata).shape[0] == query_adata.n_obs


@pytest.mark.parametrize("dispersion", ["gene", "gene-batch", "gene-label", "gene-cell"])
def test_scviva_dispersion(scviva_adata, dispersion):
    """All dispersion modes must train without error."""
    SCVIVA.preprocessing_anndata(scviva_adata, k_nn=K_NN, **setup_kwargs)
    SCVIVA.setup_anndata(scviva_adata, layer="counts", batch_key="batch", **setup_kwargs)
    model = SCVIVA(scviva_adata, dispersion=dispersion)
    model.train(max_epochs=N_EPOCHS, enable_progress_bar=False, logger=False)
    assert model.is_trained
