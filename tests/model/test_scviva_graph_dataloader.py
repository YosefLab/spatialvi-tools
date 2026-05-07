"""GraphDataLoader integration tests for SCVIVA."""

import time
import warnings

import numpy as np
import pytest
from scvi.data import synthetic_iid
from scvi.dataloaders import DataSplitter

from scviva._constants import SCVIVA_REGISTRY_KEYS
from scviva.dataloaders import GraphDataSplitter
from scviva.model import SCVIVA

K_NN = 3

SETUP_KWARGS = {
    "sample_key": "batch",
    "labels_key": "labels",
    "cell_coordinates_key": "spatial",
    "expression_embedding_key": "X_scVI",
    "expression_embedding_niche_key": "niche_activation",
    "niche_composition_key": "niche_composition",
    "niche_indexes_key": "niche_indexes",
    "niche_distances_key": "niche_distances",
}


def _prepare_scviva_adata(cls=SCVIVA):
    adata = synthetic_iid(
        batch_size=64,
        n_genes=20,
        n_batches=2,
        n_labels=3,
        dropout_ratio=0.3,
    )
    raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    adata.layers["counts"] = np.abs(raw).astype(int)

    # Filter cells with all-zero counts: log(0) in the encoder produces NaN
    cell_counts = np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
    adata = adata[cell_counts > 0].copy()

    n_obs = adata.n_obs
    adata.obsm["spatial"] = np.random.default_rng(0).random((n_obs, 2))
    adata.obsm["X_scVI"] = np.random.default_rng(1).normal(size=(n_obs, 10))
    cls.preprocessing_anndata(adata, k_nn=K_NN, **SETUP_KWARGS)
    cls.setup_anndata(adata, layer="counts", batch_key="batch", **SETUP_KWARGS)
    return adata


def test_scviva_uses_graph_datasplitter_by_default():
    """SCVIVA should opt into graph-aware training batches."""
    assert SCVIVA._data_splitter_cls is GraphDataSplitter


def _scviva_legacy_cls():
    """Return an SCVIVA subclass using plain DataSplitter (AnnDataLoader path)."""

    class SCVIVALegacy(SCVIVA):
        _data_splitter_cls = DataSplitter

    return SCVIVALegacy


def _train_legacy(model, max_epochs=2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(max_epochs=max_epochs, enable_progress_bar=False)


def _train_graph(model, max_epochs=2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(max_epochs=max_epochs, enable_progress_bar=False)


def test_scviva_train_forwards_niche_graph_defaults():
    """SCVIVA train should map GraphDataSplitter to niche index/distance fields."""

    class RecordingGraphDataSplitter(GraphDataSplitter):
        recorded_kwargs = None

        def __init__(self, adata_manager, **kwargs):
            type(self).recorded_kwargs = dict(kwargs)
            super().__init__(adata_manager, **kwargs)

    class SCVIVARecording(SCVIVA):
        _data_splitter_cls = RecordingGraphDataSplitter

    adata = _prepare_scviva_adata(SCVIVARecording)
    model = SCVIVARecording(adata, prior_mixture=False)
    model.train(
        max_epochs=1,
        train_size=1.0,
        validation_size=0.0,
        enable_progress_bar=False,
        logger=False,
    )

    assert model.is_trained
    recorded = RecordingGraphDataSplitter.recorded_kwargs
    assert recorded["neighbor_indices_key"] == SCVIVA_REGISTRY_KEYS.NICHE_INDEXES_KEY
    assert recorded["edge_obsm_keys"] == [SCVIVA_REGISTRY_KEYS.NICHE_DISTANCES_KEY]
    assert recorded["load_neighbor_expression"] is False


def test_scviva_graph_train_completes():
    """SCVIVA with default GraphDataSplitter must reach is_trained and produce valid latent."""
    adata = _prepare_scviva_adata()
    model = SCVIVA(adata, prior_mixture=False)
    _train_graph(model)
    assert model.is_trained

    latent = model.get_latent_representation()
    assert latent.shape == (adata.n_obs, model.module.n_latent)
    assert np.isfinite(latent).all()


def test_scviva_graph_downstream():
    """Downstream SCVIVA APIs must work after graph-path training."""
    adata = _prepare_scviva_adata()
    model = SCVIVA(adata, prior_mixture=False)
    _train_graph(model)

    model.get_normalized_expression()
    model.get_composition_error(return_mean=True)
    model.get_niche_error(return_mean=True)

    pred_alpha = model.predict_neighborhood()
    assert pred_alpha.shape == (adata.n_obs, model.n_labels)
    assert np.allclose(pred_alpha.sum(), adata.n_obs, atol=1e-4)

    pred_eta = model.predict_niche_activation()
    assert pred_eta.shape == (adata.n_obs, model.n_labels, model.module.n_latent)


@pytest.mark.optional
def test_scviva_graph_save_load(tmp_path):
    """Graph-path training preserves save/load round-trip."""
    adata = _prepare_scviva_adata()
    model = SCVIVA(adata, prior_mixture=False)
    _train_graph(model)

    hist_elbo = model.history_["elbo_train"].copy()
    latent = model.get_latent_representation()

    save_path = str(tmp_path / "test_scviva_graph")
    model.save(save_path, save_anndata=True, overwrite=True)
    model2 = SCVIVA.load(save_path)

    np.testing.assert_array_equal(model2.history_["elbo_train"], hist_elbo)
    latent2 = model2.get_latent_representation()
    assert np.allclose(latent, latent2, atol=1e-5)


@pytest.mark.optional
def test_scviva_graph_differential_expression():
    """differential_expression runs on graph-trained SCVIVA."""
    adata = _prepare_scviva_adata()
    model = SCVIVA(adata, prior_mixture=False)
    _train_graph(model, max_epochs=2)

    model.differential_expression(
        groupby="labels",
        group1="label_0",
        group2="label_1",
        batch_correction=False,
        niche_mode=False,
        fdr_target=1,
        delta=0.5,
    )


@pytest.mark.optional
def test_scviva_graph_scarches():
    """scArches query workflow works after graph-path training."""
    adata = _prepare_scviva_adata()
    coords = adata.obsm["spatial"]
    median_x = np.median(coords[:, 0])
    ref = adata[coords[:, 0] <= median_x].copy()
    query = adata[coords[:, 0] > median_x].copy()

    SCVIVACls = SCVIVA
    SCVIVACls.preprocessing_anndata(ref, k_nn=K_NN, **SETUP_KWARGS)
    SCVIVACls.setup_anndata(ref, layer="counts", batch_key="batch", **SETUP_KWARGS)
    model = SCVIVACls(ref, prior_mixture=False)
    _train_graph(model)

    model.preprocessing_query_anndata(query, reference_model=model, k_nn=K_NN, **SETUP_KWARGS)
    query_model = model.load_query_data(query, reference_model=model)
    _train_graph(query_model, max_epochs=1)

    pred = query_model.predict_neighborhood(query)
    assert pred.shape == (query.n_obs, query_model.n_labels)


@pytest.mark.benchmark
def test_scviva_dataloader_speed_comparison():
    """Side-by-side wall-clock comparison: AnnDataLoader (legacy) vs GraphDataLoader."""
    SCVIVALegacy = _scviva_legacy_cls()
    n_epochs = 5

    adata_legacy = _prepare_scviva_adata(SCVIVALegacy)
    model_ann = SCVIVALegacy(adata_legacy, prior_mixture=False)
    t0 = time.perf_counter()
    _train_legacy(model_ann, max_epochs=n_epochs)
    t_ann = time.perf_counter() - t0

    adata_graph = _prepare_scviva_adata()
    model_graph = SCVIVA(adata_graph, prior_mixture=False)
    t0 = time.perf_counter()
    _train_graph(model_graph, max_epochs=n_epochs)
    t_graph = time.perf_counter() - t0

    print(f"\nAnnDataLoader:   {t_ann:.2f}s total  ({t_ann / n_epochs:.3f}s/epoch)")
    print(f"GraphDataLoader: {t_graph:.2f}s total  ({t_graph / n_epochs:.3f}s/epoch)")
    print(f"Ratio (graph/ann): {t_graph / t_ann:.2f}x")

    # SCVIVA graph path processes niche-composition fields per batch; allow generous overhead.
    assert t_graph / t_ann < 10.0, (
        f"GraphDataLoader is {t_graph / t_ann:.1f}x slower than AnnDataLoader"
    )


@pytest.mark.benchmark
def test_scviva_elbo_comparable_between_paths():
    """Both dataloader paths should reach similar final ELBO after training."""
    SCVIVALegacy = _scviva_legacy_cls()
    n_epochs = 10

    adata_legacy = _prepare_scviva_adata(SCVIVALegacy)
    model_ann = SCVIVALegacy(adata_legacy, prior_mixture=False)
    _train_legacy(model_ann, max_epochs=n_epochs)
    elbo_ann = model_ann.history_["elbo_train"].iloc[-1].values[0]

    adata_graph = _prepare_scviva_adata()
    model_graph = SCVIVA(adata_graph, prior_mixture=False)
    _train_graph(model_graph, max_epochs=n_epochs)
    elbo_graph = model_graph.history_["elbo_train"].iloc[-1].values[0]

    print(f"\nFinal ELBO - AnnDataLoader: {elbo_ann:.2f}  GraphDataLoader: {elbo_graph:.2f}")

    assert np.isfinite(elbo_graph), "GraphDataLoader ELBO is not finite"
    assert abs(elbo_graph - elbo_ann) / (abs(elbo_ann) + 1e-8) < 0.5, (
        f"ELBO diverged: ann={elbo_ann:.2f} graph={elbo_graph:.2f}"
    )


@pytest.mark.benchmark
def test_scviva_graph_elbo_decreases():
    """ELBO must decrease over training with GraphDataLoader."""
    adata = _prepare_scviva_adata()
    model = SCVIVA(adata, prior_mixture=False)
    # Use explicit train_size to avoid degenerate validation splits on small data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(
            max_epochs=5,
            enable_progress_bar=False,
            train_size=0.9,
            validation_size=0.1,
        )

    history = model.history_["elbo_train"]
    assert history.iloc[-1].values[0] < history.iloc[0].values[0], (
        "ELBO did not decrease with GraphDataLoader"
    )


@pytest.mark.benchmark
def test_scviva_latent_shape_graph_path():
    """get_latent_representation() returns (n_obs, n_latent) with GraphDataLoader."""
    adata = _prepare_scviva_adata()
    model = SCVIVA(adata, prior_mixture=False)
    _train_graph(model, max_epochs=3)
    latent = model.get_latent_representation()
    assert latent.shape == (adata.n_obs, model.module.n_latent)
