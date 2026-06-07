"""Unit tests for SpatialPredictiveMixin (mixin-level, no full model training)."""

from __future__ import annotations

import numpy as np
import pytest
from scvi.data import synthetic_iid

from scviva.model._resolvi import ResolVI

N_EPOCHS = 1
N_NEIGHS = 10


def _make_resolvi_adata():
    adata = synthetic_iid()  # default: n_genes=100, n_batches=2 — avoids NaN in Pyro Gamma
    n = adata.n_obs
    rng = np.random.default_rng(0)
    adata.obsm["spatial"] = rng.random((n, 2))
    idx = np.zeros((n, N_NEIGHS), dtype=np.int64)
    dst = np.ones((n, N_NEIGHS), dtype=np.float32)
    for i in range(n):
        idx[i] = rng.choice([j for j in range(n) if j != i], size=N_NEIGHS, replace=False)
        dst[i] = rng.uniform(0.1, 2.0, size=N_NEIGHS)
    adata.obsm["index_neighbor"] = idx
    adata.obsm["distance_neighbor"] = dst
    return adata


@pytest.fixture(scope="module")
def resolvi_model():
    """Minimal trained ResolVI for testing SpatialPredictiveMixin methods."""
    adata = _make_resolvi_adata()
    ResolVI.setup_anndata(adata, labels_key="labels")
    model = ResolVI(adata, semisupervised=True)  # semisupervised needed for probs_prediction site
    model.train(max_epochs=N_EPOCHS, accelerator="cpu")
    return model, adata


@pytest.mark.optional
def test_get_neighbor_abundance_shape(resolvi_model):
    model, adata = resolvi_model
    result = model.get_neighbor_abundance(return_numpy=True)
    assert result.ndim == 2
    assert result.shape[0] == adata.n_obs
    assert not np.isnan(result).any()


def test_get_normalized_expression_importance_pytorch():
    """SpatialPredictiveMixin default (PyTorch path) shape matches get_normalized_expression."""
    from scviva.model._scviva import SCVIVA

    adata = synthetic_iid(n_genes=20, n_batches=2, n_labels=3, sparse_format=None)
    n = adata.n_obs
    adata.obsm["spatial"] = np.random.rand(n, 2)
    adata.obsm["X_scVI"] = np.random.normal(size=(n, 20))
    raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    adata.layers["counts"] = np.abs(raw).astype(int)

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
    SCVIVA.preprocessing_anndata(adata, k_nn=5, **setup_kwargs)
    SCVIVA.setup_anndata(adata, layer="counts", batch_key="batch", **setup_kwargs)
    model = SCVIVA(adata, prior_mixture=False)
    model.train(max_epochs=1, accelerator="cpu")

    result = model.get_normalized_expression_importance(n_samples=3, return_numpy=True)
    expected = model.get_normalized_expression(n_samples=3, return_numpy=True)
    assert result.shape == expected.shape
    assert not np.isnan(result).any()
