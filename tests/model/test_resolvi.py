import numpy as np
import pytest
from scvi.data import synthetic_iid

from scviva.model._resolvi import ResolVI


def _prepare_adata_with_neighbors(n_neighs=10):
    """Create a synthetic AnnData with spatial coords and neighbor arrays."""
    adata = synthetic_iid()
    n = adata.n_obs
    rng = np.random.default_rng(42)
    adata.obsm["spatial"] = rng.random((n, 2))
    adata.obs["cell_area"] = rng.gamma(2.0, 1.0, size=n)

    # Pre-compute neighbor arrays (index + distance) manually so tests
    # do not require squidpy or scanpy.
    index_neighbor = np.zeros((n, n_neighs), dtype=np.int64)
    distance_neighbor = np.ones((n, n_neighs), dtype=np.float32)
    for i in range(n):
        neighbors = rng.choice([j for j in range(n) if j != i], size=n_neighs, replace=False)
        index_neighbor[i] = neighbors
        distance_neighbor[i] = rng.uniform(0.1, 2.0, size=n_neighs)

    adata.obsm["index_neighbor"] = index_neighbor
    adata.obsm["distance_neighbor"] = distance_neighbor
    return adata


@pytest.fixture(scope="module")
def resolvi_adata():
    return _prepare_adata_with_neighbors()


def test_resolvi_setup_anndata(resolvi_adata):
    """setup_anndata registers the AnnDataManager without errors."""
    ResolVI.setup_anndata(resolvi_adata)
    assert resolvi_adata is not None


def test_resolvi_train(resolvi_adata):
    """Model initialises and trains for 2 epochs without error."""
    ResolVI.setup_anndata(resolvi_adata)
    model = ResolVI(resolvi_adata)
    model.train(max_epochs=1)
    assert model.is_trained


def test_resolvi_get_latent_cpu(resolvi_adata):
    """get_latent_representation returns an array with the correct shape."""
    ResolVI.setup_anndata(resolvi_adata)
    model = ResolVI(resolvi_adata)
    model.train(max_epochs=1)
    latent = model.get_latent_representation(backend="cpu")
    assert latent.shape[0] == resolvi_adata.n_obs


def test_resolvi_compute_neighbors(resolvi_adata):
    """compute_neighbors (squidpy backend) populates index/distance obsm keys."""
    pytest.importorskip("squidpy")
    ResolVI.setup_anndata(resolvi_adata)
    model = ResolVI(resolvi_adata)
    model.compute_neighbors(resolvi_adata, spatial_key="spatial", n_neighs=5)
    assert "index_neighbor" in resolvi_adata.obsm
    assert "distance_neighbor" in resolvi_adata.obsm


@pytest.mark.optional
def test_resolvi_neighbor_abundance(resolvi_adata):
    """get_neighbor_abundance returns shape (n_obs, n_cell_types) with no NaNs."""
    ResolVI.setup_anndata(resolvi_adata, labels_key="labels")
    model = ResolVI(resolvi_adata, semisupervised=True)  # probs_prediction requires semisupervised
    model.train(max_epochs=1, accelerator="cpu")
    result = model.get_neighbor_abundance(return_numpy=True)
    assert result.ndim == 2
    assert result.shape[0] == resolvi_adata.n_obs
    assert not np.isnan(result).any()
