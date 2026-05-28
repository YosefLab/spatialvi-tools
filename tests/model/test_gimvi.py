"""Tests for GIMVI model."""

from __future__ import annotations

import numpy as np
import pytest
from scvi.data import synthetic_iid

from scviva.model._gimvi import GIMVI


@pytest.fixture(scope="module")
def gimvi_data():
    """Shared seq + spatial synthetic datasets."""
    np.random.seed(0)
    adata_seq = synthetic_iid(n_genes=50, n_batches=2, sparse_format=None)
    adata_seq.layers["counts"] = adata_seq.X.copy()

    adata_spatial = synthetic_iid(n_genes=20, n_batches=1, sparse_format=None)
    adata_spatial.layers["counts"] = adata_spatial.X.copy()
    # Make spatial genes a strict subset of seq genes
    adata_spatial.var_names = adata_seq.var_names[:20]

    return adata_seq, adata_spatial


def _setup_and_build(adata_seq, adata_spatial, **model_kwargs):
    GIMVI.setup_anndata(adata_seq, layer="counts", batch_key="batch")
    GIMVI.setup_anndata(adata_spatial, layer="counts")
    return GIMVI(adata_seq, adata_spatial, **model_kwargs)


def test_gimvi_train(gimvi_data):
    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)
    assert model.is_trained


def test_gimvi_get_latent_representation(gimvi_data):
    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)

    latents = model.get_latent_representation()
    assert isinstance(latents, list)
    assert len(latents) == 2
    assert latents[0].shape[0] == adata_seq.n_obs
    assert latents[1].shape[0] == adata_spatial.n_obs
    assert latents[0].shape[1] == latents[1].shape[1]  # same latent dim


def test_gimvi_get_latent_backend_cpu(gimvi_data):
    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1, accelerator="cpu")

    latents = model.get_latent_representation(backend="cpu")
    assert all(isinstance(z, np.ndarray) for z in latents)


def test_gimvi_get_latent_rapids_stub(gimvi_data, monkeypatch):
    """Verify rapids path converts to cupy arrays (stubbed)."""
    import sys
    import types

    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)

    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(asarray=np.asarray))
    latents = model.get_latent_representation(backend="rapids")
    assert len(latents) == 2


def test_gimvi_get_imputed_values(gimvi_data):
    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)

    imputed = model.get_imputed_values()
    assert isinstance(imputed, list)
    assert len(imputed) == 2
    assert imputed[0].shape[0] == adata_seq.n_obs
    assert imputed[1].shape[0] == adata_spatial.n_obs


def test_gimvi_same_adata_raises(gimvi_data):
    adata_seq, _ = gimvi_data
    GIMVI.setup_anndata(adata_seq, layer="counts", batch_key="batch")
    with pytest.raises(ValueError, match="cannot point to the same object"):
        GIMVI(adata_seq, adata_seq)


def test_gimvi_spatial_genes_not_subset_raises(gimvi_data):
    adata_seq, _ = gimvi_data
    np.random.seed(1)
    bad_spatial = synthetic_iid(n_genes=20, n_batches=1, sparse_format=None)
    bad_spatial.layers["counts"] = bad_spatial.X.copy()
    # Force var_names that are NOT in adata_seq
    bad_spatial.var_names = [f"nonexistent_gene_{i}" for i in range(20)]

    GIMVI.setup_anndata(adata_seq, layer="counts", batch_key="batch")
    GIMVI.setup_anndata(bad_spatial, layer="counts")
    with pytest.raises(ValueError, match="subset"):
        GIMVI(adata_seq, bad_spatial)


def test_gimvi_setup_spatialdata_inherits():
    """setup_spatialdata classmethod must be present (inherited from SpatialBaseModel)."""
    assert hasattr(GIMVI, "setup_spatialdata")
    assert hasattr(GIMVI, "from_spatialdata")


def test_gimvi_gene_label_dispersion():
    """dispersion='gene-label' must not crash (y.type was a method object bug)."""
    np.random.seed(0)
    adata_seq = synthetic_iid(n_genes=50, n_batches=2, n_labels=3, sparse_format=None)
    adata_seq.layers["counts"] = adata_seq.X.copy()
    adata_spatial = synthetic_iid(n_genes=20, n_batches=1, sparse_format=None)
    adata_spatial.layers["counts"] = adata_spatial.X.copy()
    adata_spatial.var_names = adata_seq.var_names[:20]

    GIMVI.setup_anndata(adata_seq, layer="counts", batch_key="batch", labels_key="labels")
    GIMVI.setup_anndata(adata_spatial, layer="counts")
    n_labels = adata_seq.obs["labels"].nunique()
    model = GIMVI(adata_seq, adata_spatial, dispersion="gene-label", n_labels=n_labels)
    model.train(max_epochs=1)
    assert model.is_trained


def test_gimvi_lazy_import():
    """GIMVI must be accessible via scviva top-level lazy import."""
    import scviva

    assert hasattr(scviva, "GIMVI") or "GIMVI" in scviva.__all__


def test_gimvi_save_load(gimvi_data, tmp_path):
    """Save/load round-trip preserves latent representation."""
    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)

    z1 = model.get_latent_representation([adata_seq])
    save_path = str(tmp_path / "gimvi_save")
    model.save(save_path, overwrite=True, save_anndata=True, prefix="GIMVI_")

    model2 = GIMVI.load(save_path, prefix="GIMVI_")
    z2 = model2.get_latent_representation([adata_seq])
    np.testing.assert_array_almost_equal(z1, z2, decimal=3)
    assert model2.is_trained is True

    # Load providing explicit adatas
    model3 = GIMVI.load(
        save_path,
        adata_seq=adata_seq,
        adata_spatial=adata_spatial,
        prefix="GIMVI_",
    )
    z3 = model3.get_latent_representation([adata_seq])
    np.testing.assert_array_almost_equal(z1, z3, decimal=3)


def test_gimvi_model_library_size(gimvi_data):
    """model_library_size=[True, True] adds library_log_means for both modalities."""
    adata_seq, adata_spatial = gimvi_data
    GIMVI.setup_anndata(adata_seq, layer="counts", batch_key="batch")
    GIMVI.setup_anndata(adata_spatial, layer="counts")
    model = GIMVI(
        adata_seq,
        adata_spatial,
        model_library_size=[True, True],
        n_latent=10,
    )
    assert hasattr(model.module, "library_log_means_0")
    assert hasattr(model.module, "library_log_means_1")
    model.train(max_epochs=1)
    latents = model.get_latent_representation()
    assert len(latents) == 2
    model.get_imputed_values()


def test_gimvi_reinit(gimvi_data):
    """GIMVI can be re-initialized and retrained without error."""
    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)
    # Reinitialize and retrain — should not raise
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)
    assert model.is_trained


def test_gimvi_get_imputed_values_unnormalized(gimvi_data):
    """get_imputed_values(normalized=False) returns raw rates without error."""
    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)

    imputed = model.get_imputed_values(normalized=False)
    assert isinstance(imputed, list)
    assert len(imputed) == 2
    assert imputed[0].shape[0] == adata_seq.n_obs
    assert imputed[1].shape[0] == adata_spatial.n_obs
    # Unnormalized values should be non-negative
    assert (imputed[0] >= 0).all()
    assert (imputed[1] >= 0).all()


def test_gimvi_load_wrong_genes_raises(gimvi_data, tmp_path):
    """GIMVI.load with mismatched var_names must raise ValueError."""
    adata_seq, adata_spatial = gimvi_data
    model = _setup_and_build(adata_seq, adata_spatial)
    model.train(max_epochs=1)
    save_path = str(tmp_path / "gimvi_bad_load")
    model.save(save_path, overwrite=True, save_anndata=True)

    tmp_seq = synthetic_iid(n_genes=200, sparse_format=None)
    tmp_spatial = synthetic_iid(n_genes=200, sparse_format=None)
    with pytest.raises(ValueError):
        GIMVI.load(
            save_path,
            adata_seq=tmp_seq,
            adata_spatial=tmp_spatial,
        )
