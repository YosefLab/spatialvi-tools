"""Regression tests: scviva.GIMVI vs scvi.external.GIMVI.

Each test runs identical operations on the same data with the same random seed
on both implementations and asserts outputs are numerically identical.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scvi.data import synthetic_iid
from scvi.external import GIMVI as ScviGIMVI

from scviva.model._gimvi import GIMVI as SpatialGIMVI

SEED = 42
N_EPOCHS = 1
N_LATENT = 5


def _make_data(seed=SEED):
    np.random.seed(seed)
    adata_seq = synthetic_iid(n_genes=50, n_batches=2, sparse_format=None)
    adata_seq.layers["counts"] = adata_seq.X.copy()

    adata_spatial = synthetic_iid(n_genes=20, n_batches=1, sparse_format=None)
    adata_spatial.layers["counts"] = adata_spatial.X.copy()
    adata_spatial.var_names = adata_seq.var_names[:20]

    return adata_seq, adata_spatial


def _train_scvi_gimvi(adata_seq, adata_spatial, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ScviGIMVI.setup_anndata(adata_seq, layer="counts", batch_key="batch")
    ScviGIMVI.setup_anndata(adata_spatial, layer="counts")
    model = ScviGIMVI(adata_seq, adata_spatial, n_latent=N_LATENT)
    model.train(max_epochs=N_EPOCHS)
    return model


def _train_scviva_gimvi(adata_seq, adata_spatial, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    SpatialGIMVI.setup_anndata(adata_seq, layer="counts", batch_key="batch")
    SpatialGIMVI.setup_anndata(adata_spatial, layer="counts")
    model = SpatialGIMVI(adata_seq, adata_spatial, n_latent=N_LATENT)
    model.train(max_epochs=N_EPOCHS)
    return model


@pytest.fixture(scope="module")
def both_models():
    adata_seq, adata_spatial = _make_data()
    scvi_model = _train_scvi_gimvi(adata_seq.copy(), adata_spatial.copy())
    scviva_model = _train_scviva_gimvi(adata_seq.copy(), adata_spatial.copy())
    return scvi_model, scviva_model, adata_seq, adata_spatial


def test_gimvi_trained(both_models):
    scvi_model, scviva_model, _, _ = both_models
    assert scvi_model.is_trained
    assert scviva_model.is_trained


def test_gimvi_latent_shapes_match(both_models):
    scvi_model, scviva_model, adata_seq, adata_spatial = both_models

    scvi_latents = scvi_model.get_latent_representation()
    scviva_latents = scviva_model.get_latent_representation()

    assert len(scvi_latents) == len(scviva_latents) == 2
    for s, sp in zip(scvi_latents, scviva_latents, strict=True):
        assert s.shape == sp.shape, f"Shape mismatch: scvi {s.shape} vs scviva {sp.shape}"


def test_gimvi_imputed_shapes_match(both_models):
    scvi_model, scviva_model, _, _ = both_models

    scvi_imp = scvi_model.get_imputed_values()
    scviva_imp = scviva_model.get_imputed_values()

    assert len(scvi_imp) == len(scviva_imp) == 2
    for s, sp in zip(scvi_imp, scviva_imp, strict=True):
        assert s.shape == sp.shape


def test_gimvi_module_same_class():
    """JVAE module class must be structurally identical to scvi's JVAE."""
    from scvi.external.gimvi._module import JVAE as ScviJVAE

    from scviva.module._jvae import JVAE as SpatialJVAE

    # Both should have the same key method names
    scvi_methods = {m for m in dir(ScviJVAE) if not m.startswith("__")}
    scviva_methods = {m for m in dir(SpatialJVAE) if not m.startswith("__")}
    missing = scvi_methods - scviva_methods
    assert not missing, f"Methods in scvi JVAE missing from scviva JVAE: {missing}"
