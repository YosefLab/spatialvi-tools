"""Regression tests: scviva.external.diagvi.DIAGVI vs scvi.external.diagvi.DIAGVI.

Each test runs identical operations on the same data with the same random seed
on both implementations and asserts outputs are structurally identical (shapes
match; exact numeric equality is not required across the two independent
training runs, matching the same tolerance used by test_gimvi_upstream.py).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scvi.data import synthetic_iid
from scvi.external.diagvi import DIAGVI as ScviDIAGVI

from scviva.external.diagvi import DIAGVI as SpatialDIAGVI

SEED = 42
N_EPOCHS = 1
N_LATENT = 5


def _make_data(seed=SEED):
    np.random.seed(seed)
    adata_a = synthetic_iid(n_genes=50, n_batches=2, sparse_format=None)
    adata_a.layers["counts"] = adata_a.X.copy()

    adata_b = synthetic_iid(n_genes=50, n_batches=1, sparse_format=None)
    adata_b.layers["counts"] = adata_b.X.copy()
    adata_b.var_names = adata_a.var_names

    return adata_a, adata_b


def _train_scvi_diagvi(adata_a, adata_b, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ScviDIAGVI.setup_anndata(adata_a, layer="counts", batch_key="batch", likelihood="nb")
    ScviDIAGVI.setup_anndata(adata_b, layer="counts", batch_key="batch", likelihood="nb")
    model = ScviDIAGVI({"seq": adata_a, "spatial": adata_b}, n_latent=N_LATENT)
    model.train(max_epochs=N_EPOCHS, batch_size=16)
    return model


def _train_scviva_diagvi(adata_a, adata_b, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    SpatialDIAGVI.setup_anndata(adata_a, layer="counts", batch_key="batch", likelihood="nb")
    SpatialDIAGVI.setup_anndata(adata_b, layer="counts", batch_key="batch", likelihood="nb")
    model = SpatialDIAGVI({"seq": adata_a, "spatial": adata_b}, n_latent=N_LATENT)
    model.train(max_epochs=N_EPOCHS, batch_size=16)
    return model


@pytest.fixture(scope="module")
def both_models():
    adata_a, adata_b = _make_data()
    scvi_model = _train_scvi_diagvi(adata_a.copy(), adata_b.copy())
    scviva_model = _train_scviva_diagvi(adata_a.copy(), adata_b.copy())
    return scvi_model, scviva_model


def test_diagvi_trained(both_models):
    scvi_model, scviva_model = both_models
    assert scvi_model.is_trained_
    assert scviva_model.is_trained_


def test_diagvi_latent_shapes_match(both_models):
    scvi_model, scviva_model = both_models

    scvi_latents = scvi_model.get_latent_representation()
    scviva_latents = scviva_model.get_latent_representation()

    assert set(scvi_latents.keys()) == set(scviva_latents.keys()) == {"seq", "spatial"}
    for key in scvi_latents:
        assert scvi_latents[key].shape == scviva_latents[key].shape, (
            f"Shape mismatch for {key!r}: scvi {scvi_latents[key].shape} "
            f"vs scviva {scviva_latents[key].shape}"
        )


def test_diagvi_imputed_shapes_match(both_models):
    scvi_model, scviva_model = both_models

    scvi_imp = scvi_model.get_imputed_values(query_name="spatial")
    scviva_imp = scviva_model.get_imputed_values(query_name="spatial")

    assert scvi_imp.shape == scviva_imp.shape


def test_diagvi_module_same_class():
    """DIAGVAE module class must be structurally identical to scvi's DIAGVAE."""
    from scvi.external.diagvi._module import DIAGVAE as ScviDIAGVAE

    from scviva.external.diagvi._module import DIAGVAE as SpatialDIAGVAE

    scvi_methods = {m for m in dir(ScviDIAGVAE) if not m.startswith("__")}
    scviva_methods = {m for m in dir(SpatialDIAGVAE) if not m.startswith("__")}
    missing = scvi_methods - scviva_methods
    assert not missing, f"Methods in scvi DIAGVAE missing from scviva DIAGVAE: {missing}"
