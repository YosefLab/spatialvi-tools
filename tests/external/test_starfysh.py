from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def _make_starfysh_adata(n_obs: int = 24, n_genes: int = 10) -> AnnData:
    rng = np.random.default_rng(1)
    counts = rng.poisson(lam=3.0, size=(n_obs, n_genes)).astype(np.float32)
    adata = AnnData(X=counts)
    adata.layers["counts"] = counts.copy()
    adata.obsm["spatial"] = rng.normal(size=(n_obs, 2)).astype(np.float32)
    return adata


def _make_signature_scores(n_obs: int = 24, n_cell_types: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    scores = rng.uniform(0.1, 1.0, size=(n_obs, n_cell_types)).astype(np.float32)
    scores = scores / scores.sum(axis=1, keepdims=True)
    return pd.DataFrame(scores, columns=[f"ct_{i}" for i in range(n_cell_types)])


def test_starfysh_setup_anndata_registers_spatial_fields():
    from scviva.external.starfysh import Starfysh

    adata = _make_starfysh_adata()
    Starfysh.setup_anndata(adata, layer="counts", spatial_key="spatial")

    manager = Starfysh._get_most_recent_anndata_manager(adata)
    assert manager is not None


def test_starfysh_train_and_get_proportions():
    from scviva.external.starfysh import Starfysh

    adata = _make_starfysh_adata()
    signatures = _make_signature_scores(adata.n_obs)
    Starfysh.setup_anndata(adata, layer="counts", spatial_key="spatial")
    model = Starfysh(
        adata,
        signature_scores=signatures,
        n_latent=4,
        n_hidden=16,
    )

    model.train(max_epochs=1, batch_size=8, lr=1e-2)
    proportions = model.get_proportions()

    assert model.is_trained
    assert isinstance(proportions, pd.DataFrame)
    assert proportions.shape == (adata.n_obs, signatures.shape[1])
    np.testing.assert_allclose(proportions.sum(axis=1).values, 1.0, atol=1e-5)
    assert not np.any(np.isnan(proportions.values))


def test_starfysh_get_proportions_df_via_mixin():
    from scviva.external.starfysh import Starfysh

    adata = _make_starfysh_adata()
    signatures = _make_signature_scores(adata.n_obs)
    Starfysh.setup_anndata(adata, layer="counts", spatial_key="spatial")
    model = Starfysh(adata, signature_scores=signatures, n_latent=4, n_hidden=16)
    model.train(max_epochs=1, batch_size=8, lr=1e-2)

    proportions = model.get_proportions_df()

    assert proportions.shape == (adata.n_obs, signatures.shape[1])
    assert list(proportions.columns) == list(signatures.columns)


def test_starfysh_external_import():
    from scviva.external import Starfysh
    from scviva.external.starfysh import Starfysh as SubmoduleStarfysh
    from scviva.external.starfysh import StarfyshModule

    assert Starfysh is SubmoduleStarfysh
    assert StarfyshModule is not None
