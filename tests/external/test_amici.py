from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData


def _make_amici_adata(n_obs: int = 24, n_genes: int = 12, n_labels: int = 3) -> AnnData:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.0, scale=1.0, size=(n_obs, n_genes)).astype(np.float32)
    adata = AnnData(X=x)
    adata.layers["expr"] = x.copy()
    adata.obs["cell_type"] = [f"ct_{i % n_labels}" for i in range(n_obs)]
    adata.obsm["spatial"] = rng.normal(size=(n_obs, 2)).astype(np.float32)
    return adata


def test_amici_setup_anndata_registers_neighbors():
    from scviva.external.amici import AMICI
    from scviva.external.amici._constants import AMICI_REGISTRY_KEYS

    adata = _make_amici_adata()
    AMICI.setup_anndata(
        adata,
        layer="expr",
        labels_key="cell_type",
        spatial_key="spatial",
        n_neighbors=4,
    )

    assert AMICI_REGISTRY_KEYS.NN_IDX_KEY in adata.obsm
    assert AMICI_REGISTRY_KEYS.NN_DIST_KEY in adata.obsm
    assert adata.obsm[AMICI_REGISTRY_KEYS.NN_IDX_KEY].shape == (adata.n_obs, 4)
    assert adata.obsm[AMICI_REGISTRY_KEYS.NN_DIST_KEY].shape == (adata.n_obs, 4)


def test_amici_train_and_predict_shape():
    from scviva.external.amici import AMICI

    adata = _make_amici_adata()
    AMICI.setup_anndata(
        adata,
        layer="expr",
        labels_key="cell_type",
        spatial_key="spatial",
        n_neighbors=3,
    )
    model = AMICI(
        adata,
        n_label_embed=4,
        n_nn_embed=8,
        n_hidden=16,
    )

    model.train(max_epochs=1, batch_size=8, lr=1e-2)
    predictions = model.get_predictions(batch_size=8)

    assert model.is_trained
    assert predictions.shape == (adata.n_obs, adata.n_vars)
    assert not np.any(np.isnan(predictions))


def test_amici_external_import():
    from scviva.external import AMICI
    from scviva.external.amici import AMICI as SubmoduleAMICI
    from scviva.external.amici import AMICIModule

    assert AMICI is SubmoduleAMICI
    assert AMICIModule is not None


def test_amici_setup_anndata_requires_labels_for_label_exclusion():
    from scviva.external.amici import AMICI

    adata = _make_amici_adata()
    with pytest.raises(ValueError, match="labels_key"):
        AMICI.setup_anndata(
            adata,
            layer="expr",
            labels_key=None,
            spatial_key="spatial",
            exclude_self_labels=True,
        )
