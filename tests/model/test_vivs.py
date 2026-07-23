"""Tests for VIVS model."""

import numpy as np
import pytest
import torch
from scvi.data import synthetic_iid

from scviva._constants import VIVS_REGISTRY_KEYS


def test_vivs_registry_keys():
    assert VIVS_REGISTRY_KEYS.Y_KEY == "Y"


def test_importance_score_net_mlp_shapes():
    from scviva.module._vivs import ImportanceScoreNet

    net = ImportanceScoreNet(n_input=20, n_responses=5, n_hidden=8, loss_type="mse", linear=False)
    x = torch.rand(16, 20)
    y = torch.randn(16, 5)
    out = net(x, y)
    assert out["h"].shape == (16, 5)
    assert out["all_loss"].shape == (16, 5)
    assert out["loss"].dim() == 0


def test_importance_score_net_linear_binary_shapes():
    from scviva.module._vivs import ImportanceScoreNet

    net = ImportanceScoreNet(n_input=20, n_responses=1, loss_type="binary", linear=True)
    x = torch.rand(16, 20)
    y = (torch.rand(16, 1) > 0.5).float()
    out = net(x, y)
    assert out["h"].shape == (16, 1)
    assert out["all_loss"].shape == (16, 1)
    assert torch.isfinite(out["loss"])


def test_vivs_module_phase_x_loss():
    from scvi.module.base import LossOutput

    from scviva.module._vivs import VIVSModule

    module = VIVSModule(n_input=20, n_responses=5, n_batch=2)
    assert module._phase == "x"
    tensors = {
        "X": torch.rand(8, 20) * 10,
        "batch": torch.randint(0, 2, (8, 1)),
        "labels": torch.zeros(8, 1, dtype=torch.long),
        "Y": torch.randn(8, 5),
    }
    inference_out = module.inference(**module._get_inference_input(tensors))
    generative_out = module.generative(**module._get_generative_input(tensors, inference_out))
    loss_out = module.loss(tensors, inference_out, generative_out)
    assert isinstance(loss_out, LossOutput)
    assert torch.isfinite(loss_out.loss)


def test_vivs_module_phase_xy_loss():
    from scviva.module._vivs import VIVSModule

    module = VIVSModule(n_input=20, n_responses=5, n_batch=2)
    module._phase = "xy"
    tensors = {
        "X": torch.rand(8, 20) * 10,
        "batch": torch.randint(0, 2, (8, 1)),
        "labels": torch.zeros(8, 1, dtype=torch.long),
        "Y": torch.randn(8, 5),
    }
    inference_out = module.inference(**module._get_inference_input(tensors))
    generative_out = module.generative(**module._get_generative_input(tensors, inference_out))
    loss_out = module.loss(tensors, inference_out, generative_out)
    assert torch.isfinite(loss_out.loss)
    # xy phase must not touch x_module's parameters
    loss_out.loss.backward()
    for p in module.x_module.parameters():
        assert p.grad is None
    for p in module.xy_module.parameters():
        assert p.grad is not None


@pytest.fixture
def vivs_adata():
    adata = synthetic_iid(n_genes=50)
    return adata


def test_vivs_setup_anndata_and_init(vivs_adata):
    from scviva.model._vivs import VIVS

    VIVS.setup_anndata(vivs_adata, y_obsm_key="protein_expression", batch_key="batch")
    model = VIVS(vivs_adata, n_hidden=8, n_latent=4)
    n_proteins = vivs_adata.obsm["protein_expression"].shape[1]
    assert model.module.xy_module.dense_out.out_features == n_proteins
    assert not model.module.x_module_is_pretrained


def test_vivs_train_fresh_vae(vivs_adata):
    from scviva.model._vivs import VIVS

    VIVS.setup_anndata(vivs_adata, y_obsm_key="protein_expression", batch_key="batch")
    model = VIVS(vivs_adata, n_hidden=8, n_latent=4)
    model.train(max_epochs=1)
    assert model.is_trained
    assert model.module._phase == "xy"
    assert not any(p.requires_grad for p in model.module.x_module.parameters())


def test_vivs_get_latent_representation(vivs_adata):
    from scviva.model._vivs import VIVS

    VIVS.setup_anndata(vivs_adata, y_obsm_key="protein_expression", batch_key="batch")
    model = VIVS(vivs_adata, n_hidden=8, n_latent=4)
    model.train(max_epochs=1)
    z = model.get_latent_representation()
    assert z.shape == (vivs_adata.n_obs, 4)


def test_vivs_pretrained_x_model(vivs_adata):
    from scvi.model import SCVI

    from scviva.model._vivs import VIVS

    SCVI.setup_anndata(vivs_adata, batch_key="batch")
    scvi_model = SCVI(vivs_adata, n_hidden=8, n_latent=4)
    scvi_model.train(max_epochs=1)

    VIVS.setup_anndata(vivs_adata, y_obsm_key="protein_expression", batch_key="batch")
    model = VIVS(vivs_adata, x_model=scvi_model)
    assert model.module.x_module_is_pretrained
    assert model.module.x_module is scvi_model.module

    model.train(max_epochs=1)
    assert model.is_trained
    # phase-1 training must have been skipped: x_module's optimizer never ran under VIVS
    assert model.module._phase == "xy"


def test_vivs_untrained_x_model_raises(vivs_adata):
    from scvi.model import SCVI

    from scviva.model._vivs import VIVS

    SCVI.setup_anndata(vivs_adata, batch_key="batch")
    scvi_model = SCVI(vivs_adata)  # not trained

    VIVS.setup_anndata(vivs_adata, y_obsm_key="protein_expression", batch_key="batch")
    with pytest.raises(ValueError, match="must already be trained"):
        VIVS(vivs_adata, x_model=scvi_model)


def test_vivs_predict_t(vivs_adata):
    from scviva.model._vivs import VIVS

    VIVS.setup_anndata(vivs_adata, y_obsm_key="protein_expression", batch_key="batch")
    model = VIVS(vivs_adata, n_hidden=8, n_latent=4)
    model.train(max_epochs=1)
    t = model.predict_t()
    assert t.shape == (vivs_adata.n_obs, vivs_adata.obsm["protein_expression"].shape[1])
    assert np.isfinite(t).all()
