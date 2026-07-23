"""Tests for VIVS model."""

import torch

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
