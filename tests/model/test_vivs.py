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
