import torch

from scviva.utils import resolve_device, stats_dtype


def test_resolve_device_auto_falls_back_when_cuda_unusable(monkeypatch):
    """`cuda.is_available()` can be True while torch has no working CUDA build;
    `resolve_device` must fall back gracefully rather than raise."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_resolve_device_auto_prefers_mps_over_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_resolve_device_explicit_string():
    assert resolve_device("cpu") == torch.device("cpu")


def test_stats_dtype_mps_is_float32():
    assert stats_dtype(torch.device("mps")) == torch.float32


def test_stats_dtype_cpu_is_float64():
    assert stats_dtype(torch.device("cpu")) == torch.float64


def test_stats_dtype_accepts_device_string():
    assert stats_dtype("mps") == torch.float32
    assert stats_dtype("cpu") == torch.float64
