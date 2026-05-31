"""Tests for ImagingBaseModel."""

from __future__ import annotations

import pytest

from tests.imaging.conftest import _MinimalImagingModel


def test_from_pretrained_local_sets_module(minimal_checkpoint):
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    assert model.module is not None


def test_from_pretrained_local_module_in_eval(minimal_checkpoint):
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    assert not model.module.training


def test_from_pretrained_bad_path_raises():
    with pytest.raises(FileNotFoundError, match="Could not resolve model path"):
        _MinimalImagingModel.from_pretrained("/does/not/exist.pt")


def test_train_raises(minimal_checkpoint):
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    with pytest.raises(NotImplementedError):
        model.train()


def test_from_pretrained_hf_calls_hub_download(tmp_path, minimal_checkpoint, monkeypatch):
    import huggingface_hub

    call_log = {}

    def mock_download(repo_id, filename, **kwargs):
        call_log["repo_id"] = repo_id
        call_log["filename"] = filename
        return minimal_checkpoint

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", mock_download)

    model = _MinimalImagingModel.from_pretrained("testuser/testmodel")
    assert model.module is not None
    assert call_log["repo_id"] == "testuser/testmodel"
    assert call_log["filename"] == "model.pt"
