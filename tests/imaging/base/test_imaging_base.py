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


def test_setup_anndata_stores_img_path_col(imaging_adata):
    _MinimalImagingModel.setup_anndata(
        imaging_adata, img_path_col="img_path", spatial_key="spatial"
    )
    assert "scviva_imaging" in imaging_adata.uns
    assert imaging_adata.uns["scviva_imaging"]["img_path_col"] == "img_path"


def test_from_spatialdata_returns_adata(tmp_path):
    from unittest.mock import MagicMock

    import numpy as np
    from anndata import AnnData

    adata = AnnData(X=np.zeros((5, 1)))
    adata.obsm["spatial"] = np.random.rand(5, 2).astype(np.float32)
    adata.obs["img_path"] = [str(tmp_path / f"c{i}.png") for i in range(5)]

    sdata = MagicMock()
    sdata.__getitem__ = MagicMock(return_value=adata)

    result = _MinimalImagingModel.from_spatialdata(
        sdata, table_key="table", img_path_col="img_path"
    )
    assert isinstance(result, AnnData)
    assert "scviva_imaging" in result.uns
