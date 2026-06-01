"""Tests for ImagingBaseModel."""

from __future__ import annotations

import numpy as np
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


def test_from_spatialdata_with_region_preserves_registration(tmp_path):
    from unittest.mock import MagicMock

    import numpy as np
    from anndata import AnnData

    adata = AnnData(X=np.zeros((5, 1)))
    adata.obsm["spatial"] = np.random.rand(5, 2).astype(np.float32)
    adata.obs["img_path"] = [str(tmp_path / f"c{i}.png") for i in range(5)]
    adata.obs["region"] = "r1"
    adata.uns["spatialdata_attrs"] = {"region_key": "region"}

    sdata = MagicMock()
    sdata.__getitem__ = MagicMock(return_value=adata)

    result = _MinimalImagingModel.from_spatialdata(
        sdata, table_key="table", region="r1", img_path_col="img_path"
    )
    assert isinstance(result, AnnData)
    assert "scviva_imaging" in result.uns
    assert result.uns["scviva_imaging"]["img_path_col"] == "img_path"


def test_get_latent_representation_shape(imaging_adata, minimal_checkpoint):
    _MinimalImagingModel.setup_anndata(
        imaging_adata, img_path_col="img_path", spatial_key="spatial"
    )
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    result = model.get_latent_representation(imaging_adata)
    assert result.shape == (len(imaging_adata), 8)


def test_get_latent_representation_writes_obsm(imaging_adata, minimal_checkpoint):
    _MinimalImagingModel.setup_anndata(
        imaging_adata, img_path_col="img_path", spatial_key="spatial"
    )
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    model.get_latent_representation(imaging_adata)
    assert "X_test_imaging" in imaging_adata.obsm
    assert imaging_adata.obsm["X_test_imaging"].shape == (len(imaging_adata), 8)


def test_get_latent_representation_no_nan(imaging_adata, minimal_checkpoint):
    _MinimalImagingModel.setup_anndata(
        imaging_adata, img_path_col="img_path", spatial_key="spatial"
    )
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    result = model.get_latent_representation(imaging_adata)
    assert not np.any(np.isnan(result))


def test_get_latent_representation_obsm_key_override(imaging_adata, minimal_checkpoint):
    _MinimalImagingModel.setup_anndata(
        imaging_adata, img_path_col="img_path", spatial_key="spatial"
    )
    model = _MinimalImagingModel.from_pretrained(minimal_checkpoint)
    model.get_latent_representation(imaging_adata, obsm_key="X_custom")
    assert "X_custom" in imaging_adata.obsm


def test_imaging_base_importable_from_imaging():
    from scviva.imaging import ImagingBaseModel

    assert ImagingBaseModel is not None


def test_sparl_importable_from_imaging():
    pytest.importorskip("sparl")
    from scviva.imaging import SPARL

    assert SPARL is not None


def test_sparl_importable_from_scviva_top_level():
    pytest.importorskip("sparl")
    import scviva

    SPARLClass = scviva.SPARL
    assert SPARLClass is not None
