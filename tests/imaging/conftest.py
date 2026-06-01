"""Shared fixtures for imaging tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from anndata import AnnData
from PIL import Image

from scviva.imaging.base._imaging_base import ImagingBaseModel


class _MinimalImagingModel(ImagingBaseModel):
    """Concrete subclass using a tiny Linear backbone — no sparl dependency."""

    _default_obsm_key = "X_test_imaging"

    @classmethod
    def _build_module(cls, checkpoint_data: dict) -> nn.Module:
        # Expects checkpoint with state_dict for nn.Sequential(Flatten, Linear(64, 8))
        mod = nn.Sequential(nn.Flatten(), nn.Linear(64, 8))
        mod.load_state_dict(checkpoint_data["state_dict"])
        return mod

    @classmethod
    def setup_anndata(
        cls, adata: AnnData, img_path_col: str, spatial_key: str = "spatial", **kwargs
    ) -> None:
        from scvi.data import AnnDataManager

        from scviva.data._fields import SpatialCoordsField

        fields = [SpatialCoordsField(obsm_key=spatial_key)]
        manager = AnnDataManager(fields=fields)
        manager.register_fields(adata, **kwargs)
        cls.register_manager(manager)
        adata.uns["scviva_imaging"] = {"img_path_col": img_path_col}


def make_imaging_adata(
    n: int = 20,
    n_channels: int = 1,
    img_size: int = 8,
    tmp_path=None,
) -> AnnData:
    """AnnData with spatial coords and obs['img_path'] pointing to real PNG files."""
    adata = AnnData(X=np.zeros((n, 1)))
    adata.obsm["spatial"] = np.random.rand(n, 2).astype(np.float32)
    if tmp_path is not None:
        paths = []
        for i in range(n):
            arr = np.random.randint(0, 255, (img_size, img_size), dtype=np.uint8)
            p = tmp_path / f"cell_{i}.png"
            Image.fromarray(arr).save(p)
            paths.append(str(p))
        adata.obs["img_path"] = paths
    return adata


@pytest.fixture(scope="module")
def minimal_checkpoint(tmp_path_factory):
    """Tiny checkpoint for _MinimalImagingModel: Sequential(Flatten, Linear(64, 8))."""
    tmp = tmp_path_factory.mktemp("ckpt")
    mod = nn.Sequential(nn.Flatten(), nn.Linear(64, 8))
    ckpt = {"model_type": "minimal", "state_dict": mod.state_dict()}
    path = tmp / "minimal.pt"
    torch.save(ckpt, path)
    return str(path)


@pytest.fixture(scope="module")
def imaging_adata(tmp_path_factory):
    """AnnData with 20 cells, 1-channel 8x8 PNG images."""
    tmp = tmp_path_factory.mktemp("imgs")
    return make_imaging_adata(n=20, n_channels=1, img_size=8, tmp_path=tmp)


@pytest.fixture(scope="module")
def sparl_tiny_vit():
    """Tiny DinoVisionTransformer: 3 channels, 16x16 input, patch 8, embed_dim 32."""
    sparl_pkg = pytest.importorskip("sparl")  # noqa: F841
    from sparl.models.backbones.vision_transformer import DinoVisionTransformer

    vit = DinoVisionTransformer(
        channel_names=[0, 1, 2],
        img_size=16,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=2,
        ffn_ratio=4.0,
        qkv_bias=True,
        layerscale_init_values=None,
        norm_layer="layernorm",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        untie_cls_and_patch_norms=False,
        untie_global_and_local_cls_norm=False,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        pos_embed_type="learnable",
    )
    vit.eval()
    return vit


@pytest.fixture(scope="module")
def sparl_checkpoint(tmp_path_factory, sparl_tiny_vit):
    """SPARL checkpoint for the tiny ViT (3ch, 16x16, embed_dim=32)."""
    tmp = tmp_path_factory.mktemp("sparl_ckpt")
    ckpt = {
        "epoch": 1,
        "step": 10,
        "model": {"teacher": {"backbone": sparl_tiny_vit.state_dict()}},
        "model_config": {
            "arch": "vit",
            "channel_names": [0, 1, 2],
            "img_size": 16,
            "patch_size": 8,
            "embed_dim": 32,
            "depth": 2,
            "num_heads": 2,
            "ffn_ratio": 4.0,
            "qkv_bias": True,
            "layerscale_init_values": None,
            "norm_layer": "layernorm",
            "ffn_layer": "mlp",
            "ffn_bias": True,
            "proj_bias": True,
            "untie_cls_and_patch_norms": False,
            "untie_global_and_local_cls_norm": False,
            "interpolate_antialias": False,
            "interpolate_offset": 0.1,
            "pos_embed_type": "learnable",
        },
    }
    path = tmp / "sparl_tiny.pt"
    torch.save(ckpt, path)
    return str(path)


@pytest.fixture
def sparl_imaging_adata(tmp_path_factory):
    """AnnData for SPARL: 10 cells, 3-channel 16x16 PNGs."""
    tmp = tmp_path_factory.mktemp("sparl_imgs")
    n, n_channels, img_size = 10, 3, 16
    adata = AnnData(X=np.zeros((n, 1)))
    adata.obsm["spatial"] = np.random.rand(n, 2).astype(np.float32)
    paths = []
    for i in range(n):
        arr = np.random.randint(0, 255, (img_size, img_size, n_channels), dtype=np.uint8)
        p = tmp / f"cell_{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    adata.obs["img_path"] = paths
    return adata
