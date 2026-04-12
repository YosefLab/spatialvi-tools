"""Save/load utilities for GIMVI models."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from anndata import AnnData

from anndata import read_h5ad
from scvi.data._download import _download


def _load_saved_gimvi_files(
    dir_path: str,
    load_seq_adata: bool,
    load_spatial_adata: bool,
    prefix: str | None = None,
    map_location: Literal["cpu", "cuda"] | None = None,
    backup_url: str | None = None,
) -> tuple[dict, dict, dict, AnnData | None, AnnData | None]:
    """Load GIMVI save files (dual-adata format).

    GIMVI uses a custom save format because it manages two AnnData objects and
    two registries; the standard :meth:`~scvi.model.base.BaseModelClass.load`
    only handles single-adata models.
    """
    file_name_prefix = prefix or ""
    model_file_name = f"{file_name_prefix}model.pt"
    model_path = os.path.join(dir_path, model_file_name)
    try:
        _download(backup_url, dir_path, model_file_name)
        model = torch.load(model_path, map_location=map_location, weights_only=False)
    except FileNotFoundError as exc:
        raise ValueError(f"Failed to load model file at {model_path}.") from exc

    model_state_dict = model["model_state_dict"]
    seq_var_names = model["seq_var_names"]
    spatial_var_names = model["spatial_var_names"]
    attr_dict = model["attr_dict"]

    adata_seq, adata_spatial = None, None
    if load_seq_adata:
        seq_data_path = os.path.join(dir_path, f"{file_name_prefix}adata_seq.h5ad")
        if os.path.exists(seq_data_path):
            adata_seq = read_h5ad(seq_data_path)
    if load_spatial_adata:
        spatial_data_path = os.path.join(dir_path, f"{file_name_prefix}adata_spatial.h5ad")
        if os.path.exists(spatial_data_path):
            adata_spatial = read_h5ad(spatial_data_path)

    return (
        attr_dict,
        seq_var_names,
        spatial_var_names,
        model_state_dict,
        adata_seq,
        adata_spatial,
    )
