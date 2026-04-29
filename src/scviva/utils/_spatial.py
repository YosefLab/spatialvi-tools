from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def get_spatial_coords(adata: AnnData, key: str = "spatial") -> np.ndarray:
    """Extract spatial coordinates from AnnData.obsm.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates stored in obsm.
    key
        Key in adata.obsm containing the spatial coordinates.

    Returns
    -------
    Array of shape (n_obs, 2) or (n_obs, 3).
    """
    if key not in adata.obsm:
        raise KeyError(
            f"'{key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
        )
    return np.asarray(adata.obsm[key])


def validate_spatial_coords(coords: np.ndarray) -> None:
    """Validate that spatial coordinate array has correct shape.

    Parameters
    ----------
    coords
        Array of spatial coordinates.

    Raises
    ------
    ValueError
        If coordinates are not 2D or 3D.
    """
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError(
            f"Spatial coordinates must be 2D or 3D (shape: (n_obs, 2) or (n_obs, 3)), "
            f"got shape {coords.shape}."
        )
