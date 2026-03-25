from __future__ import annotations

import numpy as np
from anndata import AnnData


def get_spatial_coords(adata: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    """Extract spatial coordinates from AnnData.obsm.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates stored in obsm.
    spatial_key
        Key in adata.obsm containing the spatial coordinates.

    Returns
    -------
    Array of shape (n_obs, 2) or (n_obs, 3).
    """
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"Spatial coordinates key '{spatial_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    coords = np.asarray(adata.obsm[spatial_key])
    validate_spatial_coords(coords)
    return coords


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
