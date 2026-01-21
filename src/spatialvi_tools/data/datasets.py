"""Small example datasets for testing and demonstration.

The functions in this module create simple `AnnData` objects with
randomly generated counts and coordinates.  They are intended for
unit tests and quick start examples.  Real data should be loaded
through appropriate loader functions.
"""

from __future__ import annotations

import numpy as np
import anndata as ad

__all__ = ["load_dummy_spatial_dataset"]


def load_dummy_spatial_dataset(
    n_cells: int = 100,
    n_genes: int = 50,
    n_dims: int = 2,
    seed: int | None = 0,
) -> ad.AnnData:
    """Generate a toy spatial transcriptomics dataset.

    Parameters
    ----------
    n_cells:
        Number of observation points (cells or spots).
    n_genes:
        Number of genes measured in the count matrix.
    n_dims:
        Dimensionality of the spatial coordinates (2 or 3).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    anndata.AnnData
        An AnnData object with a random count matrix stored in ``X`` and
        spatial coordinates in ``obsm["spatial"]``.
    """
    rng = np.random.default_rng(seed)
    # generate random counts (Poisson distributed)
    counts = rng.poisson(lam=5.0, size=(n_cells, n_genes)).astype(np.float32)
    var_names = [f"gene_{i}" for i in range(n_genes)]
    obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata = ad.AnnData(X=counts, obs={"obs_names": obs_names}, var={"var_names": var_names})
    # random spatial coordinates
    coords = rng.uniform(low=0, high=10, size=(n_cells, n_dims)).astype(np.float32)
    adata.obsm["spatial"] = coords
    # create a dummy embedding for use with models expecting precomputed features
    adata.obsm["X_scVI"] = rng.normal(size=(n_cells, 16)).astype(np.float32)
    return adata