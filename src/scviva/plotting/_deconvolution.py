from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData

logger = logging.getLogger(__name__)


def plot_cell_type_map(
    adata: AnnData,
    proportions: pd.DataFrame,
    cell_type: str | None = None,
    basis: str = "spatial",
    ax=None,
    **kwargs,
):
    """Plot spatial map of cell type proportion(s).

    Parameters
    ----------
    adata
        AnnData object to plot into. Proportion columns are written to ``adata.obs``
        as a side effect.
    proportions
        DataFrame of shape (n_spots, n_cell_types) with cell type names as columns,
        aligned to ``adata.obs_names``.
    cell_type
        Name of the cell type to visualize. Must be a column of ``proportions``.
        If None, all cell types are plotted as a grid.
    basis
        Key in ``adata.obsm`` for spatial coordinates.
    ax
        Matplotlib axes. If None, a new figure is created.
    **kwargs
        Forwarded to :func:`scanpy.pl.embedding`.
    """
    import scanpy as sc

    if cell_type is not None:
        if cell_type not in proportions.columns:
            raise ValueError(
                f"cell_type '{cell_type}' not found. Available: {list(proportions.columns)}"
            )
        key = f"_scviva_prop_{cell_type}"
        adata.obs[key] = proportions[cell_type].values
        return sc.pl.embedding(adata, basis=basis, color=key, ax=ax, **kwargs)

    # No cell_type specified — plot all as a grid; write each column to obs first
    logger.info("No cell_type specified; plotting all %d cell types.", len(proportions.columns))
    keys = []
    for ct in proportions.columns:
        obs_key = f"_scviva_prop_{ct}"
        adata.obs[obs_key] = proportions[ct].values
        keys.append(obs_key)
    return sc.pl.embedding(adata, basis=basis, color=keys, **kwargs)
