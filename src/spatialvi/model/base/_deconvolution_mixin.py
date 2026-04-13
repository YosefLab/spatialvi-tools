from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class SpatialDeconvolutionMixin:
    """Mixin for spatial deconvolution result formatting and visualization.

    Applied to: DestVI only.

    Requires the model to implement:
    - ``self.cell_type_mapping``: np.ndarray of cell type label strings
    - ``self.get_proportions(adata)``: returns np.ndarray of shape (n_spots, n_cell_types)
    """

    def get_proportions_df(self, adata: AnnData | None = None) -> pd.DataFrame:
        """Return cell type proportions as a tidy DataFrame.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the model's registered adata.

        Returns
        -------
        DataFrame of shape (n_spots, n_cell_types) with cell type names as columns.
        Rows sum to 1.
        """
        if adata is None and hasattr(self, "adata"):
            adata = self.adata

        import inspect

        sig = inspect.signature(self.get_proportions)
        params = list(sig.parameters.keys())
        # Call with adata only if the method accepts it as the first positional argument.
        if params and params[0] == "adata":
            proportions = self.get_proportions(adata)
        else:
            proportions = self.get_proportions()

        if isinstance(proportions, pd.DataFrame):
            return proportions

        # Wrap numpy array in a DataFrame with cell type names as columns.
        import numpy as np

        proportions = np.asarray(proportions)
        columns = list(self.cell_type_mapping) if hasattr(self, "cell_type_mapping") else None
        return pd.DataFrame(proportions, columns=columns)

    def plot_cell_type_map(
        self,
        adata: AnnData | None = None,
        cell_type: str | None = None,
        basis: str = "spatial",
        ax=None,
        **kwargs,
    ):
        """Plot spatial map of a single cell type's proportion.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the model's registered adata.
        cell_type
            Name of the cell type to visualize. Must be in ``self.cell_type_mapping``.
        basis
            Key in ``adata.obsm`` for spatial coordinates.
        ax
            Matplotlib axes. If None, a new figure is created.
        **kwargs
            Forwarded to :func:`scanpy.pl.embedding`.
        """
        import scanpy as sc

        if adata is None and hasattr(self, "adata"):
            adata = self.adata

        df = self.get_proportions_df(adata)
        if cell_type is not None:
            if cell_type not in df.columns:
                raise ValueError(
                    f"cell_type '{cell_type}' not found. Available: {list(df.columns)}"
                )
            key = f"_spatialvi_prop_{cell_type}"
            adata.obs[key] = df[cell_type].values
            return sc.pl.embedding(adata, basis=basis, color=key, ax=ax, **kwargs)

        # No cell_type specified — plot all as a grid; write each column to obs first
        logger.info("No cell_type specified; plotting all %d cell types.", len(df.columns))
        keys = []
        for ct in df.columns:
            obs_key = f"_spatialvi_prop_{ct}"
            adata.obs[obs_key] = df[ct].values
            keys.append(obs_key)
        return sc.pl.embedding(adata, basis=basis, color=keys, **kwargs)
