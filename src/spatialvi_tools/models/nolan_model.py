"""Wrapper for the NOLAN algorithm.

This module exposes the :class:`NolanModel`, a high‑level interface to the
NOLAN ("NO Label Analysis of Niches") algorithm for self‑supervised
identification of spatial tissue domains【688704965977238†L7-L13】.  The model accepts a
precomputed low‑dimensional representation (e.g. scVI or ResolVI
embeddings) together with spatial coordinates and learns to assign
each cell or spot to a "niche" and to reconstruct the connectivity
architecture between niches【688704965977238†L7-L13】.  Training leverages the
implementation provided by the external ``nolan`` package; this module
merely wraps it into a consistent API.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import anndata as ad

from ..mixin import AnnDataMixin
from .base._base_model import BaseSpatialModel

try:  # optional dependency
    import nolan
except ImportError:  # pragma: no cover - we deliberately ignore coverage here
    nolan = None  # type: ignore


class NolanModel(AnnDataMixin, BaseSpatialModel):
    """High‑level interface to the NOLAN spatial niche model.

    Parameters
    ----------
    adata:
        AnnData object containing spatial transcriptomic data.  It must have
        a low‑dimensional embedding in ``obsm[emb_key]`` and spatial
        coordinates in ``obsm[spatial_key]``.
    emb_key:
        Key in ``adata.obsm`` pointing to a cell embedding.  Defaults to
        ``"X_scVI"``.
    spatial_key:
        Key in ``adata.obsm`` with spatial coordinates.  Defaults to
        ``"spatial"``.
    batch_key:
        Optional key in ``adata.obs`` specifying slide or batch identifiers.
        If provided, NOLAN will compute a grid size per batch.
    num_niches:
        Dimensionality of the niche representation (i.e. number of niches).
    **model_kwargs:
        Additional keyword arguments forwarded to ``nolan.tl.NOLAN``.

    Notes
    -----
    NOLAN is distributed separately from spatialvi‑tools.  To use this
    wrapper you must install the ``nolan`` package (e.g. via
    ``pip install nolan``).  If the dependency is missing, the
    constructor will still succeed but calls to :meth:`train` and
    :meth:`predict` will raise :class:`ImportError`.
    """

    _REQUIRED_KEYS = {
        # key : message
    }

    def __init__(
        self,
        adata: ad.AnnData,
        emb_key: str = "X_scVI",
        spatial_key: str = "spatial",
        batch_key: Optional[str] = None,
        num_niches: int = 50,
        **model_kwargs: Any,
    ) -> None:
        # Validate required keys using mixin; we don't specify required keys
        AnnDataMixin.__init__(self, adata)
        BaseSpatialModel.__init__(self, adata)
        self.emb_key = emb_key
        self.spatial_key = spatial_key
        self.batch_key = batch_key
        self.num_niches = num_niches
        self.model_kwargs = model_kwargs
        self._model: Any = None

    def _require_nolan(self) -> None:
        if nolan is None:
            raise ImportError(
                "The 'nolan' package is not installed. Install it with ``pip install nolan`` "
            )

    def train(self, ckpt_dir: Optional[str] = None, num_epochs: int = 10) -> None:
        """Fit the NOLAN model to the stored AnnData.

        Parameters
        ----------
        ckpt_dir:
            Directory in which model checkpoints will be stored.  If None,
            checkpoints are not saved.
        num_epochs:
            Number of training epochs.

        Notes
        -----
        This method delegates to ``nolan.tl.NOLAN`` and ``nolan.tl.NOLAN.fit``.
        Internally it computes a global crop radius and number of cells in
        global crops using ``nolan.pp.find_grid_size`` as recommended in
        the NOLAN tutorial【688704965977238†L40-L54】.
        """
        self._require_nolan()
        # Determine embedding and spatial matrices
        emb = self.adata.obsm[self.emb_key]
        spatial = self.adata.obsm[self.spatial_key]
        # Determine grid size per batch (if batch_key provided) or global
        if self.batch_key is not None and self.batch_key in self.adata.obs:
            batch_key = self.batch_key
        else:
            batch_key = None
        # Compute grid size using NOLAN preprocessing utilities
        grid_results = nolan.pp.find_grid_size(
            self.adata,
            expected_num_cells=50,
            batch_key=batch_key,
            spatial_key=self.spatial_key,
        )
        global_crop_rad, mean_cell_count, max_num_cell = grid_results
        # Initialise the underlying NOLAN model
        self._model = nolan.tl.NOLAN(
            input_dim=emb.shape[1],
            out_proj_dim=self.num_niches,
            global_crop_rad=global_crop_rad,
            num_cells_in_global_crop=max_num_cell,
            local_crop_rad=global_crop_rad * 0.75,
            pos_key=self.spatial_key,
            **self.model_kwargs,
        )
        # Optionally configure logging and checkpointing
        if ckpt_dir is not None:
            self._model.set_checkpointing(ckpt_dir=ckpt_dir, every_n_epochs=num_epochs)
        # Fit the model
        self._model.fit(
            self.adata,
            num_epochs=num_epochs,
            batch_key=self.batch_key,
            emb_layer=self.emb_key,
        )

    def predict(
        self,
        adata: Optional[ad.AnnData] = None,
        batch_size: int = 2048,
        store_key: str = "X_nolan",
        **kwargs: Any,
    ) -> ad.AnnData:
        """Generate niche embeddings for the given AnnData.

        Parameters
        ----------
        adata:
            The AnnData on which to perform prediction.  If ``None``, the
            training AnnData is used.
        batch_size:
            Batch size used during prediction.
        store_key:
            Name under which to store the embeddings in ``adata.obsm``.
        **kwargs:
            Additional keyword arguments forwarded to ``nolan.tl.NOLAN.predict``.

        Returns
        -------
        AnnData
            AnnData with a new ``obsm[store_key]`` containing the predicted
            niche embeddings.
        """
        self._require_nolan()
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call `.train()` first.")
        if adata is None:
            adata = self.adata
        # Compute embeddings
        embeddings = self._model.predict(
            adata,
            bs=batch_size,
            batch_key=self.batch_key,
            emb_layer=self.emb_key,
            **kwargs,
        )
        adata.obsm[store_key] = np.asarray(embeddings)
        return adata