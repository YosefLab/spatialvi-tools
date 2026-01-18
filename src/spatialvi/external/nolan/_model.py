"""Wrapper for the NOLAN algorithm.

This module exposes the :class:`Nolan`, a high-level interface to the
NOLAN ("NO Label Analysis of Niches") algorithm for self-supervised
identification of spatial tissue domains. The model accepts a
precomputed low-dimensional representation (e.g. scVI or ResolVI
embeddings) together with spatial coordinates and learns to assign
each cell or spot to a "niche" and to reconstruct the connectivity
architecture between niches. Training leverages the implementation
provided by the external ``nolan`` package.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)

try:
    import nolan
except ImportError:
    nolan = None


class Nolan:
    """High-level interface to the NOLAN spatial niche model.

    NOLAN (NO Label ANalysis) uses self-supervised learning to identify
    spatial niches without requiring cell type annotations.

    Parameters
    ----------
    adata
        AnnData object containing spatial transcriptomic data. It must have
        a low-dimensional embedding in ``obsm[emb_key]`` and spatial
        coordinates in ``obsm[spatial_key]``.
    emb_key
        Key in ``adata.obsm`` pointing to a cell embedding. Defaults to
        ``"X_scVI"``.
    spatial_key
        Key in ``adata.obsm`` with spatial coordinates. Defaults to
        ``"spatial"``.
    batch_key
        Optional key in ``adata.obs`` specifying slide or batch identifiers.
        If provided, NOLAN will compute a grid size per batch.
    num_niches
        Dimensionality of the niche representation (i.e. number of niches).
    **model_kwargs
        Additional keyword arguments forwarded to ``nolan.tl.NOLAN``.

    Notes
    -----
    NOLAN is distributed separately from spatialvi-tools. To use this
    wrapper you must install the ``nolan`` package (e.g. via
    ``pip install nolan``). If the dependency is missing, the
    constructor will still succeed but calls to :meth:`train` and
    :meth:`predict` will raise :class:`ImportError`.

    Examples
    --------
    >>> import scanpy as sc
    >>> from spatialvi.external import Nolan
    >>> adata = sc.read("spatial_data.h5ad")
    >>> # Ensure embeddings exist
    >>> assert "X_scVI" in adata.obsm
    >>> model = Nolan(adata, num_niches=20)
    >>> model.train(num_epochs=50)
    >>> adata = model.predict()
    """

    def __init__(
        self,
        adata: AnnData,
        emb_key: str = "X_scVI",
        spatial_key: str = "spatial",
        batch_key: str | None = None,
        num_niches: int = 50,
        **model_kwargs: Any,
    ) -> None:
        self.adata = adata
        self.emb_key = emb_key
        self.spatial_key = spatial_key
        self.batch_key = batch_key
        self.num_niches = num_niches
        self.model_kwargs = model_kwargs
        self._model: Any = None
        self._is_trained = False

    @staticmethod
    def _require_nolan() -> None:
        if nolan is None:
            raise ImportError(
                "The 'nolan' package is not installed. "
                "Install it with ``pip install nolan``."
            )

    def train(
        self,
        ckpt_dir: str | None = None,
        num_epochs: int = 10,
        **train_kwargs: Any,
    ) -> None:
        """Fit the NOLAN model to the stored AnnData.

        Parameters
        ----------
        ckpt_dir
            Directory in which model checkpoints will be stored. If None,
            checkpoints are not saved.
        num_epochs
            Number of training epochs.
        **train_kwargs
            Additional keyword arguments forwarded to the training method.

        Notes
        -----
        This method delegates to ``nolan.tl.NOLAN`` and ``nolan.tl.NOLAN.fit``.
        Internally it computes a global crop radius and number of cells in
        global crops using ``nolan.pp.find_grid_size``.
        """
        self._require_nolan()

        # Determine embedding and spatial matrices
        emb = self.adata.obsm[self.emb_key]
        spatial = self.adata.obsm[self.spatial_key]

        # Determine grid size per batch (if batch_key provided) or global
        batch_key = None
        if self.batch_key is not None and self.batch_key in self.adata.obs:
            batch_key = self.batch_key

        # Compute grid size using NOLAN preprocessing utilities
        grid_results = nolan.pp.find_grid_size(
            self.adata,
            expected_num_cells=50,
            batch_key=batch_key,
            spatial_key=self.spatial_key,
        )
        global_crop_rad, mean_cell_count, max_num_cell = grid_results

        # Initialize the underlying NOLAN model
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
            **train_kwargs,
        )
        self._is_trained = True
        logger.info("NOLAN training complete.")

    def predict(
        self,
        adata: AnnData | None = None,
        batch_size: int = 2048,
        store_key: str = "X_nolan",
        **kwargs: Any,
    ) -> AnnData:
        """Generate niche embeddings for the given AnnData.

        Parameters
        ----------
        adata
            The AnnData on which to perform prediction. If ``None``, the
            training AnnData is used.
        batch_size
            Batch size used during prediction.
        store_key
            Name under which to store the embeddings in ``adata.obsm``.
        **kwargs
            Additional keyword arguments forwarded to ``nolan.tl.NOLAN.predict``.

        Returns
        -------
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

    def get_niche_assignments(
        self,
        adata: AnnData | None = None,
        n_clusters: int | None = None,
        resolution: float = 1.0,
        store_key: str = "niche_cluster",
    ) -> AnnData:
        """Cluster cells into discrete niches based on NOLAN embeddings.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the training data.
        n_clusters
            Number of clusters. If None, uses Leiden clustering.
        resolution
            Resolution parameter for Leiden clustering.
        store_key
            Key to store cluster assignments in obs.

        Returns
        -------
        AnnData with niche cluster assignments in ``obs[store_key]``.
        """
        import scanpy as sc

        if adata is None:
            adata = self.adata

        if "X_nolan" not in adata.obsm:
            adata = self.predict(adata)

        # Build neighbor graph on NOLAN embedding
        sc.pp.neighbors(adata, use_rep="X_nolan")

        if n_clusters is not None:
            # Use k-means for fixed number of clusters
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            adata.obs[store_key] = kmeans.fit_predict(adata.obsm["X_nolan"]).astype(str)
        else:
            # Use Leiden clustering
            sc.tl.leiden(adata, resolution=resolution, key_added=store_key)

        return adata
