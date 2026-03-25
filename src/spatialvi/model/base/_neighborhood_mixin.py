from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)

_VALID_BACKENDS = ("squidpy", "rapids")


class SpatialNeighborhoodMixin:
    """Mixin for spatial neighbor graph computation.

    Applied to: SCVIVA, ResolVI.
    Provides a single entry point for neighbor graph computation with
    pluggable backends (squidpy on CPU, RAPIDS on GPU).

    The computed neighbor arrays are stored in:
    - ``adata.obsm["index_neighbor"]``    — dense int array, shape (n_cells, n_neighs)
    - ``adata.obsm["distance_neighbor"]`` — dense float array, shape (n_cells, n_neighs)

    These keys match the upstream ResolVI module's expected input format.
    """

    def compute_neighbors(
        self,
        adata: AnnData,
        spatial_key: str = "spatial",
        coord_type: str = "generic",
        n_neighs: int = 6,
        backend: Literal["squidpy", "rapids"] = "squidpy",
    ) -> None:
        """Compute spatial neighbor graph and store in ``adata.obsm``.

        Parameters
        ----------
        adata
            AnnData object with spatial coordinates in ``adata.obsm[spatial_key]``.
        spatial_key
            Key in ``adata.obsm`` for spatial coordinates.
        coord_type
            Coordinate type passed to squidpy (``"generic"`` or ``"visium"``).
            Ignored when ``backend="rapids"``.
        n_neighs
            Number of nearest neighbors.
        backend
            ``"squidpy"`` (default): uses :func:`squidpy.gr.spatial_neighbors`.
            ``"rapids"``: uses cuGraph/cuML for GPU-accelerated computation.
        """
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got '{backend}'.")
        if backend == "squidpy":
            self._compute_neighbors_squidpy(adata, spatial_key, coord_type, n_neighs)
        else:
            self._compute_neighbors_rapids(adata, spatial_key, n_neighs)

    def _compute_neighbors_squidpy(
        self,
        adata: AnnData,
        spatial_key: str,
        coord_type: str,
        n_neighs: int,
    ) -> None:
        try:
            import squidpy as sq
        except ImportError as e:
            raise ImportError(
                "squidpy is required for backend='squidpy'. "
                "Install with: pip install 'spatialvi-tools[spatial]'"
            ) from e

        sq.gr.spatial_neighbors(
            adata,
            spatial_key=spatial_key,
            coord_type=coord_type,
            n_neighs=n_neighs,
            key_added="spatial_neighbors",
        )

        import scipy.sparse as sp

        conn = adata.obsp.get("spatial_neighbors_connectivities")
        dist = adata.obsp.get("spatial_neighbors_distances")

        n = adata.n_obs
        idx = np.zeros((n, n_neighs), dtype=np.int64)
        dst = np.zeros((n, n_neighs), dtype=np.float32)

        if conn is not None:
            cx = sp.csr_matrix(conn)
            dist_csr = sp.csr_matrix(dist) if dist is not None else None
            for i in range(n):
                row = cx[i].indices
                d_row = (
                    dist_csr[i].data
                    if dist_csr is not None
                    else np.ones(len(row), dtype=np.float32)
                )
                k = min(n_neighs, len(row))
                idx[i, :k] = row[:k]
                dst[i, :k] = d_row[:k]

        adata.obsm["index_neighbor"] = idx
        adata.obsm["distance_neighbor"] = dst
        logger.info("Computed %d spatial neighbors (squidpy backend).", n_neighs)

    def _compute_neighbors_rapids(
        self,
        adata: AnnData,
        spatial_key: str,
        n_neighs: int,
    ) -> None:
        try:
            import cuml
            import cupy as cp
        except ImportError as e:
            raise ImportError(
                "backend='rapids' requires cuml and cupy. "
                "Install with: pip install 'spatialvi-tools[rapids]'"
            ) from e

        coords = cp.asarray(adata.obsm[spatial_key].astype(np.float32))
        nn = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighs + 1)
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
        # drop self (index 0)
        adata.obsm["index_neighbor"] = cp.asnumpy(indices[:, 1:]).astype(np.int64)
        adata.obsm["distance_neighbor"] = cp.asnumpy(distances[:, 1:]).astype(np.float32)
        logger.info("Computed %d spatial neighbors (RAPIDS backend).", n_neighs)

    def _setup_neighbor_field(self, adata: AnnData) -> list:
        """Return neighbor obsm fields for registration in AnnDataManager.

        Call this inside setup_anndata after computing neighbors. Returns
        a list of NeighborhoodGraphField instances to include in the
        AnnDataManager fields list.
        """
        from spatialvi.data._fields import NeighborhoodGraphField

        for key in ("index_neighbor", "distance_neighbor"):
            if key not in adata.obsm:
                raise KeyError(
                    f"'{key}' not found in adata.obsm. "
                    "Call model.compute_neighbors(adata) before setup_anndata."
                )
        return [
            NeighborhoodGraphField(obsm_key="index_neighbor"),
            NeighborhoodGraphField(obsm_key="distance_neighbor"),
        ]
