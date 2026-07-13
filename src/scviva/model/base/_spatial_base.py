from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scvi.model.base import BaseModelClass

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class SpatialBaseModel(BaseModelClass):
    """Base class for all scviva models.

    Extends scvi's BaseModelClass with:
    - SpatialData integration (setup_spatialdata / from_spatialdata)
    - RAPIDS-accelerated latent representation
    - Spatial embedding and prediction plots

    All scviva core models (SCVIVA, DestVI, ResolVI) inherit from this class.
    """

    # ------------------------------------------------------------------ #
    # Backend helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _maybe_rapids(result, backend: str):
        """Optionally move ``result`` to a cupy array when ``backend == "rapids"``.

        Generic backend cast shared by all scviva models, so the
        ``backend="rapids"`` dispatch is implemented once.

        Parameters
        ----------
        result
            A numpy array or a tuple of numpy arrays.
        backend
            ``"cpu"`` (returns ``result`` unchanged) or ``"rapids"`` (casts to cupy).
        """
        if backend != "rapids":
            return result
        try:
            import cupy as cp
        except ImportError as e:
            raise ImportError(
                "backend='rapids' requires cupy. Install with: pip install 'scviva-tools[rapids]'"
            ) from e
        if isinstance(result, tuple):
            return tuple(cp.asarray(r) for r in result)
        return cp.asarray(result)

    # ------------------------------------------------------------------ #
    # SpatialData integration
    # ------------------------------------------------------------------ #

    @classmethod
    def setup_spatialdata(
        cls,
        sdata,
        table_key: str = "table",
        region: str | None = None,
        **kwargs,
    ) -> None:
        """Register fields from a SpatialData object.

        Extracts the AnnData table at ``sdata[table_key]`` and calls
        :meth:`setup_anndata`. Follows the same classmethod convention as
        :meth:`setup_anndata` — call this before constructing the model.

        Parameters
        ----------
        sdata
            A :class:`spatialdata.SpatialData` object.
        table_key
            Key in ``sdata`` pointing to the AnnData table.
        region
            Region name to subset (stored in ``sdata[table_key].obs``).
            If None, the full table is used.
        **kwargs
            Passed to :meth:`setup_anndata`.
        """
        try:
            import spatialdata  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "spatialdata is required for setup_spatialdata. "
                "Install with: pip install 'scviva-tools[spatial]'"
            ) from e

        if not hasattr(sdata, "__getitem__"):
            raise TypeError(
                f"Expected a SpatialData object, got {type(sdata).__name__}. "
                "Install spatialdata or pass an AnnData to setup_anndata."
            )

        adata = sdata[table_key]
        if region is not None:
            region_key = adata.uns.get("spatialdata_attrs", {}).get("region_key", "region")
            adata = adata[adata.obs[region_key] == region].copy()

        cls.setup_anndata(adata, **kwargs)

    @classmethod
    def from_spatialdata(
        cls,
        sdata,
        table_key: str = "table",
        region: str | None = None,
        **model_kwargs,
    ):
        """Convenience constructor from a SpatialData object.

        Calls :meth:`setup_spatialdata` then constructs and returns the model.

        Parameters
        ----------
        sdata
            A :class:`spatialdata.SpatialData` object.
        table_key
            Key in ``sdata`` pointing to the AnnData table.
        region
            Region name to subset.
        **model_kwargs
            Passed to the model constructor.

        Returns
        -------
        Instantiated model.
        """
        cls.setup_spatialdata(sdata, table_key=table_key, region=region)
        adata = sdata[table_key]
        if region is not None:
            region_key = adata.uns.get("spatialdata_attrs", {}).get("region_key", "region")
            adata = adata[adata.obs[region_key] == region].copy()
        return cls(adata, **model_kwargs)

    # ------------------------------------------------------------------ #
    # Latent representation with RAPIDS dispatch
    # ------------------------------------------------------------------ #

    def get_latent_representation(
        self,
        adata=None,
        indices=None,
        give_mean: bool = True,
        batch_size: int | None = None,
        backend: str = "cpu",
        **kwargs,
    ) -> np.ndarray:
        """Return latent representation with optional RAPIDS acceleration.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the model's registered adata.
        indices
            Cell indices to use. If None, all cells are used.
        give_mean
            Return distribution mean rather than a sample.
        batch_size
            Mini-batch size.
        backend
            ``"cpu"`` (default) returns a numpy array as normal.
            ``"rapids"`` transfers the result to a cupy array for downstream
            GPU-accelerated UMAP/clustering (requires ``pip install cuml``).
        **kwargs
            Forwarded to the parent ``get_latent_representation``.

        Returns
        -------
        Latent representation as a numpy array (cpu) or cupy array (rapids).
        """
        latent = super().get_latent_representation(
            adata=adata,
            indices=indices,
            give_mean=give_mean,
            batch_size=batch_size,
            **kwargs,
        )
        return self._maybe_rapids(latent, backend)

    # ------------------------------------------------------------------ #
    # Spatial visualisation
    # ------------------------------------------------------------------ #

    def plot_spatial_embedding(
        self,
        adata=None,
        basis: str = "spatial",
        color=None,
        **kwargs,
    ):
        """Plot latent embedding overlaid on tissue spatial coordinates.

        A thin wrapper around :func:`scanpy.pl.embedding` that defaults
        ``basis`` to the spatial coordinate key so cells are displayed in
        tissue space.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the model's registered adata.
        basis
            Key in ``adata.obsm`` containing 2D spatial coordinates.
        color
            Keys to color cells by (obs columns, gene names, etc.).
        **kwargs
            Forwarded to :func:`scanpy.pl.embedding`.
        """
        import scanpy as sc

        adata = self._validate_anndata(adata)
        return sc.pl.embedding(adata, basis=basis, color=color, **kwargs)
