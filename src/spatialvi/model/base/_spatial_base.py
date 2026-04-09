from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scvi.model.base import BaseModelClass

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


# =========================================================================
# Consolidation log (Task 21)
# =========================================================================
# The following patterns were audited across scVIVA, DestVI, and ResolVI:
#
# DONE (extracted to shared base):
# - Spatial coordinate validation → SpatialCoordsField (Task 4)
# - Neighbor graph registration → SpatialNeighborhoodMixin._setup_neighbor_field() (Task 6)
# - RAPIDS latent representation dispatch → SpatialBaseModel.get_latent_representation() (Task 5)
# - SpatialData integration → SpatialBaseModel.setup_spatialdata() / from_spatialdata() (Task 5)
#
# AUDITED - LEFT IN MODELS:
# - NB/ZINB dispersion string validation in modules (_nichevae.py, _resolvae.py):
#   Each module validates its own dispersion string against a model-specific set of allowed
#   values ("gene", "gene-batch", "gene-label", "gene-cell" for scVIVA; only "gene" /
#   "gene-batch" for ResolVI; none for DestVI which uses a fixed deconvolution formulation).
#   The allowed sets differ per model, so extraction would require branching logic that adds
#   more abstraction than it removes (< 5 lines saved per model).
# - get_normalized_expression with size-factor scaling:
#   scVIVA inherits this from scvi's RNASeqMixin; ResolVI uses a bespoke
#   ResolVIPredictiveMixin (get_normalized_expression / get_normalized_expression_importance)
#   that applies residual-model sampling — entirely different logic, not shareable.
#   DestVI exposes get_proportions / get_gamma, not normalized expression.
# - Neighbor index → one-hot encoding in modules:
#   Only _resolvae.py uses one_hot on neighbor indices (for cell-type composition
#   in the spatial prior). _nichevae.py uses raw niche composition obsm arrays.
#   No shared pattern exists across models.
# - _prepare_data (neighbor precomputation):
#   Added to ResolVI as a static method (Task 21) to match upstream RESOLVI API.
#   scVIVA uses a higher-level preprocessing_anndata() with different parameters.
#   DestVI has no spatial neighbor requirement. Not generalizable to a shared base.
# =========================================================================


class SpatialBaseModel(BaseModelClass):
    """Base class for all spatialvi models.

    Extends scvi's BaseModelClass with:
    - SpatialData integration (setup_spatialdata / from_spatialdata)
    - RAPIDS-accelerated latent representation
    - Spatial embedding and prediction plots

    All spatialvi core models (SCVIVA, DestVI, ResolVI) inherit from this class.
    """

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
                "Install with: pip install 'spatialvi-tools[spatial]'"
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
        if backend == "rapids":
            try:
                import cupy as cp

                return cp.asarray(latent)
            except ImportError as e:
                raise ImportError(
                    "backend='rapids' requires cupy. "
                    "Install with: pip install 'spatialvi-tools[rapids]'"
                ) from e
        return latent

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

    def plot_spatial_predictions(
        self,
        adata=None,
        key=None,
        basis: str = "spatial",
        **kwargs,
    ):
        """Plot any obsm/obs key overlaid on tissue coordinates.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the model's registered adata.
        key
            obs or obsm key to visualize (e.g., latent cluster, cell type).
        basis
            Spatial coordinate key.
        **kwargs
            Forwarded to :func:`scanpy.pl.embedding`.
        """
        return self.plot_spatial_embedding(adata=adata, basis=basis, color=key, **kwargs)
