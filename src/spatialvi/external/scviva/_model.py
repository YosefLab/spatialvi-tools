"""scVIVA wrapper model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData
    from collections.abc import Sequence
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _check_scvi_import():
    """Check if scvi-tools is available with scVIVA."""
    try:
        from scvi.external import SCVIVA as _SCVIVA
        return _SCVIVA
    except ImportError:
        raise ImportError(
            "scVIVA requires scvi-tools>=1.1.0 with scVIVA support. "
            "Install with: pip install scvi-tools[scviva]"
        )


class scVIVA:
    """Wrapper for scvi-tools scVIVA model.

    scVIVA models cellular microenvironments by learning
    niche-aware representations that capture both cell-intrinsic
    and neighborhood-specific factors.

    This is a thin wrapper around the scvi-tools implementation
    that provides a consistent interface with spatialvi-tools.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers.
    dropout_rate
        Dropout rate for neural networks.
    **kwargs
        Additional keyword arguments for scvi.external.SCVIVA.

    Examples
    --------
    >>> import spatialvi
    >>> adata = spatialvi.data.synthetic_spatial()
    >>> scVIVA.setup_anndata(adata, spatial_key="spatial")
    >>> model = scVIVA(adata)
    >>> model.train()
    >>> niche_effects = model.get_niche_effects()
    """

    def __init__(
        self,
        adata: "AnnData",
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        _SCVIVA = _check_scvi_import()

        self._model = _SCVIVA(
            adata,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.adata = adata

    @classmethod
    def setup_anndata(
        cls,
        adata: "AnnData",
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        spatial_key: str = "spatial",
        **kwargs,
    ) -> None:
        """Setup AnnData for scVIVA.

        Parameters
        ----------
        adata
            AnnData object.
        layer
            Layer to use for expression data.
        batch_key
            Key for batch information in obs.
        labels_key
            Key for cell type labels in obs.
        spatial_key
            Key for spatial coordinates in obsm.
        **kwargs
            Additional keyword arguments.
        """
        _SCVIVA = _check_scvi_import()
        _SCVIVA.setup_anndata(
            adata,
            layer=layer,
            batch_key=batch_key,
            labels_key=labels_key,
            **kwargs,
        )

    def train(
        self,
        max_epochs: int = 400,
        lr: float = 1e-3,
        accelerator: str = "auto",
        devices: int | str = "auto",
        **kwargs,
    ) -> None:
        """Train the model.

        Parameters
        ----------
        max_epochs
            Maximum number of epochs.
        lr
            Learning rate.
        accelerator
            Accelerator to use.
        devices
            Devices to use.
        **kwargs
            Additional keyword arguments for training.
        """
        self._model.train(
            max_epochs=max_epochs,
            lr=lr,
            accelerator=accelerator,
            devices=devices,
            **kwargs,
        )

    def get_latent_representation(
        self,
        adata: "AnnData" | None = None,
        indices: "Sequence[int]" | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> "NDArray":
        """Get latent representation.

        Parameters
        ----------
        adata
            AnnData object.
        indices
            Indices to use.
        give_mean
            Whether to return mean.
        batch_size
            Batch size.

        Returns
        -------
        Latent representation array.
        """
        return self._model.get_latent_representation(
            adata=adata,
            indices=indices,
            give_mean=give_mean,
            batch_size=batch_size,
        )

    def get_niche_effects(
        self,
        adata: "AnnData" | None = None,
        indices: "Sequence[int]" | None = None,
        batch_size: int | None = None,
    ) -> "NDArray":
        """Get niche effects for each cell.

        Parameters
        ----------
        adata
            AnnData object.
        indices
            Indices to use.
        batch_size
            Batch size.

        Returns
        -------
        Niche effects array.
        """
        if hasattr(self._model, "get_niche_effects"):
            return self._model.get_niche_effects(
                adata=adata,
                indices=indices,
                batch_size=batch_size,
            )
        else:
            # Fallback - get niche-specific latent
            return self.get_latent_representation(
                adata=adata,
                indices=indices,
                batch_size=batch_size,
            )

    def differential_niche_expression(
        self,
        groupby: str,
        group1: str | list[str],
        group2: str | list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Perform niche-aware differential expression.

        Parameters
        ----------
        groupby
            Key in obs for grouping.
        group1
            First group.
        group2
            Second group.
        **kwargs
            Additional arguments.

        Returns
        -------
        DataFrame with DE results.
        """
        if hasattr(self._model, "differential_niche_expression"):
            return self._model.differential_niche_expression(
                groupby=groupby,
                group1=group1,
                group2=group2,
                **kwargs,
            )
        else:
            return self._model.differential_expression(
                groupby=groupby,
                group1=group1,
                group2=group2,
                **kwargs,
            )

    def save(self, dir_path: str, **kwargs) -> None:
        """Save model to disk."""
        self._model.save(dir_path, **kwargs)

    @classmethod
    def load(cls, dir_path: str, adata: "AnnData" | None = None, **kwargs) -> "scVIVA":
        """Load model from disk."""
        _SCVIVA = _check_scvi_import()
        instance = cls.__new__(cls)
        instance._model = _SCVIVA.load(dir_path, adata=adata, **kwargs)
        instance.adata = instance._model.adata
        return instance
