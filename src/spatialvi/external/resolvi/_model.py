"""ResolVI wrapper model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _check_scvi_import():
    """Check if scvi-tools is available with ResolVI."""
    try:
        from scvi.external import RESOLVI as _RESOLVI

        return _RESOLVI
    except ImportError:
        raise ImportError(
            "ResolVI requires scvi-tools>=1.1.0 with ResolVI support. Install with: pip install scvi-tools"
        ) from None


class ResolVI:
    """Wrapper for scvi-tools ResolVI model.

    ResolVI denoises cellular-resolved spatial transcriptomics data
    by modeling background contamination and segmentation errors.
    Suitable for Xenium, MERFISH, CosMx, and similar technologies.

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
        Additional keyword arguments for scvi.external.RESOLVI.

    Examples
    --------
    >>> import spatialvi
    >>> adata = spatialvi.data.load_xenium()
    >>> ResolVI.setup_anndata(adata, spatial_key="spatial")
    >>> model = ResolVI(adata)
    >>> model.train()
    >>> denoised = model.get_denoised_expression()
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        _RESOLVI = _check_scvi_import()

        self._model = _RESOLVI(
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
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        spatial_key: str = "spatial",
        **kwargs,
    ) -> None:
        """Setup AnnData for ResolVI.

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
        _RESOLVI = _check_scvi_import()
        _RESOLVI.setup_anndata(
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
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> NDArray:
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

    def get_denoised_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        n_samples: int = 1,
        batch_size: int | None = None,
    ) -> NDArray:
        """Get denoised expression.

        Parameters
        ----------
        adata
            AnnData object.
        indices
            Indices to use.
        n_samples
            Number of posterior samples.
        batch_size
            Batch size.

        Returns
        -------
        Denoised expression array.
        """
        if hasattr(self._model, "get_denoised_expression"):
            return self._model.get_denoised_expression(
                adata=adata,
                indices=indices,
                n_samples=n_samples,
                batch_size=batch_size,
            )
        else:
            return self._model.get_normalized_expression(
                adata=adata,
                indices=indices,
                n_samples=n_samples,
                batch_size=batch_size,
            )

    def get_background_fraction(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> NDArray:
        """Get estimated background fraction per cell.

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
        Background fraction array.
        """
        if hasattr(self._model, "get_background_fraction"):
            return self._model.get_background_fraction(
                adata=adata,
                indices=indices,
                batch_size=batch_size,
            )
        else:
            logger.warning("Background fraction not available in this version")
            return np.zeros(self.adata.n_obs if adata is None else adata.n_obs)

    def predict(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> pd.DataFrame:
        """Predict cell types.

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
        DataFrame with predictions.
        """
        if hasattr(self._model, "predict"):
            return self._model.predict(
                adata=adata,
                indices=indices,
                batch_size=batch_size,
            )
        else:
            raise NotImplementedError("Prediction not available")

    def save(self, dir_path: str, **kwargs) -> None:
        """Save model to disk."""
        self._model.save(dir_path, **kwargs)

    @classmethod
    def load(cls, dir_path: str, adata: AnnData | None = None, **kwargs) -> ResolVI:
        """Load model from disk."""
        _RESOLVI = _check_scvi_import()
        instance = cls.__new__(cls)
        instance._model = _RESOLVI.load(dir_path, adata=adata, **kwargs)
        instance.adata = instance._model.adata
        return instance
