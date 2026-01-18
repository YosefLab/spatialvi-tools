"""DestVI wrapper model."""

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
    """Check if scvi-tools is available with DestVI."""
    try:
        from scvi.model import DestVI as _DestVI

        return _DestVI
    except ImportError:
        raise ImportError("DestVI requires scvi-tools>=1.0.0. Install with: pip install scvi-tools") from None


def _check_condscvi_import():
    """Check if CondSCVI is available."""
    try:
        from scvi.model import CondSCVI as _CondSCVI

        return _CondSCVI
    except ImportError:
        raise ImportError("CondSCVI requires scvi-tools>=1.0.0. Install with: pip install scvi-tools") from None


class DestVI:
    """Wrapper for scvi-tools DestVI model.

    DestVI performs multi-resolution spatial deconvolution, estimating
    both cell type proportions and continuous sub-cell-type variation
    within spatial transcriptomics spots.

    This is a thin wrapper around the scvi-tools implementation
    that provides a consistent interface with spatialvi-tools.

    Parameters
    ----------
    st_adata
        Spatial transcriptomics AnnData.
    sc_model
        Trained CondSCVI model on reference single-cell data.
    **kwargs
        Additional keyword arguments for scvi.model.DestVI.

    Examples
    --------
    >>> import spatialvi
    >>> # First train CondSCVI on reference
    >>> sc_adata = spatialvi.data.synthetic_scrna()
    >>> DestVI.setup_anndata(sc_adata, labels_key="cell_type")
    >>> sc_model = spatialvi.external.CondSCVI(sc_adata)
    >>> sc_model.train()
    >>> # Then train DestVI on spatial
    >>> st_adata = spatialvi.data.synthetic_spatial()
    >>> model = DestVI.from_rna_model(st_adata, sc_model)
    >>> model.train()
    >>> proportions = model.get_proportions()
    """

    def __init__(
        self,
        st_adata: AnnData,
        sc_model,
        **kwargs,
    ):
        _DestVI = _check_scvi_import()

        self._model = _DestVI.from_rna_model(
            st_adata,
            sc_model,
            **kwargs,
        )
        self.adata = st_adata

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        labels_key: str | None = None,
        **kwargs,
    ) -> None:
        """Setup AnnData for DestVI (reference single-cell data).

        Parameters
        ----------
        adata
            AnnData object (single-cell reference).
        layer
            Layer to use for expression data.
        labels_key
            Key for cell type labels in obs.
        **kwargs
            Additional keyword arguments.
        """
        _CondSCVI = _check_condscvi_import()
        _CondSCVI.setup_anndata(
            adata,
            layer=layer,
            labels_key=labels_key,
            **kwargs,
        )

    @classmethod
    def from_rna_model(
        cls,
        st_adata: AnnData,
        sc_model,
        **kwargs,
    ) -> DestVI:
        """Create DestVI from a trained CondSCVI model.

        Parameters
        ----------
        st_adata
            Spatial transcriptomics AnnData.
        sc_model
            Trained CondSCVI model.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        DestVI model instance.
        """
        instance = cls.__new__(cls)
        _DestVI = _check_scvi_import()
        instance._model = _DestVI.from_rna_model(st_adata, sc_model, **kwargs)
        instance.adata = st_adata
        return instance

    def train(
        self,
        max_epochs: int = 2500,
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

    def get_proportions(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        return_dataframe: bool = True,
    ) -> NDArray | pd.DataFrame:
        """Get estimated cell type proportions.

        Parameters
        ----------
        adata
            AnnData object.
        indices
            Indices to use.
        batch_size
            Batch size.
        return_dataframe
            Whether to return as DataFrame.

        Returns
        -------
        Cell type proportions.
        """
        proportions = self._model.get_proportions(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )

        if return_dataframe and isinstance(proportions, np.ndarray):
            adata = adata if adata is not None else self.adata
            obs_names = adata.obs_names if indices is None else adata.obs_names[indices]
            # Get cell type names from the model
            if hasattr(self._model, "cell_type_mapping"):
                columns = self._model.cell_type_mapping
            else:
                columns = [f"CellType_{i}" for i in range(proportions.shape[1])]
            return pd.DataFrame(proportions, index=obs_names, columns=columns)

        return proportions

    def get_gamma(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, NDArray]:
        """Get sub-cell-type variation (gamma) per cell type.

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
        Dictionary mapping cell types to gamma arrays.
        """
        return self._model.get_gamma(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )

    def get_scale_for_ct(
        self,
        cell_type: str,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> NDArray:
        """Get cell type-specific expression scale.

        Parameters
        ----------
        cell_type
            Cell type name.
        adata
            AnnData object.
        indices
            Indices to use.
        batch_size
            Batch size.

        Returns
        -------
        Expression scale array for the cell type.
        """
        return self._model.get_scale_for_ct(
            cell_type,
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )

    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> NDArray:
        """Get latent representation.

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
        Latent representation array.
        """
        if hasattr(self._model, "get_latent_representation"):
            return self._model.get_latent_representation(
                adata=adata,
                indices=indices,
                batch_size=batch_size,
            )
        else:
            # Return proportions as representation
            return self.get_proportions(
                adata=adata,
                indices=indices,
                batch_size=batch_size,
                return_dataframe=False,
            )

    def save(self, dir_path: str, **kwargs) -> None:
        """Save model to disk."""
        self._model.save(dir_path, **kwargs)

    @classmethod
    def load(cls, dir_path: str, adata: AnnData | None = None, **kwargs) -> DestVI:
        """Load model from disk."""
        _DestVI = _check_scvi_import()
        instance = cls.__new__(cls)
        instance._model = _DestVI.load(dir_path, adata=adata, **kwargs)
        instance.adata = instance._model.adata
        return instance


class CondSCVI:
    """Wrapper for CondSCVI (required for DestVI reference).

    CondSCVI is a conditional VAE for single-cell data that serves
    as the reference model for DestVI spatial deconvolution.
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        _CondSCVI = _check_condscvi_import()
        self._model = _CondSCVI(
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
        labels_key: str | None = None,
        **kwargs,
    ) -> None:
        """Setup AnnData for CondSCVI."""
        _CondSCVI = _check_condscvi_import()
        _CondSCVI.setup_anndata(
            adata,
            layer=layer,
            labels_key=labels_key,
            **kwargs,
        )

    def train(self, max_epochs: int = 400, **kwargs) -> None:
        """Train the model."""
        self._model.train(max_epochs=max_epochs, **kwargs)

    def get_latent_representation(self, **kwargs) -> NDArray:
        """Get latent representation."""
        return self._model.get_latent_representation(**kwargs)

    def save(self, dir_path: str, **kwargs) -> None:
        """Save model to disk."""
        self._model.save(dir_path, **kwargs)

    @classmethod
    def load(cls, dir_path: str, adata: AnnData | None = None, **kwargs) -> CondSCVI:
        """Load model from disk."""
        _CondSCVI = _check_condscvi_import()
        instance = cls.__new__(cls)
        instance._model = _CondSCVI.load(dir_path, adata=adata, **kwargs)
        instance.adata = instance._model.adata
        return instance
