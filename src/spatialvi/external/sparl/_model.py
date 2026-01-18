"""Wrapper for the SPARL spatial proteomics model.

SPARL stands for Spatial Proteomics Analysis with Representation Learning.
The package aims to learn latent representations of spatial proteomic data
which can be used for downstream tasks like cell type annotation,
spatial domain detection, and protein imputation.

This module provides a wrapper interface to the SPARL package. Since SPARL
is designed specifically for spatial proteomics data (e.g., from imaging
mass cytometry or CODEX), it handles protein expression matrices with
spatial coordinates differently from RNA-focused methods.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from anndata import AnnData

logger = logging.getLogger(__name__)

try:
    import sparl
except ImportError:
    sparl = None


class SPARL:
    """Interface to the SPARL representation learning model.

    SPARL learns spatial-aware representations of protein expression
    data from spatial proteomics technologies.

    Parameters
    ----------
    adata
        AnnData object containing spatial proteomics data. The main
        expression matrix (X or a layer) should contain protein
        measurements. Spatial coordinates should be in obsm.
    spatial_key
        Key in ``adata.obsm`` for spatial coordinates.
    layer
        Layer in ``adata.layers`` to use. If None, uses ``adata.X``.
    n_latent
        Dimensionality of the latent space.
    **model_params
        Additional parameters passed to the SPARL model.

    Notes
    -----
    SPARL is distributed separately from spatialvi-tools. To use this
    wrapper you must install the ``sparl`` package.

    Examples
    --------
    >>> from spatialvi.external import SPARL
    >>> # Load CODEX data
    >>> adata = sc.read("codex_data.h5ad")
    >>> model = SPARL(adata, n_latent=20)
    >>> model.train(max_epochs=100)
    >>> # Get latent representation
    >>> adata.obsm["X_sparl"] = model.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        spatial_key: str = "spatial",
        layer: str | None = None,
        n_latent: int = 32,
        **model_params: Any,
    ) -> None:
        self.adata = adata
        self.spatial_key = spatial_key
        self.layer = layer
        self.n_latent = n_latent
        self.model_params = model_params
        self._model: Any = None
        self._is_trained = False

    @staticmethod
    def _require_sparl() -> None:
        if sparl is None:
            raise ImportError("The 'sparl' package is not installed. Install it with ``pip install sparl``.")

    def train(
        self,
        max_epochs: int = 100,
        batch_size: int = 128,
        lr: float = 1e-3,
        **train_kwargs: Any,
    ) -> None:
        """Train the SPARL model.

        Parameters
        ----------
        max_epochs
            Maximum number of training epochs.
        batch_size
            Batch size for training.
        lr
            Learning rate.
        **train_kwargs
            Additional keyword arguments for training.

        Notes
        -----
        Currently this wrapper provides a placeholder implementation.
        Full training functionality will be added when the SPARL API
        stabilizes.
        """
        self._require_sparl()

        # Get expression data
        if self.layer is not None:
            X = self.adata.layers[self.layer]
        else:
            X = self.adata.X

        if hasattr(X, "toarray"):
            X = X.toarray()

        # Get spatial coordinates
        coords = self.adata.obsm[self.spatial_key]
        if hasattr(coords, "values"):
            coords = coords.values

        # Initialize and train SPARL model
        self._model = sparl.SPARL(
            n_input=X.shape[1],
            n_latent=self.n_latent,
            **self.model_params,
        )

        self._model.fit(
            X,
            coords,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            **train_kwargs,
        )

        self._is_trained = True
        logger.info("SPARL training complete.")

    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Get latent representations for cells.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the training data.
        batch_size
            Batch size for inference.

        Returns
        -------
        Latent representation array of shape (n_cells, n_latent).
        """
        self._require_sparl()
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call `.train()` first.")

        if adata is None:
            adata = self.adata

        # Get expression data
        if self.layer is not None:
            X = adata.layers[self.layer]
        else:
            X = adata.X

        if hasattr(X, "toarray"):
            X = X.toarray()

        # Get spatial coordinates
        coords = adata.obsm[self.spatial_key]
        if hasattr(coords, "values"):
            coords = coords.values

        return self._model.transform(X, coords, batch_size=batch_size)

    def predict(
        self,
        adata: AnnData | None = None,
        store_key: str = "X_sparl",
    ) -> AnnData:
        """Compute and store latent representations.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the training data.
        store_key
            Key to store latent representation in obsm.

        Returns
        -------
        AnnData with latent representation stored in obsm.
        """
        if adata is None:
            adata = self.adata

        adata.obsm[store_key] = self.get_latent_representation(adata)
        return adata

    def reconstruct(
        self,
        adata: AnnData | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Reconstruct protein expression from latent space.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the training data.
        batch_size
            Batch size for inference.

        Returns
        -------
        Reconstructed expression array of shape (n_cells, n_proteins).
        """
        self._require_sparl()
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call `.train()` first.")

        if adata is None:
            adata = self.adata

        latent = self.get_latent_representation(adata, batch_size=batch_size)
        return self._model.decode(latent, batch_size=batch_size)

    def impute_proteins(
        self,
        adata: AnnData | None = None,
        store_layer: str = "sparl_imputed",
    ) -> AnnData:
        """Impute missing protein values using the trained model.

        Parameters
        ----------
        adata
            AnnData object. If None, uses the training data.
        store_layer
            Key to store imputed expression in layers.

        Returns
        -------
        AnnData with imputed expression stored in layers.
        """
        if adata is None:
            adata = self.adata

        adata.layers[store_layer] = self.reconstruct(adata)
        return adata
