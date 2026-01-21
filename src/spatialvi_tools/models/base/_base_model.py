"""Abstract base classes for spatialvi‑tools models.

This module defines simple base classes that specify the public API for
all models in the library.  Inspired by the design of scvi‑tools,
models derived from :class:`BaseSpatialModel` accept an
``AnnData`` object, can be trained via :meth:`train` and produce
inference results via :meth:`predict` or :meth:`infer`.
"""

from __future__ import annotations

import abc
import anndata as ad


class BaseSpatialModel(abc.ABC):
    """Abstract base class defining the common model interface.

    Subclasses should implement the :meth:`train` method to fit model
    parameters and the :meth:`predict` (or :meth:`infer`) method to
    generate outputs for a new dataset.
    """

    def __init__(self, adata: ad.AnnData) -> None:
        self.adata = adata

    @abc.abstractmethod
    def train(self, *args, **kwargs) -> None:
        """Train the model on the associated AnnData.

        Parameters
        ----------
        *args, **kwargs:
            Additional model‑specific training parameters.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, adata: ad.AnnData | None = None, *args, **kwargs):
        """Generate predictions or latent representations for the given AnnData.

        Parameters
        ----------
        adata:
            The AnnData to run prediction on.  If ``None``, use the
            AnnData provided at initialization.

        Returns
        -------
        object
            Model‑specific predictions.  For example, an array of embeddings
            or a dataframe of annotations.
        """
        raise NotImplementedError