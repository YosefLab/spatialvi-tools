"""Wrapper for the SPARL spatial proteomics model.

SPARL stands for Spatial Proteomics Analysis with Representation Learning【218342748980084†L8-L10】.
The package aims to learn latent representations of spatial proteomic data
which can be used for downstream tasks.  At the time of writing there
is no stable Python API; this module therefore provides a placeholder
interface.  Future versions of spatialvi‑tools may integrate SPARL
models once the upstream implementation becomes available.
"""

from __future__ import annotations

from typing import Any

import anndata as ad

from ..mixin import AnnDataMixin
from .base._base_model import BaseSpatialModel

try:
    import sparl  # type: ignore
except ImportError:  # pragma: no cover
    sparl = None  # type: ignore


class SparlModel(AnnDataMixin, BaseSpatialModel):
    """Placeholder interface to the SPARL representation learning model."""

    def __init__(self, adata: ad.AnnData, **model_params: Any) -> None:
        AnnDataMixin.__init__(self, adata)
        BaseSpatialModel.__init__(self, adata)
        self.model_params = model_params
        self._model: Any = None

    def _require_sparl(self) -> None:
        if sparl is None:
            raise ImportError(
                "The 'SPARL' package is not installed. Install it with ``pip install SPARL``."
            )

    def train(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """Placeholder training method for SPARL.

        Currently this wrapper does not expose training functionality.
        """
        self._require_sparl()
        raise NotImplementedError(
            "Training SPARL models is not yet supported in this wrapper."
        )

    def predict(self, *args: Any, **kwargs: Any):  # pragma: no cover
        """Placeholder predict method.

        SPARL produces latent representations of proteomics data, but
        prediction is not yet implemented here.
        """
        self._require_sparl()
        raise NotImplementedError(
            "Prediction with SPARL models is not yet supported in this wrapper."
        )