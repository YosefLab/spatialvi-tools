"""Wrapper for the Starfysh model for spatial deconvolution and integration.

Starfysh performs reference‑free deconvolution of spatial transcriptomic
spots, integrates histology images when available, and identifies
spatial hubs with unique cell compositions【835840889957633†L6-L10】.  It can also
perform multi‑sample integration【835840889957633†L6-L9】.  At present this wrapper
provides a placeholder interface; a future implementation could rely
on the ``starfysh`` package once a Python API is stabilised.
"""

from __future__ import annotations

from typing import Any

import anndata as ad

from ..mixin import AnnDataMixin
from .base._base_model import BaseSpatialModel

try:
    import starfysh  # type: ignore
except ImportError:  # pragma: no cover
    starfysh = None  # type: ignore


class StarfyshModel(AnnDataMixin, BaseSpatialModel):
    """Placeholder interface to the Starfysh generative model.

    Parameters
    ----------
    adata:
        Spatial transcriptomics dataset.
    signature_file:
        Optional path to a file containing annotated signature gene sets for
        deconvolution【835840889957633†L37-L40】.
    histology:
        Optional paired histology image (e.g. a path to a TIFF file) for
        histology integration【835840889957633†L6-L10】.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        signature_file: str | None = None,
        histology: str | None = None,
        **model_params: Any,
    ) -> None:
        AnnDataMixin.__init__(self, adata)
        BaseSpatialModel.__init__(self, adata)
        self.signature_file = signature_file
        self.histology = histology
        self.model_params = model_params
        self._model: Any = None

    def _require_starfysh(self) -> None:
        if starfysh is None:
            raise ImportError(
                "The 'starfysh' package is not installed. Install it with ``pip install starfysh``."
            )

    def train(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """Placeholder training routine.

        Currently there is no Python API for Starfysh; therefore this
        method is not implemented.  A future version may call into
        starfysh's generative model to fit parameters on the dataset.
        """
        self._require_starfysh()
        raise NotImplementedError(
            "Training Starfysh models is not yet supported in this wrapper."
        )

    def predict(
        self, adata: ad.AnnData | None = None, *args: Any, **kwargs: Any
    ) -> ad.AnnData:  # pragma: no cover
        """Placeholder prediction routine.

        Returns
        -------
        AnnData
            Unmodified input AnnData.  No deconvolution is currently
            performed.
        """
        self._require_starfysh()
        raise NotImplementedError(
            "Prediction with Starfysh models is not yet supported in this wrapper."
        )