"""Wrapper for the AMICI cell–cell interaction model.

AMICI (Attention Mechanism Interpretation of Cell‑cell Interactions) uses a
cross‑attention neural network to infer which cell types interact with
each other in spatial transcriptomics data【767658563887085†L7-L10】.  This module
provides a convenience wrapper for integrating AMICI into the
spatialvi‑tools API.  The underlying implementation is provided by the
``amici`` package; users must install that package separately.
"""

from __future__ import annotations

from typing import Any, Optional

import anndata as ad

from ..mixin import AnnDataMixin
from .base._base_model import BaseSpatialModel

try:
    from amici import AMICI  # type: ignore
except ImportError:  # pragma: no cover
    AMICI = None  # type: ignore


class AmiciModel(AnnDataMixin, BaseSpatialModel):
    """High‑level interface to the AMICI cell–cell interaction model.

    Parameters
    ----------
    adata:
        AnnData with spatial coordinates and cell type labels.  Coordinates
        should be in ``obsm[coord_key]`` and labels in ``obs[labels_key]``.
    labels_key:
        Key in ``adata.obs`` indicating the cell type labels.  Passed to
        ``AMICI.setup_anndata``【767658563887085†L31-L37】.
    coord_key:
        Key in ``adata.obsm`` containing spatial coordinates.  Defaults to
        ``"spatial"``.
    model_params:
        Additional keyword arguments forwarded to the ``AMICI`` constructor.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        labels_key: str,
        coord_key: str = "spatial",
        **model_params: Any,
    ) -> None:
        AnnDataMixin.__init__(self, adata)
        BaseSpatialModel.__init__(self, adata)
        self.labels_key = labels_key
        self.coord_key = coord_key
        self.model_params = model_params
        self._model: Any = None

    def _require_amici(self) -> None:
        if AMICI is None:
            raise ImportError(
                "The 'amici-st' package is not installed. Install it with ``pip install amici-st``."
            )

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Train the AMICI model on the provided data.

        This method initialises the underlying ``AMICI`` object, prepares
        the AnnData via ``AMICI.setup_anndata`` and runs its ``train``
        method.  Any additional keyword arguments are passed through to
        ``AMICI(train)``.
        """
        self._require_amici()
        # set up anndata for amici
        AMICI.setup_anndata(
            self.adata,
            labels_key=self.labels_key,
            coord_obsm_key=self.coord_key,
        )
        # instantiate model
        self._model = AMICI(self.adata, **self.model_params)
        # train
        self._model.train(*args, **kwargs)

    def predict(
        self,
        adata: Optional[ad.AnnData] = None,
        batch_size: Optional[int] = None,
        store_key: str = "amici_attention",
        **kwargs: Any,
    ) -> ad.AnnData:
        """Compute attention scores indicating cell–cell interactions.

        Parameters
        ----------
        adata:
            AnnData to run prediction on.  If ``None``, use the training
            AnnData.
        batch_size:
            Unused placeholder (present for API consistency).
        store_key:
            Name under which to store the resulting attention matrix in
            ``adata.obsm``.
        **kwargs:
            Additional keyword arguments passed to ``AMICI.predict``.

        Returns
        -------
        AnnData
            Input AnnData with cell–cell attention matrix stored in
            ``obsm[store_key]``.
        """
        self._require_amici()
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call `.train()` first.")
        if adata is None:
            adata = self.adata
        # run prediction; amici returns a dense matrix of shape (n_cells, n_cells)
        attention = self._model.predict(**kwargs)
        # store as obsp or obsm; we use obsp to reflect pairwise relationships
        try:
            adata.obsp[store_key] = attention
        except Exception:
            # if obsp is not available (older anndata), fall back to obsm
            adata.obsm[store_key] = attention
        return adata