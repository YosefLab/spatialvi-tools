"""Wrapper for the VIVS variable selection model.

VIVS (Variational Inference for Variable Selection) identifies key
molecular features driving a response of interest in omics data【666593781079131†L1-L8】.  This
wrapper exposes a scvi‑tools–style interface built on top of the
``vivs`` package.  The underlying model is trained via variational
inference and uses a hierarchical clustering of genes to compute
importance scores.
"""

from __future__ import annotations

from typing import Any, Iterable, List

import anndata as ad

from ..mixin import AnnDataMixin
from .base._base_model import BaseSpatialModel

try:  # optional dependency
    from vivs import VIVS  # type: ignore
except ImportError:  # pragma: no cover
    VIVS = None  # type: ignore


class VIVSModel(AnnDataMixin, BaseSpatialModel):
    """High‑level interface to the VIVS variable selection algorithm.

    Parameters
    ----------
    adata:
        AnnData object containing raw gene counts (not normalized)【666593781079131†L28-L33】.
        Response variables (e.g. phenotypes or metadata) should be stored in
        ``adata.obsm`` under the key given by ``feature_obsm_key``【666593781079131†L33-L34】.
    feature_obsm_key:
        Name of the entry in ``adata.obsm`` that stores the response(s).
    xy_linear:
        Whether to use a linear model for the effect of predictors on
        responses.  Defaults to ``False`` (nonlinear model).
    xy_model_kwargs:
        Additional keyword arguments passed to the VIVS model constructor
        for configuring the neural network.  See VIVS documentation for
        details【666593781079131†L47-L52】.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        feature_obsm_key: str,
        xy_linear: bool = False,
        xy_model_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        AnnDataMixin.__init__(self, adata)
        BaseSpatialModel.__init__(self, adata)
        self.feature_obsm_key = feature_obsm_key
        self.xy_linear = xy_linear
        self.xy_model_kwargs = xy_model_kwargs or {}
        self.kwargs = kwargs
        self._model: Any = None

    def _require_vivs(self) -> None:
        if VIVS is None:
            raise ImportError(
                "The 'vivs' package is not installed. Install it with ``pip install vivs``."
            )

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Fit the VIVS model to the data.

        This method constructs the underlying ``vivs.VIVS`` object and
        trains it on the supplied AnnData.  Any additional keyword
        arguments are forwarded to ``VIVS``.
        """
        self._require_vivs()
        # Instantiate VIVS model
        self._model = VIVS(
            self.adata,
            feature_obsm_key=self.feature_obsm_key,
            xy_linear=self.xy_linear,
            xy_model_kwargs=self.xy_model_kwargs,
            **self.kwargs,
        )
        # Train the model using the default train_all routine【666593781079131†L47-L52】
        self._model.train_all()

    def predict(
        self,
        adata: Optional[ad.AnnData] = None,
        n_clusters_list: Iterable[int] = (50,),
        store_key_prefix: str = "vivs",
        **kwargs: Any,
    ) -> Any:
        """Compute hierarchical importance scores using the trained model.

        Parameters
        ----------
        adata:
            Not used (VIVS operates on the training data).  Included for
            compatibility with the base interface.
        n_clusters_list:
            List of cluster numbers over which to compute hierarchical
            importance scores【666593781079131†L63-L67】.
        store_key_prefix:
            Prefix under which to store the resulting importance tables in
            ``adata.uns``.
        **kwargs:
            Additional arguments forwarded to
            ``VIVS.get_hier_importance``.

        Returns
        -------
        result:
            The nested dictionary of importance scores returned by
            ``VIVS.get_hier_importance``.
        """
        self._require_vivs()
        if self._model is None:
            raise RuntimeError("Model has not been trained. Call `.train()` first.")
        n_clusters = list(n_clusters_list)
        result = self._model.get_hier_importance(n_clusters_list=n_clusters, **kwargs)
        # store result in adata.uns for reference
        self.adata.uns[f"{store_key_prefix}_hier_importance"] = result
        return result