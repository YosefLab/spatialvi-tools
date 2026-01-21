"""Wrapper for the Harreman metabolic exchange analysis.

Harreman is an algorithm for inferring metabolic exchanges in tissues
using spatial transcriptomics data【504187329151369†L11-L13】.  It computes a variety
of spatial correlation statistics to identify metabolic zones, infer
metabolite exchange and gene–metabolite interactions【504187329151369†L31-L42】.  The
present wrapper provides a common interface for invoking Harreman
statistics.  At this stage only a stub implementation is provided: the
methods raise :class:`NotImplementedError` unless the ``harreman``
package is installed and a future version of this wrapper is updated.
"""

from __future__ import annotations

from typing import Any

import anndata as ad

from ..mixin import AnnDataMixin
from .base._base_model import BaseSpatialModel

try:
    import harreman  # type: ignore
except ImportError:  # pragma: no cover
    harreman = None  # type: ignore


class HarremanModel(AnnDataMixin, BaseSpatialModel):
    """Interface to the Harreman metabolic exchange pipeline.

    Parameters
    ----------
    adata:
        AnnData containing spatial transcriptomics data.
    """

    def __init__(self, adata: ad.AnnData, **kwargs: Any) -> None:
        AnnDataMixin.__init__(self, adata)
        BaseSpatialModel.__init__(self, adata)
        self.kwargs = kwargs

    def _require_harreman(self) -> None:
        if harreman is None:
            raise ImportError(
                "The 'harreman' package is not installed. Install it with ``pip install harreman``."
            )

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Harreman does not currently require training.

        The analysis pipeline is run directly via the individual test
        statistic methods implemented on the class.
        """
        # no training routine for Harreman
        return None

    def predict(self, *args: Any, **kwargs: Any):  # pragma: no cover - placeholder
        """Placeholder predict method.

        Harreman is not a predictive model in the usual sense; instead
        call one of the test statistic methods provided on this class.
        """
        raise NotImplementedError(
            "HarremanModel does not implement a general predict method."
        )

    # Placeholder methods for specific statistics; these should be
    # implemented once the harreman API stabilises.

    def test_statistic_1(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        """Is gene a spatially autocorrelated?【504187329151369†L31-L34】"""
        self._require_harreman()
        raise NotImplementedError

    def test_statistic_2(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        """Are genes a and b co‑localized?【504187329151369†L34-L37】"""
        self._require_harreman()
        raise NotImplementedError

    def test_statistic_3(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        """Is metabolite m spatially autocorrelated?【504187329151369†L34-L39】"""
        self._require_harreman()
        raise NotImplementedError

    # ... additional test statistics would follow similarly ...