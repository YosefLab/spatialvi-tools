from __future__ import annotations

import os
import tempfile

import numba
from scvi.external.harreman._analysis import HarremanAnalysis

_NUMBA_CACHE_DIR = os.path.join(tempfile.gettempdir(), "numba_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", _NUMBA_CACHE_DIR)
numba.config.CACHE_DIR = _NUMBA_CACHE_DIR


def test_pl_accessor_methods(adata_spatial):
    ha = HarremanAnalysis(adata_spatial)
    for method in [
        "local_correlation_plot",
        "average_local_correlation_plot",
        "module_score_correlation_plot",
        "plot_interacting_cell_scores",
        "plot_ct_interacting_cell_scores",
        "plot_interaction_module_correlation",
    ]:
        assert callable(getattr(ha.pl, method)), f"ha.pl.{method} not callable"
