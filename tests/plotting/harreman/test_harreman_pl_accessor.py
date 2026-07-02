from __future__ import annotations

import os
import tempfile

import numba
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scviva.tools.harreman._analysis import HarremanAnalysis

_NUMBA_CACHE_DIR = os.path.join(tempfile.gettempdir(), "numba_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", _NUMBA_CACHE_DIR)
numba.config.CACHE_DIR = _NUMBA_CACHE_DIR


@pytest.fixture
def adata_spatial():
    n_obs, n_vars = 50, 20
    rng = np.random.default_rng(42)
    X = rng.poisson(1.0, size=(n_obs, n_vars)).astype(float)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(np.tile(["TypeA", "TypeB"], n_obs // 2))},
        index=[f"cell{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])
    obsm = {"spatial": rng.random((n_obs, 2)) * 100}
    return AnnData(X=X, obs=obs, var=var, obsm=obsm)


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
