"""Tests for module-level utility functions in _scviva.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from spatialvi.model._scviva import get_niche_indexes


def _make_adata(n_cells=20, n_genes=10, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n_cells, 2))
    obs = pd.DataFrame(
        {"sample": ["s1"] * 10 + ["s2"] * 10},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    adata = AnnData(
        X=rng.poisson(5, size=(n_cells, n_genes)).astype(float),
        obs=obs,
    )
    adata.obsm["spatial"] = coords
    return adata


def test_get_niche_indexes_shapes():
    """Index and distance arrays must match (n_cells, k_nn)."""
    adata = _make_adata()
    k_nn = 5
    get_niche_indexes(
        adata,
        sample_key="sample",
        niche_indexes_key="niche_idx",
        cell_coordinates_key="spatial",
        k_nn=k_nn,
        niche_distances_key="niche_dist",
    )
    assert adata.obsm["niche_idx"].shape == (20, k_nn)
    assert adata.obsm["niche_dist"].shape == (20, k_nn)


def test_get_niche_indexes_no_self_loops():
    """No cell should appear as its own neighbor."""
    adata = _make_adata()
    get_niche_indexes(
        adata,
        sample_key="sample",
        niche_indexes_key="niche_idx",
        cell_coordinates_key="spatial",
        k_nn=4,
        niche_distances_key="niche_dist",
    )
    idx = adata.obsm["niche_idx"]
    for i, row in enumerate(idx):
        assert i not in row, f"cell {i} listed as its own neighbor"


def test_get_niche_indexes_neighbors_within_sample():
    """All returned neighbor indices must belong to the same sample."""
    adata = _make_adata()
    get_niche_indexes(
        adata,
        sample_key="sample",
        niche_indexes_key="niche_idx",
        cell_coordinates_key="spatial",
        k_nn=4,
        niche_distances_key="niche_dist",
    )
    idx = adata.obsm["niche_idx"].astype(int)
    samples = adata.obs["sample"].values
    for i in range(len(adata)):
        for neighbor in idx[i]:
            assert samples[neighbor] == samples[i], (
                f"cell {i} (sample {samples[i]}) has cross-sample neighbor "
                f"{neighbor} (sample {samples[neighbor]})"
            )


def test_get_niche_indexes_int_dtype():
    """Index array must be integer-typed."""
    adata = _make_adata()
    get_niche_indexes(
        adata,
        sample_key="sample",
        niche_indexes_key="niche_idx",
        cell_coordinates_key="spatial",
        k_nn=3,
        niche_distances_key="niche_dist",
    )
    assert np.issubdtype(adata.obsm["niche_idx"].dtype, np.integer)


def test_get_niche_indexes_distances_nonnegative():
    """All distances must be >= 0."""
    adata = _make_adata()
    get_niche_indexes(
        adata,
        sample_key="sample",
        niche_indexes_key="niche_idx",
        cell_coordinates_key="spatial",
        k_nn=3,
        niche_distances_key="niche_dist",
    )
    assert (adata.obsm["niche_dist"] >= 0).all()
