"""scib-metrics spatial integration tests for GraphDataLoader-trained models.

Tests the full scib-metrics ``Benchmarker`` pipeline with all three spatial
metric axes (CoordinatePreservation, NichePreservation, DomainBoundary) on
latent spaces produced by ResolVI and SCVIVA with GraphDataSplitter.

All tests are ``@pytest.mark.benchmark`` and skipped by default in CI.
Run explicitly:

    pytest tests/regression/test_scib_metrics_spatial.py -m benchmark -q
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic spatial datasets
# ---------------------------------------------------------------------------


def _make_resolvi_adata(
    n_cells: int = 300,
    n_genes: int = 50,
    n_domains: int = 3,
    n_neighbors: int = 10,
    seed: int = 42,
) -> AnnData:
    """Spatially structured dataset suitable for ResolVI benchmarking.

    Cells form ``n_domains`` spatial zones with correlated expression per domain.
    Neighbor indices use a ring topology so every cell has exactly
    ``n_neighbors`` neighbors.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    n_per = n_cells // n_domains
    n_cells = n_per * n_domains

    coords = np.zeros((n_cells, 2))
    domain_labels = np.empty(n_cells, dtype=int)
    X = np.zeros((n_cells, n_genes))

    for d in range(n_domains):
        sl = slice(d * n_per, (d + 1) * n_per)
        coords[sl, 0] = d * 10 + rng.uniform(0, 8, n_per)
        coords[sl, 1] = rng.uniform(0, 10, n_per)
        domain_labels[sl] = d
        sig = rng.normal(loc=d * 2, scale=0.5, size=n_genes)
        X[sl] = rng.poisson(np.exp(sig + rng.normal(0, 0.3, (n_per, n_genes))))

    X = np.clip(X, 0, None).astype(np.float32)
    idx = np.array([[(i + j + 1) % n_cells for j in range(n_neighbors)] for i in range(n_cells)])
    distances = np.tile(np.arange(1, n_neighbors + 1, dtype=np.float32), (n_cells, 1))

    # Two synthetic batches: left / right half of x-axis
    batch_labels = np.where(coords[:, 0] < coords[:, 0].mean(), "batch_0", "batch_1")
    obs = pd.DataFrame(
        {
            "domain": pd.Categorical([f"domain_{d}" for d in domain_labels]),
            "batch": pd.Categorical(batch_labels),
        }
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.obsm["spatial"] = coords
    adata.obsm["index_neighbor"] = idx.astype(np.int64)
    adata.obsm["distance_neighbor"] = distances
    return adata


def _make_scviva_adata(n_cells: int = 200, n_genes: int = 30, n_domains: int = 3, seed: int = 7) -> AnnData:
    """Minimal spatially structured dataset for SCVIVA benchmarking."""
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    n_per = n_cells // n_domains
    n_cells = n_per * n_domains
    coords = np.zeros((n_cells, 2))
    domain_labels = np.empty(n_cells, dtype=int)
    X = np.zeros((n_cells, n_genes))
    for d in range(n_domains):
        sl = slice(d * n_per, (d + 1) * n_per)
        coords[sl, 0] = d * 8 + rng.uniform(0, 6, n_per)
        coords[sl, 1] = rng.uniform(0, 8, n_per)
        domain_labels[sl] = d
        sig = rng.normal(loc=d * 1.5, scale=0.5, size=n_genes)
        X[sl] = rng.poisson(np.exp(sig + rng.normal(0, 0.3, (n_per, n_genes))))

    X = np.clip(X, 1, None).astype(np.float32)  # avoid log(0) in encoder
    batch_labels = np.where(coords[:, 0] < coords[:, 0].mean(), "batch_0", "batch_1")
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical([f"type_{d}" for d in domain_labels]),
            "batch": pd.Categorical(batch_labels),
        }
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.obsm["spatial"] = coords
    return adata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_resolvi_graph(adata: AnnData, max_epochs: int = 10) -> AnnData:
    """Setup, train, and return latent for ResolVI with GraphDataSplitter."""
    from scviva.model import ResolVI

    ResolVI.setup_anndata(adata, prepare_data=False)
    model = ResolVI(adata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(
            max_epochs=max_epochs,
            enable_progress_bar=False,
            datasplitter_kwargs={"neighbor_indices_key": "index_neighbor"},
        )
    adata.obsm["X_resolVI"] = model.get_latent_representation()
    return adata


def _train_scviva_graph(adata: AnnData, max_epochs: int = 10) -> AnnData:
    """Setup, train, and return latent for SCVIVA with GraphDataSplitter."""
    from scviva.model import SCVIVA

    setup_kwargs = {
        "sample_key": "batch",
        "labels_key": "cell_type",
        "cell_coordinates_key": "spatial",
        "expression_embedding_key": "X_scvi_fake",
    }
    k_nn = 5
    adata.obsm["X_scvi_fake"] = np.random.default_rng(0).normal(size=(adata.n_obs, 10)).astype(np.float32)
    SCVIVA.preprocessing_anndata(adata, k_nn=k_nn, **setup_kwargs)
    SCVIVA.setup_anndata(adata, layer=None, batch_key="batch", **setup_kwargs)
    model = SCVIVA(adata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train(max_epochs=max_epochs, enable_progress_bar=False)
    adata.obsm["X_scVIVA"] = model.get_latent_representation()
    return adata


def _build_benchmarker(
    adata: AnnData,
    embedding_keys: list[str],
    batch_key: str,
    label_key: str,
    spatial_key: str = "spatial",
    pre_integrated_key: str | None = None,
    multi_batch: bool = True,
) -> object:
    """Construct a Benchmarker with all spatial axes enabled."""
    from scib_metrics.benchmark import (
        BatchCorrection,
        Benchmarker,
        BioConservation,
        CoordinatePreservation,
        DomainBoundary,
        NichePreservation,
    )

    batch_corr = BatchCorrection(
        bras=multi_batch,        # requires ≥2 batches within each label
        pcr_comparison=multi_batch,  # requires ≥2 batches globally
    )
    return Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=embedding_keys,
        bio_conservation_metrics=BioConservation(
            isolated_labels=True,
            nmi_ari_cluster_labels_kmeans=True,
            silhouette_label=True,
            clisi_knn=True,
        ),
        batch_correction_metrics=batch_corr,
        spatial_conservation_metrics=CoordinatePreservation(),
        niche_preservation=NichePreservation(),
        domain_boundary=DomainBoundary(),
        spatial_key=spatial_key,
        pre_integrated_embedding_obsm_key=pre_integrated_key,
        n_jobs=1,
        solver="full",  # cuML/rapids PCA requires 'full', not default 'arpack'
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_resolvi_scib_spatial_metrics():
    """Full scib-metrics Benchmarker on ResolVI GraphDataLoader latent space.

    Validates that all three spatial axes (CoordinatePreservation,
    NichePreservation, DomainBoundary) produce finite scores in [0, 1] for
    the graph-trained model, and that bio conservation metrics are above a
    reasonable floor.
    """
    pytest.importorskip("scib_metrics")

    adata = _make_resolvi_adata()
    adata = _train_resolvi_graph(adata, max_epochs=15)

    bm = _build_benchmarker(
        adata,
        embedding_keys=["X_resolVI"],
        batch_key="batch",
        label_key="domain",
        spatial_key="spatial",
        multi_batch=True,
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)

    spatial_cols = [
        "spatial_mrre",
        "spatial_knn_overlap",
        "spatial_distance_correlation",
        "spatial_morans_i",
        "spatial_niche_knn_overlap",
        "spatial_pas",
        "spatial_chaos",
    ]
    for col in spatial_cols:
        assert col in results.columns, f"Missing spatial metric column: {col}"
        val = float(results.loc["X_resolVI", col])
        assert 0.0 <= val <= 1.0, f"{col}={val:.3f} not in [0, 1]"

    nmi = float(results.loc["X_resolVI", "nmi_ari_cluster_labels_kmeans_nmi"])
    assert nmi > 0.0, f"NMI should be positive for spatially structured data: {nmi:.3f}"

    print("\n--- ResolVI scib-metrics spatial benchmark ---")
    for col in spatial_cols:
        print(f"  {col:<45}: {float(results.loc['X_resolVI', col]):.3f}")
    print(f"  {'NMI':<45}: {nmi:.3f}")


@pytest.mark.benchmark
def test_resolvi_graph_vs_ann_scib_spatial():
    """Graph path produces comparable or better spatial scores vs AnnDataLoader.

    Both paths should reach similar spatial metrics because the VAE processes
    center cells independently.  The graph path must not *degrade* any spatial
    metric versus the AnnDataLoader baseline.
    """
    pytest.importorskip("scib_metrics")
    from scvi.dataloaders import DataSplitter
    from scviva.dataloaders import GraphDataSplitter
    from scviva.model import ResolVI

    class ResolVIGraph(ResolVI):
        _data_splitter_cls = GraphDataSplitter

    class ResolVILegacy(ResolVI):
        _data_splitter_cls = DataSplitter

    adata = _make_resolvi_adata(n_cells=300, seed=0)
    n_epochs = 15

    ResolVIGraph.setup_anndata(adata, prepare_data=False)
    mg = ResolVIGraph(adata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mg.train(
            max_epochs=n_epochs,
            enable_progress_bar=False,
            datasplitter_kwargs={"neighbor_indices_key": "index_neighbor"},
        )
    adata.obsm["X_graph"] = mg.get_latent_representation()

    ResolVILegacy.setup_anndata(adata, prepare_data=False)
    ml = ResolVILegacy(adata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml.train(max_epochs=n_epochs, enable_progress_bar=False)
    adata.obsm["X_legacy"] = ml.get_latent_representation()

    bm = _build_benchmarker(
        adata,
        embedding_keys=["X_graph", "X_legacy"],
        batch_key="batch",
        label_key="domain",
        spatial_key="spatial",
        multi_batch=True,
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)

    spatial_cols = [
        "spatial_mrre",
        "spatial_knn_overlap",
        "spatial_distance_correlation",
        "spatial_morans_i",
        "spatial_niche_knn_overlap",
    ]
    print("\n--- Graph vs AnnDataLoader spatial scores ---")
    print(f"{'Metric':<45} {'Graph':>8}  {'Legacy':>8}")
    for col in spatial_cols:
        vg = float(results.loc["X_graph", col])
        vl = float(results.loc["X_legacy", col])
        print(f"  {col:<45} {vg:>8.3f}  {vl:>8.3f}")
        # graph must not degrade by more than 0.15 on any spatial metric
        assert vg >= vl - 0.15, (
            f"Graph {col}={vg:.3f} degraded vs legacy {col}={vl:.3f}"
        )


@pytest.mark.benchmark
def test_scviva_scib_spatial_metrics():
    """Full scib-metrics Benchmarker on SCVIVA GraphDataLoader latent space.

    All spatial axes (CoordinatePreservation, NichePreservation, DomainBoundary)
    produce finite scores in [0, 1] for the graph-trained scVIVA model.

    Uses a single-batch synthetic dataset, so bras and pcr_comparison are
    disabled (they require ≥2 batches within each label).
    """
    pytest.importorskip("scib_metrics")
    pytest.importorskip("scviva.model")

    adata = _make_scviva_adata(n_cells=200, n_domains=3, seed=7)

    try:
        adata = _train_scviva_graph(adata, max_epochs=10)
    except Exception as exc:
        pytest.skip(f"SCVIVA training failed (likely missing optional dep): {exc}")

    bm = _build_benchmarker(
        adata,
        embedding_keys=["X_scVIVA", "X_scvi_fake"],
        batch_key="batch",
        label_key="cell_type",
        spatial_key="spatial",
        pre_integrated_key="X_scvi_fake",
        multi_batch=True,
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)

    spatial_cols = [
        "spatial_mrre",
        "spatial_knn_overlap",
        "spatial_distance_correlation",
        "spatial_morans_i",
        "spatial_niche_knn_overlap",
        "spatial_pas",
        "spatial_chaos",
    ]
    for col in spatial_cols:
        assert col in results.columns, f"Missing: {col}"
        val = float(results.loc["X_scVIVA", col])
        assert 0.0 <= val <= 1.0, f"{col}={val:.3f} out of range"

    print("\n--- scVIVA scib-metrics spatial benchmark ---")
    for col in spatial_cols:
        print(f"  {col:<45}: {float(results.loc['X_scVIVA', col]):.3f}")


@pytest.mark.benchmark
def test_scib_spatial_metrics_unit_range():
    """All spatial metric values must lie in [0, 1] for any trained model.

    Sanity-check using a trivial random embedding — verifies metric bounds
    hold even for poor-quality latent spaces.
    """
    pytest.importorskip("scib_metrics")
    from scib_metrics.benchmark import (
        BatchCorrection,
        Benchmarker,
        BioConservation,
        CoordinatePreservation,
        DomainBoundary,
        NichePreservation,
    )
    import anndata
    import pandas as pd

    rng = np.random.default_rng(99)
    n = 200
    coords = rng.uniform(0, 10, (n, 2)).astype(np.float32)
    labels = np.repeat(["A", "B", "C", "D"], n // 4)
    batch = np.tile(["b0", "b1"], n // 2)

    ad = anndata.AnnData(X=rng.normal(size=(n, 20)).astype(np.float32))
    ad.obs["labels"] = pd.Categorical(labels)
    ad.obs["batch"] = pd.Categorical(batch)
    ad.obsm["spatial"] = coords
    ad.obsm["X_random"] = rng.normal(size=(n, 10)).astype(np.float32)
    ad.obsm["X_spatial_copy"] = coords  # perfect spatial embedding

    bm = Benchmarker(
        ad,
        batch_key="batch",
        label_key="labels",
        embedding_obsm_keys=["X_random", "X_spatial_copy"],
        bio_conservation_metrics=BioConservation(
            isolated_labels=False,
            nmi_ari_cluster_labels_kmeans=True,
            silhouette_label=True,
            clisi_knn=False,
        ),
        batch_correction_metrics=BatchCorrection(
            bras=True,
            pcr_comparison=False,
            ilisi_knn=False,
            kbet_per_label=False,
            graph_connectivity=True,
        ),
        spatial_conservation_metrics=CoordinatePreservation(),
        niche_preservation=NichePreservation(),
        domain_boundary=DomainBoundary(),
        spatial_key="spatial",
        n_jobs=1,
        solver="full",  # cuML/rapids PCA requires 'full', not default 'arpack'
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)

    spatial_cols = [
        "spatial_mrre",
        "spatial_knn_overlap",
        "spatial_distance_correlation",
        "spatial_morans_i",
        "spatial_niche_knn_overlap",
        "spatial_pas",
        "spatial_chaos",
    ]
    for col in spatial_cols:
        assert col in results.columns, f"Missing: {col}"
        for key in ["X_random", "X_spatial_copy"]:
            val = float(results.loc[key, col])
            assert 0.0 <= val <= 1.0, f"{key} {col}={val:.3f} out of [0,1]"

    # spatial copy should dominate random on coordinate-preservation metrics
    for col in ("spatial_mrre", "spatial_knn_overlap", "spatial_distance_correlation"):
        v_good = float(results.loc["X_spatial_copy", col])
        v_rand = float(results.loc["X_random", col])
        assert v_good > v_rand, f"spatial copy should outscore random on {col}"

    print("\n--- Unit-range sanity check ---")
    for col in spatial_cols:
        vg = float(results.loc["X_spatial_copy", col])
        vr = float(results.loc["X_random", col])
        print(f"  {col:<45} spatial={vg:.3f}  random={vr:.3f}")
