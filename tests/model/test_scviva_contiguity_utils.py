from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scvi import REGISTRY_KEYS
from scvi.data import synthetic_iid

from scviva._constants import SCVIVA_REGISTRY_KEYS
from scviva.model._scviva import SCVIVA
from scviva.model.utils._contiguity import (
    CONTIGUITY_EDGE_INDEX_KEY,
    SEED_COUNT_KEY,
    ContiguityDataLoader,
    ContiguityDataSplitter,
    build_same_label_edges,
    filter_edges_to_indices,
    global_edges_to_local,
    sample_edge_columns,
    stable_unique_seed_first,
)


@pytest.fixture
def contiguity_manager():
    """Return a registered manager with guaranteed same-label edges in every split."""
    rng = np.random.default_rng(34)
    adata = synthetic_iid(
        batch_size=16,
        n_genes=20,
        n_batches=2,
        n_labels=2,
        dropout_ratio=0.1,
    )
    adata.obs["labels"] = pd.Categorical(
        ["type_0"] * 8 + ["type_1"] * 8 + ["type_0"] * 8 + ["type_1"] * 8
    )
    adata.obsm["spatial"] = np.vstack(
        [
            rng.normal((0, 0), 0.1, (8, 2)),
            rng.normal((10, 10), 0.1, (8, 2)),
            rng.normal((0, 0), 0.1, (8, 2)),
            rng.normal((10, 10), 0.1, (8, 2)),
        ]
    )
    adata.obsm["X_scVI"] = rng.normal(size=(adata.n_obs, 10))
    raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    adata.layers["counts"] = np.abs(raw).astype(int)
    setup_kwargs = {
        "sample_key": "batch",
        "labels_key": "labels",
        "cell_coordinates_key": "spatial",
        "expression_embedding_key": "X_scVI",
        "expression_embedding_niche_key": "niche_activation",
        "niche_composition_key": "niche_composition",
        "niche_indexes_key": "niche_indexes",
        "niche_distances_key": "niche_distances",
    }
    SCVIVA.preprocessing_anndata(adata, k_nn=3, **setup_kwargs)
    SCVIVA.setup_anndata(adata, layer="counts", batch_key="batch", **setup_kwargs)
    return SCVIVA._get_most_recent_anndata_manager(adata, required=True)


def test_build_same_label_edges_filters_and_canonicalizes():
    """Catch retained self/cross-label/invalid edges or duplicate directions."""
    niche_indexes = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 1, 3],
            [1, 2, 99],
        ]
    )
    labels = np.array([0, 0, 1, 1])

    edges = build_same_label_edges(niche_indexes, labels)

    torch.testing.assert_close(edges, torch.tensor([[0, 2], [1, 3]]))


def test_build_same_label_edges_validates_shape():
    """Catch acceptance of malformed one-dimensional neighbor storage."""
    with pytest.raises(ValueError, match="two-dimensional"):
        build_same_label_edges(np.array([0, 1]), np.array([0, 0]))


def test_filter_edges_to_indices_prevents_split_leakage():
    """Catch edges with either endpoint outside the active data split."""
    edges = torch.tensor([[0, 0, 2, 3], [1, 2, 3, 4]])

    filtered = filter_edges_to_indices(edges, np.array([0, 1, 4]), n_obs=5)

    torch.testing.assert_close(filtered, torch.tensor([[0], [1]]))


def test_sample_edge_columns_uses_replacement_only_when_required():
    """Catch undersized sampling or sampling outside the eligible edge set."""
    edges = torch.tensor([[0, 2], [1, 3]])
    generator = torch.Generator().manual_seed(34)

    sampled, replacement = sample_edge_columns(edges, budget=4, generator=generator)

    assert sampled.shape == (2, 4)
    assert replacement is True
    assert set(map(tuple, sampled.T.tolist())) <= {(0, 1), (2, 3)}


def test_seed_first_order_and_local_mapping():
    """Catch context nodes displacing seeds or incorrect global-to-local edges."""
    global_ids = stable_unique_seed_first(np.array([5, 2]), np.array([8, 5, 9, 8]))
    local_edges = global_edges_to_local(torch.tensor([[8, 5], [9, 8]]), global_ids)

    np.testing.assert_array_equal(global_ids, np.array([5, 2, 8, 9]))
    torch.testing.assert_close(local_edges, torch.tensor([[2, 0], [3, 2]]))


def test_contiguity_loader_keeps_seeds_first_and_edges_local(contiguity_manager):
    """Catch batches that lose seed cardinality or emit global edge indices."""
    edges = build_same_label_edges(
        contiguity_manager.get_from_registry(SCVIVA_REGISTRY_KEYS.NICHE_INDEXES_KEY),
        contiguity_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY),
    )
    loader = ContiguityDataLoader(
        contiguity_manager,
        indices=np.arange(16),
        eligible_edges=filter_edges_to_indices(edges, np.arange(16), 32),
        batch_size=4,
        edge_budget=8,
        shuffle=False,
        seed=34,
    )

    batch = next(iter(loader))

    assert int(batch[SEED_COUNT_KEY]) == 4
    local_edges = batch[CONTIGUITY_EDGE_INDEX_KEY]
    assert local_edges.shape == (2, 8)
    assert int(local_edges.max()) < batch[REGISTRY_KEYS.X_KEY].shape[0]


def test_splitter_edges_never_cross_splits(contiguity_manager):
    """Catch train, validation, or test edges borrowing endpoints from another split."""
    edges = build_same_label_edges(
        contiguity_manager.get_from_registry(SCVIVA_REGISTRY_KEYS.NICHE_INDEXES_KEY),
        contiguity_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY),
    )
    split = [np.arange(16), np.arange(16, 24), np.arange(24, 32)]
    splitter = ContiguityDataSplitter(
        contiguity_manager,
        eligible_edges=edges,
        loader_seed=34,
        edge_budget=8,
        external_indexing=split,
        batch_size=4,
    )
    splitter.setup()

    assert set(splitter.train_edges.flatten().tolist()) <= set(split[0])
    assert set(splitter.val_edges.flatten().tolist()) <= set(split[1])
    assert set(splitter.test_edges.flatten().tolist()) <= set(split[2])
