from __future__ import annotations

import numpy as np
import torch
from scvi.data import AnnTorchDataset
from scvi.dataloaders import DataSplitter
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_convert

from scviva._constants import (
    SCVIVA_CONTIGUITY_EDGE_INDEX_KEY as CONTIGUITY_EDGE_INDEX_KEY,
)
from scviva._constants import (
    SCVIVA_CONTIGUITY_REPLACEMENT_KEY as CONTIGUITY_REPLACEMENT_KEY,
)
from scviva._constants import (
    SCVIVA_SEED_COUNT_KEY as SEED_COUNT_KEY,
)


def build_same_label_edges(niche_indexes: np.ndarray, labels: np.ndarray) -> torch.Tensor:
    """Build canonical unique same-label edges from registered niche indices."""
    neighbors = np.asarray(niche_indexes)
    label_codes = np.asarray(labels).reshape(-1)
    if neighbors.ndim != 2:
        raise ValueError("Registered niche indexes must be two-dimensional.")
    if neighbors.shape[0] != label_codes.shape[0]:
        raise ValueError("Niche indexes and labels must contain the same number of cells.")
    if not np.issubdtype(neighbors.dtype, np.integer):
        raise TypeError("Registered niche indexes must use an integer dtype.")

    n_obs, n_neighbors = neighbors.shape
    source = np.repeat(np.arange(n_obs, dtype=np.int64), n_neighbors)
    target = neighbors.reshape(-1).astype(np.int64, copy=False)
    valid = (target >= 0) & (target < n_obs) & (source != target)
    source = source[valid]
    target = target[valid]
    same_label = label_codes[source] == label_codes[target]
    source = source[same_label]
    target = target[same_label]
    low = np.minimum(source, target)
    high = np.maximum(source, target)
    if low.size == 0:
        return torch.empty((2, 0), dtype=torch.long)
    unique = np.unique(np.column_stack([low, high]), axis=0)
    return torch.as_tensor(unique.T.copy(), dtype=torch.long)


def filter_edges_to_indices(
    edge_index: torch.Tensor, indices: np.ndarray, n_obs: int
) -> torch.Tensor:
    """Keep edges whose endpoints both belong to ``indices``."""
    allowed = torch.zeros(n_obs, dtype=torch.bool)
    allowed[torch.as_tensor(indices, dtype=torch.long)] = True
    mask = allowed[edge_index[0]] & allowed[edge_index[1]]
    return edge_index[:, mask]


def sample_edge_columns(
    edge_index: torch.Tensor, budget: int, generator: torch.Generator
) -> tuple[torch.Tensor, bool]:
    """Sample edge columns, using replacement only for undersized edge sets."""
    if budget <= 0:
        raise ValueError("Contiguity edge budget must be positive.")
    n_edges = edge_index.shape[1]
    if n_edges == 0:
        raise RuntimeError("No eligible same-cell-type spatial edges are available.")
    replacement = n_edges < budget
    selected = (
        torch.randint(n_edges, (budget,), generator=generator)
        if replacement
        else torch.randperm(n_edges, generator=generator)[:budget]
    )
    return edge_index[:, selected], replacement


def stable_unique_seed_first(seed_ids: np.ndarray, context_ids: np.ndarray) -> np.ndarray:
    """Return stable unique global IDs with all seed IDs first."""
    seen: set[int] = set()
    ordered: list[int] = []
    for value in np.concatenate([seed_ids, context_ids]).tolist():
        value = int(value)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return np.asarray(ordered, dtype=np.int64)


def global_edges_to_local(global_edges: torch.Tensor, global_ids: np.ndarray) -> torch.Tensor:
    """Map global edge endpoints to positions in a local node list."""
    mapping = {int(node): local for local, node in enumerate(global_ids.tolist())}
    try:
        local = [[mapping[int(node)] for node in row.tolist()] for row in global_edges]
    except KeyError as error:
        raise IndexError("Contiguity edge endpoint is absent from the local batch.") from error
    return torch.as_tensor(local, dtype=torch.long)


class _RegisteredTensorFetcher:
    """Fetch registered AnnData tensors for arbitrary global observation IDs."""

    def __init__(self, adata_manager):
        self.dataset = AnnTorchDataset(adata_manager, load_sparse_tensor=False)

    def __call__(self, global_ids: np.ndarray) -> dict[str, torch.Tensor]:
        raw = self.dataset[np.asarray(global_ids, dtype=np.int64)]
        return {key: default_convert(value) for key, value in raw.items()}


class ContiguityDataLoader(DataLoader):
    """Augment ordinary seed batches with independently sampled edge endpoints."""

    def __init__(
        self,
        adata_manager,
        indices: np.ndarray,
        eligible_edges: torch.Tensor,
        batch_size: int,
        edge_budget: int,
        shuffle: bool,
        seed: int,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.eligible_edges = eligible_edges.cpu()
        self.edge_budget = int(edge_budget)
        self.fetch = _RegisteredTensorFetcher(adata_manager)
        self.edge_generator = torch.Generator().manual_seed(int(seed) + 991)
        self.replacement_batches = 0
        super().__init__(
            torch.as_tensor(self.indices, dtype=torch.long),
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            drop_last=False,
            num_workers=0,
            generator=torch.Generator().manual_seed(int(seed)),
            collate_fn=self._collate,
        )

    def _collate(self, sampled_ids) -> dict[str, torch.Tensor]:
        seeds = torch.stack(list(sampled_ids)).long().reshape(-1).numpy()
        sampled_edges, replacement = sample_edge_columns(
            self.eligible_edges, self.edge_budget, self.edge_generator
        )
        global_ids = stable_unique_seed_first(seeds, sampled_edges.reshape(-1).numpy())
        batch = self.fetch(global_ids)
        batch[SEED_COUNT_KEY] = torch.tensor(len(seeds), dtype=torch.long)
        batch[CONTIGUITY_EDGE_INDEX_KEY] = global_edges_to_local(sampled_edges, global_ids)
        batch[CONTIGUITY_REPLACEMENT_KEY] = torch.tensor(replacement)
        self.replacement_batches += int(replacement)
        return batch


class ContiguityDataSplitter(DataSplitter):
    """Create split-local contiguity-aware loaders from one scvi data split."""

    def __init__(
        self,
        adata_manager,
        *,
        eligible_edges: torch.Tensor,
        loader_seed: int,
        edge_budget: int,
        **kwargs,
    ):
        super().__init__(adata_manager, **kwargs)
        self.eligible_edges = eligible_edges.cpu()
        self.loader_seed = int(loader_seed)
        self.edge_budget = int(edge_budget)
        self.n_obs = adata_manager.adata.n_obs

    def setup(self, stage: str | None = None):
        """Create the split once and filter eligible edges to each subset."""
        super().setup(stage)
        self.train_edges = filter_edges_to_indices(self.eligible_edges, self.train_idx, self.n_obs)
        self.val_edges = filter_edges_to_indices(self.eligible_edges, self.val_idx, self.n_obs)
        self.test_edges = filter_edges_to_indices(self.eligible_edges, self.test_idx, self.n_obs)

    def _loader(self, indices, edges, shuffle: bool, seed_offset: int):
        if len(indices) and edges.shape[1] == 0:
            split_name = {0: "training", 1: "validation", 2: "test"}[seed_offset]
            raise RuntimeError(
                f"The {split_name} split has no eligible same-cell-type spatial edges."
            )
        return ContiguityDataLoader(
            self.adata_manager,
            indices=indices,
            eligible_edges=edges,
            batch_size=int(self.data_loader_kwargs.get("batch_size", 128)),
            edge_budget=self.edge_budget,
            shuffle=shuffle,
            seed=self.loader_seed + seed_offset,
        )

    def train_dataloader(self):
        """Return a shuffled contiguity-aware training loader."""
        return self._loader(self.train_idx, self.train_edges, True, 0)

    def calibration_dataloader(self):
        """Return a deterministic training-only loader for auto-calibration."""
        return self._loader(self.train_idx, self.train_edges, False, 0)

    def val_dataloader(self):
        """Return a deterministic validation-only loader when validation exists."""
        if len(self.val_idx) == 0:
            return None
        return self._loader(self.val_idx, self.val_edges, False, 1)

    def test_dataloader(self):
        """Return a deterministic test-only loader when a test split exists."""
        if len(self.test_idx) == 0:
            return None
        return self._loader(self.test_idx, self.test_edges, False, 2)
