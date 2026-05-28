# Graph Dataloader Design

**Date:** 2026-05-05
**Updated for:** `scviva-tools` / `spatialvi-tools2`
**Workspace:** `/Users/orikr/PycharmProjects/spatialvi-tools2`

---

## Problem

ResolVI and related spatial models use fixed neighbor graphs stored in AnnData:

- `adata.obsm["index_neighbor"]`: neighbor observation indices, shape `[N, K]`
- `adata.obsm["distance_neighbor"]`: neighbor distances, shape `[N, K]`
- `SCVIVA` uses model-specific registered equivalents:
  - `adata.obsm["niche_indexes"]`
  - `adata.obsm["niche_distances"]`

The upstream scvi-tools ResolVI implementation fetched neighbor expression inside
`RESOLVAEModel._get_fn_args_from_batch`:

```python
# upstream reference:
# /Users/orikr/PycharmProjects/scvi-tools/src/scvi/external/resolvi/_module.py
x_n = self.expression_anntorchdata[ind_neighbors.cpu().numpy().flatten(), :]["X"]
```

That path is correct but inefficient for graph-oriented spatial training because each batch does
random access into a full `AnnTorchDataset`, then moves neighbor expression to the active device.

## Goal

Port the scvi-tools ResolVI graph dataloader work into `scviva-tools`:

- Add `scviva.dataloaders.GraphDataLoader`.
- Add `scviva.dataloaders.GraphDataSplitter`.
- Make `scviva.model.ResolVI` default to `GraphDataSplitter`.
- Make `scviva.model.SCVIVA` default to `GraphDataSplitter` where graph batches can carry
  precomputed niche graph context without changing module semantics.
- Keep legacy `AnnDataLoader` behavior working through a fallback path.
- Support a guarded model-side dense expression cache so graph batches can omit `x_n`.

The remaining development stage is to evaluate the same graph dataloader pattern for `DestVI`,
`Stereoscope`, and `GIMVI` where their data flow actually uses fixed spatial neighbors.

## Decisions

| Question | Decision |
|---|---|
| Public location | `src/scviva/dataloaders/_graph_dataloader.py`, exported from `scviva.dataloaders` |
| Batch type | `torch_geometric.data.Data` |
| Loader base | Subclass `scvi.dataloaders._ann_dataloader.AnnDataLoader` |
| Splitter base | Subclass `scvi.dataloaders._data_splitting.DataSplitter` |
| Graph shape | One mini-batch graph with center nodes and flattened neighbor-expression rows |
| Edge direction | Center local index to flattened neighbor row |
| Edge attrs | Configurable `edge_obsm_keys`, default `["distance_neighbor"]` |
| Cross-split neighbors | Allowed intentionally, matching existing ResolVI behavior |
| `torch_geometric` | Soft dependency; import only during graph batch conversion |
| Default ResolVI path | `ResolVI._data_splitter_cls = GraphDataSplitter` |
| Default SCVIVA path | `SCVIVA._data_splitter_cls = GraphDataSplitter`; train maps `niche_indexes`/`niche_distances` |
| Cache default | `ResolVI.train(cache_neighbor_expression="auto")` enables cache for graph splitters when size guard passes |

## File Changes

```text
src/scviva/dataloaders/
  _graph_dataloader.py       # GraphDataLoader, GraphDataSplitter, _GraphBatchConverter
  __init__.py                # exports GraphDataLoader, GraphDataSplitter

src/scviva/model/
  _resolvi.py                # default GraphDataSplitter and train cache wiring
  _scviva.py                 # default GraphDataSplitter and niche graph splitter wiring

src/scviva/module/
  _resolvae.py               # batch x_n support, legacy fallback, neighbor expression cache

tests/dataloaders/
  test_graph_dataloader.py   # graph dataloader unit tests

tests/model/
  test_resolvi_graph_dataloader.py  # ResolVI graph path integration and benchmark tests
  test_scviva_graph_dataloader.py   # SCVIVA graph splitter integration tests
```

The scvi-tools reference remains:

```text
/Users/orikr/PycharmProjects/scvi-tools/src/scvi/dataloaders/_graph_dataloader.py
/Users/orikr/PycharmProjects/scvi-tools/src/scvi/external/resolvi/_model.py
/Users/orikr/PycharmProjects/scvi-tools/src/scvi/external/resolvi/_module.py
```

## Data Flow

```text
GraphDataSplitter
  -> GraphDataLoader
       -> AnnTorchDataset[batch_indices]
          returns X, index_neighbor, distance_neighbor, labels, batch, indices, covariates
       -> optional full AnnTorchDataset[neighbor_indices.flatten()]
          returns x_n with shape [N*K, G]
       -> torch_geometric.data.Data(...)
```

Graph batch fields:

| Field | Shape | Meaning |
|---|---:|---|
| `x` | `[N, G]` | center-cell expression |
| `x_n` | `[N*K, G]` | optional neighbor expression |
| `index_neighbor` | `[N, K]` | global neighbor observation indices |
| `distance_neighbor` | `[N, K]` | registered distance field |
| `distances_n` | `[N, K]` | direct model-compatible distance alias |
| `edge_index` | `[2, N*K]` | center-to-neighbor-row edges |
| `edge_attr` | `[N*K, D]` | flattened configured edge attributes |

`edge_index[0]` is `torch.arange(N).repeat_interleave(K)`.
`edge_index[1]` is `torch.arange(N*K)`.

Neighbor `k` for center `i` maps to row `i*K + k` in `x_n`.

## Components

### `GraphDataLoader`

`GraphDataLoader` subclasses `AnnDataLoader` and installs `_GraphBatchConverter` as its
`collate_fn`. It rejects caller-provided `collate_fn` and `iter_ndarray=True` because graph batch
construction owns collation.

Constructor parameters:

```python
GraphDataLoader(
    adata_manager,
    full_adata_manager,
    indices=None,
    neighbor_indices_key="index_neighbor",
    edge_obsm_keys=None,
    load_sparse_neighbor_tensor=True,
    load_neighbor_expression=True,
    **kwargs,
)
```

`load_neighbor_expression=False` omits `x_n` from the `Data` object. This is used by ResolVI when
the model-side dense expression cache is active.

`load_sparse_neighbor_tensor=True` preserves sparse neighbor expression as a sparse torch tensor in
the dataloader. ResolVI densifies sparse `x` and `x_n` inside `_get_fn_args_from_batch`, preserving
legacy module semantics.

### `GraphDataSplitter`

`GraphDataSplitter` subclasses scvi-tools `DataSplitter` and returns `GraphDataLoader` from:

- `train_dataloader`
- `val_dataloader`
- `test_dataloader`

It forwards:

- `neighbor_indices_key`
- `edge_obsm_keys`
- `load_sparse_neighbor_tensor`
- `load_neighbor_expression`

`full_adata_manager` is always the full registered manager, even when the dataloader is scoped to a
train/validation/test split. This preserves cross-split neighbor context.

### ResolVI Model

`src/scviva/model/_resolvi.py` imports `GraphDataSplitter` from `scviva.dataloaders` and sets:

```python
_data_splitter_cls = GraphDataSplitter
```

`ResolVI.train()` accepts:

```python
cache_neighbor_expression: bool | Literal["auto"] = "auto"
neighbor_expression_cache_max_bytes: int | None = 1_000_000_000
```

In `"auto"` mode:

- graph splitters may enable the dense expression cache when the estimated float32 cache fits under
  `neighbor_expression_cache_max_bytes`;
- non-graph splitters leave the cache disabled unless explicitly requested;
- when graph cache is enabled, `datasplitter_kwargs["load_neighbor_expression"]` defaults to
  `False`, avoiding per-batch `x_n` transfer.

### ResolVI Module

`src/scviva/module/_resolvae.py` keeps `expression_anntorchdata` for backward compatibility.

`RESOLVAEModel._get_fn_args_from_batch()` resolves neighbor expression in this order:

1. If the model cache is enabled and `index_neighbor` is present, gather from the dense cache.
2. Else if graph batch field `x_n` is present, use it.
3. Else fall back to legacy `expression_anntorchdata[index_neighbor.flatten()]`.

The cache is stored as a plain attribute on `RESOLVAEModel`, not as a registered buffer. It is not
saved in checkpoints and can be dropped with `clear_neighbor_expression_cache()`.

### SCVIVA Model

`src/scviva/model/_scviva.py` imports `GraphDataSplitter` from `scviva.dataloaders` and sets:

```python
_data_splitter_cls = GraphDataSplitter
```

`SCVIVA.train()` maps graph splitter defaults to SCVIVA's registered niche graph fields:

```python
datasplitter_kwargs.setdefault("neighbor_indices_key", SCVIVA_REGISTRY_KEYS.NICHE_INDEXES_KEY)
datasplitter_kwargs.setdefault("edge_obsm_keys", [SCVIVA_REGISTRY_KEYS.NICHE_DISTANCES_KEY])
datasplitter_kwargs.setdefault("load_neighbor_expression", False)
```

SCVIVA does not need ResolVI-style module cache support. Its module consumes precomputed
neighborhood composition and latent niche tensors registered during `preprocessing_anndata()` and
`setup_anndata()`, so the graph dataloader is used to keep training batches graph-aware while
avoiding unused raw neighbor expression prefetch.

## Cross-Split Neighbor Policy

Cross-split neighbor expression is intentionally allowed.

Rationale:

- This matches upstream ResolVI behavior: neighbor expression has always been gathered from the full
  registered dataset.
- Spatial observations are not i.i.d.; masking cross-split neighbors would sever real spatial
  context.
- Neighbor expression is model input context. Loss is computed on the center cells in the active
  split.

Tradeoff:

- Validation loss can be slightly optimistic because validation-cell expression may appear as
  neighbor context during training.
- Strict split isolation would require recomputing or masking neighbor graphs per split and is not
  the target behavior for this port.

## Testing

Unit tests:

```text
tests/dataloaders/test_graph_dataloader.py
```

These cover:

- `torch_geometric.data.Data` output
- `x`, `x_n`, `edge_index`, and `edge_attr` shapes
- edge index mapping
- custom edge attributes
- sparse neighbor expression preservation
- cross-split neighbors
- missing `torch_geometric` error message
- collate-function graph construction
- `load_neighbor_expression=False`
- splitter forwarding

ResolVI integration tests:

```text
tests/model/test_resolvi_graph_dataloader.py
```

These cover:

- prefetched graph `x_n`
- sparse graph `x_n`
- model-side neighbor expression cache
- graph splitter default
- legacy splitter cache scope
- graph training
- size-factor training
- save/load and downstream APIs
- semisupervised APIs
- scArches query workflow
- differential expression settings
- benchmark-marked ELBO/speed checks

SCVIVA integration tests:

```text
tests/model/test_scviva_graph_dataloader.py
```

These cover:

- default `SCVIVA._data_splitter_cls`
- train-time forwarding of `niche_indexes` as the graph neighbor index key
- train-time forwarding of `niche_distances` as the graph edge attribute key
- omission of unused raw neighbor expression prefetch
- one-epoch graph training

Routine command:

```bash
pytest tests/dataloaders/test_graph_dataloader.py tests/model/test_resolvi_graph_dataloader.py -q
```

SCVIVA command:

```bash
env MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba pytest tests/model/test_scviva_graph_dataloader.py tests/model/test_scviva.py tests/regression/test_scviva_upstream.py -q
```

Benchmark subset:

```bash
pytest tests/model/test_resolvi_graph_dataloader.py -m benchmark -s
```

## Dependencies

`torch-geometric` is already listed under:

```toml
optional-dependencies.spatial = [
  "geomloss",
  "pynndescent",
  "spatialdata>=0.2",
  "squidpy>=1.4",
  "torch-geometric",
]
```

It remains optional because non-graph and non-spatial workflows should not require PyG.

## Out Of Scope

- Applying graph dataloading to `DestVI`, `Stereoscope`, and `GIMVI`.
- PyG `NeighborLoader`/`NodeLoader` sampling.
- Multi-hop graph sampling.
- GPU-resident sparse graph storage.
- Changing ResolVI model, objective, optimizer, or convergence behavior.
