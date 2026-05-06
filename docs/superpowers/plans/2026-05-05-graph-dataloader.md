# Graph Dataloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the upstream scvi-tools ResolVI graph dataloader implementation into
`scviva-tools`, with `scviva.model.ResolVI` defaulting to graph dataloading while preserving the
legacy `AnnDataLoader` path.

**Architecture:** `scviva.dataloaders.GraphDataLoader` subclasses scvi-tools `AnnDataLoader` and
returns `torch_geometric.data.Data` batches from a custom collate function.
`scviva.dataloaders.GraphDataSplitter` subclasses `DataSplitter` and creates graph dataloaders for
train/validation/test splits. `ResolVI` defaults to `GraphDataSplitter`; `RESOLVAEModel` consumes
prefetched graph `x_n`, can gather from a model-side dense expression cache, and falls back to
legacy `AnnTorchDataset` lookup for non-graph batches.

**Tech Stack:** PyTorch, PyG (`torch-geometric`, optional), scvi-tools `AnnDataManager`,
`AnnDataLoader`, `DataSplitter`, Lightning/Pyro training.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/scviva/dataloaders/_graph_dataloader.py` | Added | `GraphDataLoader`, `GraphDataSplitter`, `_GraphBatchConverter` |
| `src/scviva/dataloaders/__init__.py` | Updated | export graph dataloader classes |
| `src/scviva/model/_resolvi.py` | Update | default graph splitter and train-time cache wiring |
| `src/scviva/module/_resolvae.py` | Update | graph `x_n`, cache, and legacy fallback in `_get_fn_args_from_batch` |
| `tests/dataloaders/test_graph_dataloader.py` | Added/updated | graph dataloader unit coverage for `scviva.dataloaders` |
| `tests/model/test_resolvi_graph_dataloader.py` | Added/updated | ResolVI graph integration coverage for `scviva.model.ResolVI` |
| `pyproject.toml` | Update | register `benchmark` pytest marker |
| `CHANGELOG.md` | Update | user-visible graph dataloader and ResolVI training changes |
| `.ai_handoff.md` | Update | handoff state for future agents |

Reference implementation:

```text
/Users/orikr/PycharmProjects/scvi-tools/src/scvi/dataloaders/_graph_dataloader.py
/Users/orikr/PycharmProjects/scvi-tools/src/scvi/external/resolvi/_model.py
/Users/orikr/PycharmProjects/scvi-tools/src/scvi/external/resolvi/_module.py
```

## Task 1: Adapt Tests To scviva Paths

**Files:**

- Modify: `tests/dataloaders/test_graph_dataloader.py`
- Modify: `tests/model/test_resolvi_graph_dataloader.py`

- [x] **Step 1: Replace upstream model imports**

Use:

```python
from scviva.model import ResolVI
```

instead of:

```python
from scvi.external import RESOLVI
```

- [x] **Step 2: Replace graph dataloader imports**

Use:

```python
from scviva.dataloaders import GraphDataLoader, GraphDataSplitter
```

instead of importing graph classes from `scvi.dataloaders`.

- [x] **Step 3: Keep legacy splitter comparison**

The legacy comparison should still import scvi-tools `DataSplitter`:

```python
from scvi.dataloaders import DataSplitter
```

Then define:

```python
class ResolVILegacy(ResolVI):
    _data_splitter_cls = DataSplitter
```

- [x] **Step 4: Run the red baseline**

Command:

```bash
pytest tests/dataloaders/test_graph_dataloader.py tests/model/test_resolvi_graph_dataloader.py -q
```

Observed before implementation:

- `tests/dataloaders/test_graph_dataloader.py`: passed.
- `tests/model/test_resolvi_graph_dataloader.py`: failed on missing ResolVI graph integration.

Expected failure categories:

- graph batch `x_n` ignored by `_get_fn_args_from_batch`
- missing `configure_neighbor_expression_cache`
- missing `_neighbor_expression_cache`
- default splitter still `DataSplitter`

## Task 2: Confirm Graph Dataloader Exists In scviva

**Files:**

- Inspect: `src/scviva/dataloaders/_graph_dataloader.py`
- Inspect: `src/scviva/dataloaders/__init__.py`

- [x] **Step 1: Confirm loader implementation**

`GraphDataLoader` must:

- subclass `AnnDataLoader`;
- reject caller-provided `collate_fn`;
- reject `iter_ndarray=True`;
- install `_GraphBatchConverter` as `collate_fn`;
- support `load_sparse_neighbor_tensor`;
- support `load_neighbor_expression=False`;
- build `edge_index`, `edge_attr`, `x`, optional `x_n`, and `distances_n`.

- [x] **Step 2: Confirm splitter implementation**

`GraphDataSplitter` must:

- subclass `DataSplitter`;
- create graph train/validation/test dataloaders;
- pass full registered `adata_manager` as `full_adata_manager`;
- forward `neighbor_indices_key`, `edge_obsm_keys`, `load_sparse_neighbor_tensor`, and
  `load_neighbor_expression`.

- [x] **Step 3: Confirm exports**

`src/scviva/dataloaders/__init__.py` must export:

```python
from ._graph_dataloader import GraphDataLoader, GraphDataSplitter

__all__ = [
    "GraphDataLoader",
    "GraphDataSplitter",
]
```

## Task 3: Port ResolVI Default Splitter And Cache Wiring

**Files:**

- Modify: `src/scviva/model/_resolvi.py`

- [x] **Step 1: Import graph splitter**

Add:

```python
from scviva.dataloaders import GraphDataSplitter
```

- [x] **Step 2: Set default splitter**

Inside `class ResolVI`:

```python
_data_splitter_cls = GraphDataSplitter
```

- [x] **Step 3: Add train cache parameters**

Extend `ResolVI.train()` with:

```python
cache_neighbor_expression: bool | Literal["auto"] = "auto",
neighbor_expression_cache_max_bytes: int | None = 1_000_000_000,
```

- [x] **Step 4: Wire datasplitter kwargs**

Before `super().train(...)`:

```python
datasplitter_kwargs = dict(kwargs.pop("datasplitter_kwargs", {}) or {})
uses_graph_splitter = issubclass(self._data_splitter_cls, GraphDataSplitter)
cache_request = cache_neighbor_expression
if cache_request == "auto" and not uses_graph_splitter:
    cache_request = False
cache_enabled = self.module.configure_neighbor_expression_cache(
    cache=cache_request,
    max_bytes=neighbor_expression_cache_max_bytes,
)
if cache_enabled and uses_graph_splitter:
    datasplitter_kwargs.setdefault("load_neighbor_expression", False)
```

Then pass `datasplitter_kwargs=datasplitter_kwargs` to `super().train(...)`.

- [x] **Step 5: Document train parameters**

Document that `"auto"` cache mode is graph-splitter scoped and guarded by estimated dense float32
cache size.

## Task 4: Port ResolVI Module Batch And Cache Support

**Files:**

- Modify: `src/scviva/module/_resolvae.py`

- [x] **Step 1: Add cache state**

In `RESOLVAEModel.__init__`:

```python
self._use_neighbor_expression_cache = False
self._neighbor_expression_cache = None
self._neighbor_expression_cache_max_bytes = None
```

- [x] **Step 2: Add cache helpers**

Add:

```python
def _estimate_neighbor_expression_cache_bytes(self) -> int: ...

def configure_neighbor_expression_cache(
    self,
    cache: bool | str = "auto",
    max_bytes: int | None = 1_000_000_000,
) -> bool: ...

def clear_neighbor_expression_cache(self) -> None: ...

def _get_neighbor_expression_from_cache(
    self,
    ind_neighbors: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor: ...
```

The cache is a plain attribute, not a registered buffer.

- [x] **Step 3: Update `_get_fn_args_from_batch`**

Resolve neighbor expression in priority order:

```python
if self._use_neighbor_expression_cache and "index_neighbor" in tensor_dict:
    distances_n = tensor_dict["distances_n"] if "distances_n" in tensor_dict else tensor_dict["distance_neighbor"]
    ind_neighbors = tensor_dict["index_neighbor"].long()
    x_n = self._get_neighbor_expression_from_cache(ind_neighbors, x)
elif "x_n" in tensor_dict:
    x_n = tensor_dict["x_n"]
    distances_n = tensor_dict["distances_n"] if "distances_n" in tensor_dict else tensor_dict["distance_neighbor"]
else:
    distances_n = tensor_dict["distance_neighbor"]
    ind_neighbors = tensor_dict["index_neighbor"].long()
    x_n = self.expression_anntorchdata[ind_neighbors.cpu().numpy().flatten(), :]["X"]
```

Keep existing sparse densification and `x_n.reshape(x.shape[0], -1)`.

- [x] **Step 4: Expose cache methods on `RESOLVAE`**

Add:

```python
def configure_neighbor_expression_cache(...): ...
def clear_neighbor_expression_cache(...): ...
```

that forward to `self._model`.

## Task 5: Verify Focused Graph Behavior

**Files:**

- Test: `tests/dataloaders/test_graph_dataloader.py`
- Test: `tests/model/test_resolvi_graph_dataloader.py`

- [x] **Step 1: Run focused tests**

Command:

```bash
pytest tests/dataloaders/test_graph_dataloader.py tests/model/test_resolvi_graph_dataloader.py -q
```

Observed after port:

```text
37 passed
```

- [x] **Step 2: Register benchmark marker**

Add to `pyproject.toml`:

```toml
"benchmark: mark benchmark tests for graph-dataloader speed and ELBO comparisons",
```

## Task 6: Update Documentation And Handoff

**Files:**

- Modify: `docs/superpowers/specs/2026-05-05-graph-dataloader-design.md`
- Modify: `docs/superpowers/plans/2026-05-05-graph-dataloader.md`
- Modify: `.ai_handoff.md`
- Modify: `CHANGELOG.md`

- [x] **Step 1: Update design paths**

Use `src/scviva/...` and `tests/model/...` paths throughout.

- [x] **Step 2: Update plan paths**

Use `scviva.model.ResolVI`, `scviva.dataloaders.GraphDataLoader`, and
`scviva.dataloaders.GraphDataSplitter`.

- [x] **Step 3: Update handoff after final verification**

`.ai_handoff.md` must include:

- current workspace path;
- changed files;
- exact verification commands and results;
- current implementation state;
- remaining stage-2 work for `SCVIVA`, `DestVI`, `Stereoscope`, and `GIMVI`;
- any known caveats.

- [x] **Step 4: Update changelog**

Add Unreleased entries for:

- graph dataloaders;
- ResolVI default graph splitter and cache;
- graph tests ported to `scviva-tools`.

## Task 7: Broader Verification

**Files:**

- Test: `tests/model/test_resolvi.py` if present
- Test: `tests/regression/test_resolvi_upstream.py`
- Syntax: changed Python files and tests

- [x] **Step 1: Run existing ResolVI tests**

Command:

```bash
pytest tests/model/test_resolvi.py -q
```

Observed:

```text
4 passed
```

If the file does not exist, record that explicitly in `.ai_handoff.md`.

- [x] **Step 2: Run py_compile**

Command:

```bash
python -m py_compile \
  src/scviva/dataloaders/_graph_dataloader.py \
  src/scviva/model/_resolvi.py \
  src/scviva/module/_resolvae.py \
  tests/dataloaders/test_graph_dataloader.py \
  tests/model/test_resolvi_graph_dataloader.py
```

Expected:

```text
exit code 0
```

- [x] **Step 3: Run upstream parity regression subset**

Command:

```bash
env MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba pytest tests/regression/test_resolvi_upstream.py -q
```

Observed:

```text
8 passed
```

## Stage 2 Follow-Up

Evaluate graph dataloading for the other spatial models:

- `src/scviva/model/_scviva.py`
- `src/scviva/model/_destvi.py`
- `src/scviva/external/stereoscope/_model.py`
- `src/scviva/model/_gimvi.py`

Apply the graph dataloader only where the model uses fixed per-observation neighbor indices or can
benefit from graph batch context without changing model semantics. DestVI, Stereoscope, and GIMVI
may not use the same `index_neighbor`/`distance_neighbor` contract, so stage 2 starts with data-flow
analysis rather than blindly setting `_data_splitter_cls`.

## Completion Checklist

- [x] Tests point at `scviva.model.ResolVI`.
- [x] Tests import graph dataloaders from `scviva.dataloaders`.
- [x] `GraphDataLoader` and `GraphDataSplitter` exist under `src/scviva/dataloaders`.
- [x] `ResolVI` defaults to `GraphDataSplitter`.
- [x] `ResolVI.train()` configures graph-scoped neighbor expression cache.
- [x] `RESOLVAEModel._get_fn_args_from_batch()` supports cache, graph `x_n`, and legacy fallback.
- [x] Focused graph tests pass.
- [x] Design doc updated to scviva paths.
- [x] Plan doc updated to scviva paths.
- [x] Changelog updated.
- [x] `.ai_handoff.md` updated after final verification.
- [x] Broader verification recorded.
