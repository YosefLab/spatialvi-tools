from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scviva.model.utils._dataloaders import CyclicMultiDataLoader


def _make_loader(n, batch_size=4):
    x = torch.arange(n).float().unsqueeze(-1)
    ds = TensorDataset(x)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    dl.indices = list(range(n))
    return dl


def test_cyclic_loader_list_mode_cycles_shorter_loader():
    long_dl = _make_loader(20)
    short_dl = _make_loader(8)
    loader = CyclicMultiDataLoader([long_dl, short_dl])

    assert len(loader) == len(long_dl)
    batches = list(loader)
    assert len(batches) == len(long_dl)
    for batch in batches:
        assert isinstance(batch, tuple)
        assert len(batch) == 2


def test_cyclic_loader_dict_mode_yields_keyed_batches():
    loaders = {"seq": _make_loader(20), "spatial": _make_loader(8)}
    loader = CyclicMultiDataLoader(loaders)

    assert len(loader) == len(loaders["seq"])
    batches = list(loader)
    assert len(batches) == len(loaders["seq"])
    for batch in batches:
        assert isinstance(batch, dict)
        assert set(batch.keys()) == {"seq", "spatial"}


def test_cyclic_loader_requires_at_least_one_loader():
    with pytest.raises(ValueError, match="At least one data loader"):
        CyclicMultiDataLoader([])
