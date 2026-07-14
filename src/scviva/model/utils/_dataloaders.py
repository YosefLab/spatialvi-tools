"""Shared cyclic multi-loader for models that train on more than one dataset at once."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import cycle
from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from collections.abc import Sequence


class CyclicMultiDataLoader(DataLoader):
    """Combine multiple data loaders by cycling shorter loaders to match the longest one.

    Shared by GIMVI and DIAGVI, both of which train on more than one dataset per step
    and need batches aligned across datasets of different sizes.

    Parameters
    ----------
    data_loaders
        Either a mapping of name -> DataLoader (batches are yielded as a dict keyed by
        the same names) or a plain sequence of DataLoader (batches are yielded as a tuple).
    **data_loader_kwargs
        Forwarded to the underlying ``DataLoader.__init__``.
    """

    def __init__(
        self,
        data_loaders: Mapping[str, DataLoader] | Sequence[DataLoader],
        **data_loader_kwargs: Any,
    ):
        self._returns_mapping = isinstance(data_loaders, Mapping)
        if self._returns_mapping:
            self.input_names = list(data_loaders.keys())
            self.data_loader_list = list(data_loaders.values())
        else:
            self.input_names = None
            self.data_loader_list = list(data_loaders)

        if not self.data_loader_list:
            raise ValueError("At least one data loader is required.")

        self.largest_train_dl_idx = max(
            range(len(self.data_loader_list)),
            key=lambda idx: len(self.data_loader_list[idx].indices),
        )
        self.largest_dl = self.data_loader_list[self.largest_train_dl_idx]
        super().__init__(self.largest_dl, **data_loader_kwargs)

    def __len__(self):
        return len(self.largest_dl)

    def __iter__(self):
        data_loaders = [
            dl if i == self.largest_train_dl_idx else cycle(dl)
            for i, dl in enumerate(self.data_loader_list)
        ]

        if self._returns_mapping:
            for batches in zip(*data_loaders, strict=False):
                yield dict(zip(self.input_names, batches, strict=False))
        else:
            yield from zip(*data_loaders, strict=False)
