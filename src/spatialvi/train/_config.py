from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from typing import TypeAlias


TorchOptimizerCreator: TypeAlias = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]


@runtime_checkable
class KwargsConfig(Protocol):
    """Protocol for config objects that can be expanded into kwargs."""

    def to_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments compatible with a downstream constructor."""


KwargsLike: TypeAlias = Mapping[str, Any] | KwargsConfig


def _coerce_kwargs(value: KwargsLike | None, *, name: str) -> dict[str, Any]:
    """Normalize a kwargs-like object into a dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    to_kwargs = getattr(value, "to_kwargs", None)
    if callable(to_kwargs):
        out = to_kwargs()
        if not isinstance(out, dict):
            raise TypeError(f"{name}.to_kwargs() must return a dict, got {type(out)!r}.")
        return out
    raise TypeError(f"{name} must be a mapping or a config with to_kwargs().")


def merge_kwargs(
    config: KwargsLike | None,
    overrides: KwargsLike | None,
    *,
    name: str,
) -> dict[str, Any]:
    """Merge config kwargs with overrides, with overrides taking precedence."""
    merged = dict(_coerce_kwargs(config, name=f"{name}_config"))
    merged.update(_coerce_kwargs(overrides, name=name))
    return merged
