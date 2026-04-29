from __future__ import annotations

from importlib import import_module

from scviva._settings import settings

__version__ = "0.1.0"

_MODEL_NAMES = {"SCVIVA", "DestVI", "ResolVI", "GIMVI"}


def __getattr__(name: str):
    if name in _MODEL_NAMES:
        mod = import_module("scviva.model")
        return getattr(mod, name)
    raise AttributeError(f"module 'scviva' has no attribute {name!r}")


__all__ = ["SCVIVA", "DestVI", "ResolVI", "GIMVI", "__version__", "settings"]
