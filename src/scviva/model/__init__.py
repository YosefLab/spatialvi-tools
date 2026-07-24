from __future__ import annotations

from importlib import import_module

_lazy_model_map = {
    "SCVIVA": "scviva.model._scviva",
    "DestVI": "scviva.model._destvi",
    "ResolVI": "scviva.model._resolvi",
    "GIMVI": "scviva.model._gimvi",
    "VIVS": "scviva.model._vivs",
}


def __getattr__(name: str):
    if name in _lazy_model_map:
        mod = import_module(_lazy_model_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'scviva.model' has no attribute {name!r}")


__all__ = ["SCVIVA", "DestVI", "ResolVI", "GIMVI", "VIVS"]
