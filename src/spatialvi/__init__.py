from __future__ import annotations

from importlib import import_module

from spatialvi._settings import settings

__version__ = "0.1.0"

_lazy_map = {
    "SCVIVA": "spatialvi.model._scviva",
    "DestVI": "spatialvi.model._destvi",
    "ResolVI": "spatialvi.model._resolvi",
    "GIMVI": "spatialvi.model._gimvi",
}


def __getattr__(name: str):
    if name in _lazy_map:
        mod = import_module(_lazy_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'spatialvi' has no attribute {name!r}")


__all__ = ["SCVIVA", "DestVI", "ResolVI", "GIMVI", "__version__", "settings"]
