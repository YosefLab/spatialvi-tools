from __future__ import annotations

# Model imports are deferred to avoid ImportError while model files are being built.
# Populated fully in Task 20 (final integration).
from importlib import import_module

_lazy_model_map = {
    "SCVIVA": "spatialvi.model._scviva",
    "DestVI": "spatialvi.model._destvi",
    "ResolVI": "spatialvi.model._resolvi",
    "GIMVI": "spatialvi.model._gimvi",
}


def __getattr__(name: str):
    if name in _lazy_model_map:
        mod = import_module(_lazy_model_map[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'spatialvi.model' has no attribute {name!r}")


__all__ = ["SCVIVA", "DestVI", "ResolVI", "GIMVI"]
