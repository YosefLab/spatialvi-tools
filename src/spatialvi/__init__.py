"""spatialvi-tools: Consolidated spatial transcriptomics analysis toolkit."""

from __future__ import annotations

import logging
from importlib.metadata import version

from ._constants import REGISTRY_KEYS, SPATIAL_REGISTRY_KEYS
from ._settings import settings

# Core modules
from . import data, model, module, nn, train, utils

# External models
from . import external

package_name = "spatialvi-tools"
__version__ = version(package_name) if package_name else "0.1.0"

settings.verbosity = logging.INFO

# Configure logger
spatialvi_logger = logging.getLogger("spatialvi")
spatialvi_logger.propagate = False

__all__ = [
    "settings",
    "REGISTRY_KEYS",
    "SPATIAL_REGISTRY_KEYS",
    "data",
    "model",
    "module",
    "nn",
    "train",
    "utils",
    "external",
]
