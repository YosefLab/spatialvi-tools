"""
Top‑level package for spatialvi‑tools.

This package consolidates multiple cutting‑edge algorithms for spatial
omics analysis under a unified API.  Each model can be imported
directly from this top‑level namespace.  For example:

>>> from spatialvi_tools import NolanModel, LambdaModel
>>> model = NolanModel(adata)

The ``__all__`` variable defines the public API of the package.
"""

from .version import __version__  # noqa: F401

# high‑level model classes
from .models.nolan_model import NolanModel  # noqa: F401
from .models.lambda_model import LambdaModel  # noqa: F401
from .models.ppi_inference import PPIInference  # noqa: F401
from .models.vivs_model import VIVSModel  # noqa: F401
from .models.harreman_model import HarremanModel  # noqa: F401
from .models.amici_model import AmiciModel  # noqa: F401
from .models.starfysh_model import StarfyshModel  # noqa: F401
from .models.sparl_model import SparlModel  # noqa: F401

__all__ = [
    "__version__",
    "NolanModel",
    "LambdaModel",
    "PPIInference",
    "VIVSModel",
    "HarremanModel",
    "AmiciModel",
    "StarfyshModel",
    "SparlModel",
]