"""
High‑level model classes for spatialvi‑tools.

The models subpackage exposes a suite of classes that wrap underlying
algorithms for spatial omics analysis.  Each class conforms to a common
interface defined by the :class:`~spatialvi_tools.models.base.BaseSpatialModel`
abstract base class, providing methods for training, inference and
prediction.

Import individual models directly from this module:

>>> from spatialvi_tools.models import NolanModel, HarremanModel
>>> model = NolanModel(adata)
>>> model.train()
>>> embeddings = model.predict(adata)
"""

from .nolan_model import NolanModel  # noqa: F401
from .lambda_model import LambdaModel  # noqa: F401
from .ppi_inference import PPIInference  # noqa: F401
from .vivs_model import VIVSModel  # noqa: F401
from .harreman_model import HarremanModel  # noqa: F401
from .amici_model import AmiciModel  # noqa: F401
from .starfysh_model import StarfyshModel  # noqa: F401
from .sparl_model import SparlModel  # noqa: F401
from .base._base_model import BaseSpatialModel  # noqa: F401

__all__ = [
    "NolanModel",
    "LambdaModel",
    "PPIInference",
    "VIVSModel",
    "HarremanModel",
    "AmiciModel",
    "StarfyshModel",
    "SparlModel",
    "BaseSpatialModel",
]