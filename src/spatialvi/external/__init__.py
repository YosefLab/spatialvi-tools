"""External models for spatialvi-tools.

These models are ported from scvi-tools and adapted to use spatialvi base classes
and mixins.
"""

from __future__ import annotations

from spatialvi.external.stereoscope._model import RNAStereoscope, SpatialStereoscope

__all__ = ["RNAStereoscope", "SpatialStereoscope"]
