"""External models for scviva-tools.

These models are ported from scvi-tools and adapted to use scviva base classes
and mixins.
"""

from __future__ import annotations

from scviva.external.stereoscope._model import RNAStereoscope, SpatialStereoscope

__all__ = ["RNAStereoscope", "SpatialStereoscope"]
