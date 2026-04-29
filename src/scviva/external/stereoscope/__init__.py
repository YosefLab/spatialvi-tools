"""Stereoscope external package for spatial transcriptomics deconvolution."""

from __future__ import annotations

from scviva.external.stereoscope._model import RNAStereoscope, SpatialStereoscope
from scviva.external.stereoscope._module import RNADeconv, SpatialDeconv

__all__ = ["RNAStereoscope", "SpatialStereoscope", "RNADeconv", "SpatialDeconv"]
