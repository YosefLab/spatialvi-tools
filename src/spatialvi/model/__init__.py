"""Spatial transcriptomics models."""

from __future__ import annotations

from ._spatial_vae import SpatialVAE
from .base import BaseSpatialModel, NicheMixin, SpatialMixin

__all__ = [
    "BaseSpatialModel",
    "SpatialMixin",
    "NicheMixin",
    "SpatialVAE",
]
