"""Base model classes for spatial transcriptomics."""

from __future__ import annotations

from ._base_model import BaseSpatialModel
from ._embedding_mixin import EmbeddingMixin, SpatialEmbeddingMixin
from ._mixins import DeconvolutionMixin, NicheMixin, SpatialMixin
from ._training_mixins import (
    NicheBatchSampler,
    SpatialBatchSampler,
    SpatialRegularizationCallback,
    SpatialSamplerMixin,
    SpatialTrainingMixin,
)
from ._vaemixin import VAEMixin

__all__ = [
    "BaseSpatialModel",
    "DeconvolutionMixin",
    "EmbeddingMixin",
    "NicheBatchSampler",
    "NicheMixin",
    "SpatialBatchSampler",
    "SpatialEmbeddingMixin",
    "SpatialMixin",
    "SpatialRegularizationCallback",
    "SpatialSamplerMixin",
    "SpatialTrainingMixin",
    "VAEMixin",
]