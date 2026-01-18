"""Prediction-Powered Inference utilities.

PPI combines machine learning predictions with a small labeled dataset
to provide valid statistical inference with reduced variance.

This module provides the PPIInference wrapper class and supporting components
for prediction-powered statistical analysis.
"""

from __future__ import annotations

from ._model import PPIInference
from ._module import (
    MeanEstimator,
    OLSEstimator,
    PPIConfig,
    PPIResult,
    SpatialMeanEstimator,
    SpatialPPIResult,
)
from ._utils import (
    bootstrap_ci,
    compute_power_analysis,
    compute_rectifier,
    compute_variance_reduction,
    stratified_sample,
)

__all__ = [
    # Model wrapper
    "PPIInference",
    # Module components
    "PPIConfig",
    "PPIResult",
    "MeanEstimator",
    "OLSEstimator",
    "SpatialMeanEstimator",
    "SpatialPPIResult",
    # Utility functions
    "bootstrap_ci",
    "compute_power_analysis",
    "compute_rectifier",
    "compute_variance_reduction",
    "stratified_sample",
]
