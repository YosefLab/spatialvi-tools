"""Tests for the PPIInference utilities."""

import numpy as np
import pytest

from spatialvi_tools.models import PPIInference


def test_mean_ci_without_dependency() -> None:
    y = np.array([1, 0, 1, 0])
    yhat = np.array([0.9, 0.1, 0.8, 0.2])
    yhat_unlabeled = np.array([0.7, 0.3, 0.6])
    try:
        ci_low, ci_high = PPIInference.mean_ci(y, yhat, yhat_unlabeled, alpha=0.1)
    except ImportError:
        pytest.skip("ppi_py dependency not installed")
    else:
        assert ci_low < ci_high