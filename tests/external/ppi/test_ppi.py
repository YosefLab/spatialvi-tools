"""Tests for PPIInference (Prediction-Powered Inference)."""

import numpy as np
import pytest


class TestPPIInference:
    """Tests for PPIInference static methods."""

    def test_import(self):
        """Test PPIInference can be imported."""
        from spatialvi.external import PPIInference

        assert PPIInference is not None

    def test_classical_mean_ci(self):
        """Test classical mean CI computation."""
        from spatialvi.external import PPIInference

        y = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98])
        ci_low, ci_high = PPIInference.classical_mean_ci(y, alpha=0.1)

        # Check bounds make sense
        assert ci_low < np.mean(y)
        assert ci_high > np.mean(y)
        assert ci_low < ci_high

    def test_classical_mean_ci_different_alpha(self):
        """Test classical CI with different alpha values."""
        from spatialvi.external import PPIInference

        y = np.array([1.0, 1.1, 0.9, 1.05, 0.95])

        # 90% CI
        ci_low_90, ci_high_90 = PPIInference.classical_mean_ci(y, alpha=0.1)
        # 95% CI
        ci_low_95, ci_high_95 = PPIInference.classical_mean_ci(y, alpha=0.05)

        # 95% CI should be wider than 90% CI
        width_90 = ci_high_90 - ci_low_90
        width_95 = ci_high_95 - ci_low_95
        assert width_95 > width_90


class TestPPIMeanCI:
    """Tests for PPI mean CI methods."""

    def test_mean_ci_without_dependency(self):
        """Test mean_ci handles missing dependency."""
        from spatialvi.external import PPIInference

        y = np.array([1.0, 1.1, 0.9])
        yhat = np.array([1.05, 1.08, 0.92])
        yhat_unlabeled = np.random.normal(1.0, 0.1, 100)

        try:
            ci = PPIInference.mean_ci(y, yhat, yhat_unlabeled)
            assert len(ci) == 2
        except ImportError:
            pytest.skip("ppi_py package not installed")

    def test_mean_pointestimate_without_dependency(self):
        """Test mean_pointestimate handles missing dependency."""
        from spatialvi.external import PPIInference

        y = np.array([1.0, 1.1, 0.9])
        yhat = np.array([1.05, 1.08, 0.92])
        yhat_unlabeled = np.random.normal(1.0, 0.1, 100)

        try:
            estimate = PPIInference.mean_pointestimate(y, yhat, yhat_unlabeled)
            assert isinstance(estimate, (int, float, np.number))
        except ImportError:
            pytest.skip("ppi_py package not installed")


class TestPPIOLSMethods:
    """Tests for PPI OLS regression methods."""

    def test_ols_ci_without_dependency(self):
        """Test ols_ci handles missing dependency."""
        from spatialvi.external import PPIInference

        n_labeled = 10
        n_unlabeled = 100
        n_features = 3

        X = np.random.randn(n_labeled, n_features)
        y = np.random.randn(n_labeled)
        yhat = np.random.randn(n_labeled)
        X_unlabeled = np.random.randn(n_unlabeled, n_features)
        yhat_unlabeled = np.random.randn(n_unlabeled)

        try:
            ci_low, ci_high = PPIInference.ols_ci(X, y, yhat, X_unlabeled, yhat_unlabeled)
            assert len(ci_low) == n_features
            assert len(ci_high) == n_features
        except ImportError:
            pytest.skip("ppi_py package not installed")

    def test_ols_pointestimate_without_dependency(self):
        """Test ols_pointestimate handles missing dependency."""
        from spatialvi.external import PPIInference

        n_labeled = 10
        n_unlabeled = 100
        n_features = 3

        X = np.random.randn(n_labeled, n_features)
        y = np.random.randn(n_labeled)
        yhat = np.random.randn(n_labeled)
        X_unlabeled = np.random.randn(n_unlabeled, n_features)
        yhat_unlabeled = np.random.randn(n_unlabeled)

        try:
            coefs = PPIInference.ols_pointestimate(X, y, yhat, X_unlabeled, yhat_unlabeled)
            assert len(coefs) == n_features
        except ImportError:
            pytest.skip("ppi_py package not installed")
