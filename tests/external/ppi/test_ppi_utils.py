"""Comprehensive tests for PPI (Prediction-Powered Inference) utilities."""

import numpy as np
import pytest


class TestPPIUtils:
    """Tests for PPI utility functions."""

    def test_compute_rectifier(self):
        """Test rectifier computation."""
        from spatialvi.external.ppi import compute_rectifier

        np.random.seed(42)
        n_labeled = 100
        n_unlabeled = 1000

        # Simulated data
        y = np.random.randn(n_labeled)
        yhat = y + np.random.randn(n_labeled) * 0.2  # Predictions close to y
        yhat_unlabeled = np.random.randn(n_unlabeled)

        rectifier = compute_rectifier(y, yhat, yhat_unlabeled)

        assert isinstance(rectifier, (float, np.floating))

    def test_compute_variance_reduction(self):
        """Test variance reduction computation."""
        from spatialvi.external.ppi import compute_variance_reduction

        np.random.seed(42)
        n_labeled = 100
        n_unlabeled = 1000

        y = np.random.randn(n_labeled)
        yhat = y + np.random.randn(n_labeled) * 0.1
        yhat_unlabeled = np.random.randn(n_unlabeled)

        var_reduction = compute_variance_reduction(y, yhat, yhat_unlabeled)

        assert 0 <= var_reduction <= 2  # Should typically be between 0 and 1
        # With good predictions, should have variance reduction < 1

    def test_stratified_sample(self):
        """Test stratified sampling."""
        from spatialvi.external.ppi import stratified_sample

        np.random.seed(42)
        labels = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
        n_labeled = 30

        indices = stratified_sample(labels, n_labeled, seed=42)

        # Should have approximately equal samples from each stratum
        selected_labels = labels[indices]
        unique, counts = np.unique(selected_labels, return_counts=True)

        assert len(indices) <= n_labeled
        assert len(unique) == 3  # All strata represented

    def test_compute_power_analysis(self):
        """Test power analysis computation."""
        from spatialvi.external.ppi import compute_power_analysis

        result = compute_power_analysis(
            effect_size=0.5,
            var_y=1.0,
            var_yhat=0.8,
            corr_y_yhat=0.9,
            alpha=0.05,
            power=0.8,
        )

        assert "n_classical" in result
        assert "n_ppi" in result
        assert "reduction_factor" in result
        assert result["n_classical"] > 0
        assert result["n_ppi"] > 0
        # With high correlation, PPI should need fewer samples
        assert result["reduction_factor"] < 1

    def test_compute_power_analysis_no_correlation(self):
        """Test power analysis with no correlation."""
        from spatialvi.external.ppi import compute_power_analysis

        result = compute_power_analysis(
            effect_size=0.5,
            var_y=1.0,
            var_yhat=0.8,
            corr_y_yhat=0.0,  # No correlation
            alpha=0.05,
            power=0.8,
        )

        # With no correlation, reduction factor should be ~1
        assert result["reduction_factor"] > 0.9

    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval computation."""
        from spatialvi.external.ppi import bootstrap_ci

        np.random.seed(42)
        n_labeled = 100
        n_unlabeled = 1000

        y = np.random.randn(n_labeled) + 5  # Mean around 5
        yhat = y + np.random.randn(n_labeled) * 0.2
        yhat_unlabeled = np.random.randn(n_unlabeled) + 5

        lower, upper = bootstrap_ci(
            y, yhat, yhat_unlabeled,
            n_bootstrap=100,
            alpha=0.05,
            seed=42,
        )

        assert lower < upper
        # CI should contain true mean approximately
        assert lower < 6 and upper > 4

    def test_bootstrap_ci_reproducibility(self):
        """Test bootstrap CI reproducibility with seed."""
        from spatialvi.external.ppi import bootstrap_ci

        np.random.seed(42)
        y = np.random.randn(50) + 5
        yhat = y + np.random.randn(50) * 0.2
        yhat_unlabeled = np.random.randn(500) + 5

        lower1, upper1 = bootstrap_ci(
            y, yhat, yhat_unlabeled, n_bootstrap=50, seed=123
        )
        lower2, upper2 = bootstrap_ci(
            y, yhat, yhat_unlabeled, n_bootstrap=50, seed=123
        )

        assert lower1 == lower2
        assert upper1 == upper2


class TestPPIModelWrapper:
    """Tests for PPIInference model wrapper."""

    def test_import(self):
        """Test that PPIInference wrapper can be imported."""
        from spatialvi.external.ppi import PPIInference

        assert PPIInference is not None

    def test_utils_import(self):
        """Test that all utils can be imported."""
        from spatialvi.external.ppi import (
            bootstrap_ci,
            compute_power_analysis,
            compute_rectifier,
            compute_variance_reduction,
            stratified_sample,
        )

        assert compute_rectifier is not None
        assert bootstrap_ci is not None

    def test_classical_mean_ci(self):
        """Test classical CI computation."""
        from spatialvi.external.ppi import PPIInference

        y = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98])
        ci_low, ci_high = PPIInference.classical_mean_ci(y, alpha=0.1)

        assert ci_low < 1.0 < ci_high
        assert ci_low < ci_high
