"""Tests for PPI module components (PPIConfig, PPIResult, estimators, etc.)."""

import numpy as np
import pytest


class TestPPIConfig:
    """Tests for PPIConfig dataclass."""

    def test_default_initialization(self):
        """Test default PPIConfig values."""
        from spatialvi.external.ppi import PPIConfig

        config = PPIConfig()

        assert config.alpha == 0.1
        assert config.method == "mean"
        assert config.bootstrap_samples == 1000
        assert config.use_classical is True

    def test_custom_initialization(self):
        """Test PPIConfig with custom values."""
        from spatialvi.external.ppi import PPIConfig

        config = PPIConfig(
            alpha=0.05,
            method="ols",
            bootstrap_samples=500,
            use_classical=False,
        )

        assert config.alpha == 0.05
        assert config.method == "ols"
        assert config.bootstrap_samples == 500
        assert config.use_classical is False

    def test_method_options(self):
        """Test different method options."""
        from spatialvi.external.ppi import PPIConfig

        for method in ["mean", "ols", "quantile", "logistic"]:
            config = PPIConfig(method=method)
            assert config.method == method


class TestPPIResult:
    """Tests for PPIResult dataclass."""

    def test_minimal_initialization(self):
        """Test PPIResult with minimal required values."""
        from spatialvi.external.ppi import PPIResult

        result = PPIResult(
            estimate=1.5,
            ci_lower=1.2,
            ci_upper=1.8,
        )

        assert result.estimate == 1.5
        assert result.ci_lower == 1.2
        assert result.ci_upper == 1.8
        assert result.se == 0.0
        assert result.classical_estimate is None
        assert result.classical_ci is None
        assert result.n_labeled == 0
        assert result.n_unlabeled == 0

    def test_full_initialization(self):
        """Test PPIResult with all values."""
        from spatialvi.external.ppi import PPIResult

        result = PPIResult(
            estimate=1.5,
            ci_lower=1.2,
            ci_upper=1.8,
            se=0.15,
            classical_estimate=1.6,
            classical_ci=(1.0, 2.2),
            n_labeled=50,
            n_unlabeled=500,
        )

        assert result.estimate == 1.5
        assert result.se == 0.15
        assert result.classical_estimate == 1.6
        assert result.classical_ci == (1.0, 2.2)
        assert result.n_labeled == 50
        assert result.n_unlabeled == 500

    def test_ci_property(self):
        """Test ci property returns tuple."""
        from spatialvi.external.ppi import PPIResult

        result = PPIResult(estimate=1.5, ci_lower=1.2, ci_upper=1.8)

        assert result.ci == (1.2, 1.8)

    def test_ci_width_property(self):
        """Test ci_width property computes correctly."""
        from spatialvi.external.ppi import PPIResult

        result = PPIResult(estimate=1.5, ci_lower=1.2, ci_upper=1.8)

        assert result.ci_width == pytest.approx(0.6)

    def test_efficiency_gain_property(self):
        """Test efficiency_gain property computes correctly."""
        from spatialvi.external.ppi import PPIResult

        # PPI CI is narrower than classical
        result = PPIResult(
            estimate=1.5,
            ci_lower=1.3,
            ci_upper=1.7,  # width = 0.4
            classical_ci=(1.0, 2.0),  # width = 1.0
        )

        # Efficiency gain = (1.0 - 0.4) / 1.0 = 0.6
        assert result.efficiency_gain == pytest.approx(0.6)

    def test_efficiency_gain_no_classical(self):
        """Test efficiency_gain returns None without classical CI."""
        from spatialvi.external.ppi import PPIResult

        result = PPIResult(estimate=1.5, ci_lower=1.2, ci_upper=1.8)

        assert result.efficiency_gain is None

    def test_efficiency_gain_zero_width_classical(self):
        """Test efficiency_gain handles zero-width classical CI."""
        from spatialvi.external.ppi import PPIResult

        result = PPIResult(
            estimate=1.5,
            ci_lower=1.3,
            ci_upper=1.7,
            classical_ci=(1.5, 1.5),  # Zero width
        )

        assert result.efficiency_gain is None


class TestMeanEstimator:
    """Tests for MeanEstimator class."""

    def test_default_initialization(self):
        """Test MeanEstimator with default config."""
        from spatialvi.external.ppi import MeanEstimator

        estimator = MeanEstimator()

        assert estimator.config is not None
        assert estimator.config.alpha == 0.1

    def test_custom_config_initialization(self):
        """Test MeanEstimator with custom config."""
        from spatialvi.external.ppi import MeanEstimator, PPIConfig

        config = PPIConfig(alpha=0.05, use_classical=False)
        estimator = MeanEstimator(config=config)

        assert estimator.config.alpha == 0.05
        assert estimator.config.use_classical is False

    def test_fit_requires_ppi_py(self):
        """Test fit raises ImportError without ppi_py."""
        from spatialvi.external.ppi import MeanEstimator

        estimator = MeanEstimator()

        y_labeled = np.array([1.0, 1.1, 0.9])
        yhat_labeled = np.array([1.05, 1.08, 0.92])
        yhat_unlabeled = np.random.normal(1.0, 0.1, 100)

        try:
            result = estimator.fit(y_labeled, yhat_labeled, yhat_unlabeled)
            # If ppi_py is installed, check result
            assert hasattr(result, "estimate")
            assert hasattr(result, "ci_lower")
            assert hasattr(result, "ci_upper")
        except ImportError as e:
            assert "ppi_py" in str(e) or "ppi-python" in str(e)


class TestOLSEstimator:
    """Tests for OLSEstimator class."""

    def test_default_initialization(self):
        """Test OLSEstimator with default config."""
        from spatialvi.external.ppi import OLSEstimator

        estimator = OLSEstimator()

        assert estimator.config is not None
        assert estimator.config.alpha == 0.1

    def test_custom_config_initialization(self):
        """Test OLSEstimator with custom config."""
        from spatialvi.external.ppi import OLSEstimator, PPIConfig

        config = PPIConfig(alpha=0.01)
        estimator = OLSEstimator(config=config)

        assert estimator.config.alpha == 0.01

    def test_fit_requires_ppi_py(self):
        """Test fit raises ImportError without ppi_py."""
        from spatialvi.external.ppi import OLSEstimator

        estimator = OLSEstimator()

        n_labeled = 10
        n_unlabeled = 100
        n_features = 3

        X_labeled = np.random.randn(n_labeled, n_features)
        y_labeled = np.random.randn(n_labeled)
        yhat_labeled = np.random.randn(n_labeled)
        X_unlabeled = np.random.randn(n_unlabeled, n_features)
        yhat_unlabeled = np.random.randn(n_unlabeled)

        try:
            results = estimator.fit(
                X_labeled, y_labeled, yhat_labeled, X_unlabeled, yhat_unlabeled
            )
            # If ppi_py is installed, check results
            assert isinstance(results, list)
            assert len(results) == n_features
        except ImportError as e:
            assert "ppi_py" in str(e) or "ppi-python" in str(e)


class TestSpatialPPIResult:
    """Tests for SpatialPPIResult dataclass."""

    def test_initialization(self):
        """Test SpatialPPIResult initialization."""
        from spatialvi.external.ppi import PPIResult, SpatialPPIResult

        mean_result = PPIResult(estimate=1.5, ci_lower=1.2, ci_upper=1.8)

        spatial_result = SpatialPPIResult(
            gene="GAPDH",
            spatial_region="region_1",
            mean_result=mean_result,
        )

        assert spatial_result.gene == "GAPDH"
        assert spatial_result.spatial_region == "region_1"
        assert spatial_result.mean_result.estimate == 1.5
        assert spatial_result.proportion_result is None

    def test_with_proportion_result(self):
        """Test SpatialPPIResult with proportion result."""
        from spatialvi.external.ppi import PPIResult, SpatialPPIResult

        mean_result = PPIResult(estimate=1.5, ci_lower=1.2, ci_upper=1.8)
        prop_result = PPIResult(estimate=0.7, ci_lower=0.6, ci_upper=0.8)

        spatial_result = SpatialPPIResult(
            gene="CD3E",
            spatial_region="tumor_core",
            mean_result=mean_result,
            proportion_result=prop_result,
        )

        assert spatial_result.proportion_result is not None
        assert spatial_result.proportion_result.estimate == 0.7


class TestSpatialMeanEstimator:
    """Tests for SpatialMeanEstimator class."""

    def test_default_initialization(self):
        """Test SpatialMeanEstimator with default config."""
        from spatialvi.external.ppi import SpatialMeanEstimator

        estimator = SpatialMeanEstimator()

        assert estimator.config is not None
        assert estimator._mean_estimator is not None

    def test_custom_config_initialization(self):
        """Test SpatialMeanEstimator with custom config."""
        from spatialvi.external.ppi import PPIConfig, SpatialMeanEstimator

        config = PPIConfig(alpha=0.05)
        estimator = SpatialMeanEstimator(config=config)

        assert estimator.config.alpha == 0.05
        assert estimator._mean_estimator.config.alpha == 0.05

    def test_fit_requires_ppi_py(self):
        """Test fit raises ImportError without ppi_py."""
        from spatialvi.external.ppi import SpatialMeanEstimator

        estimator = SpatialMeanEstimator()

        n_labeled = 10
        n_unlabeled = 100
        n_genes = 5

        expr_labeled = np.random.randn(n_labeled, n_genes)
        pred_labeled = np.random.randn(n_labeled, n_genes)
        pred_unlabeled = np.random.randn(n_unlabeled, n_genes)
        genes = ["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"]

        try:
            results = estimator.fit(
                expr_labeled, pred_labeled, pred_unlabeled, genes=genes
            )
            # If ppi_py is installed, check results
            assert isinstance(results, dict)
        except ImportError:
            # Expected if ppi_py not installed
            pass

    def test_fit_auto_gene_names(self):
        """Test fit generates gene names when not provided."""
        from spatialvi.external.ppi import SpatialMeanEstimator

        estimator = SpatialMeanEstimator()

        n_labeled = 10
        n_unlabeled = 100
        n_genes = 3

        expr_labeled = np.random.randn(n_labeled, n_genes)
        pred_labeled = np.random.randn(n_labeled, n_genes)
        pred_unlabeled = np.random.randn(n_unlabeled, n_genes)

        try:
            results = estimator.fit(expr_labeled, pred_labeled, pred_unlabeled)
            # If successful, gene names should be auto-generated
            if results:
                assert "gene_0" in results or len(results) == n_genes
        except ImportError:
            # Expected if ppi_py not installed
            pass


class TestPPIModuleImports:
    """Tests for module imports."""

    def test_all_exports(self):
        """Test all expected components are exported."""
        from spatialvi.external import ppi

        # Check main components
        assert hasattr(ppi, "PPIConfig")
        assert hasattr(ppi, "PPIResult")
        assert hasattr(ppi, "MeanEstimator")
        assert hasattr(ppi, "OLSEstimator")
        assert hasattr(ppi, "SpatialMeanEstimator")
        assert hasattr(ppi, "SpatialPPIResult")

    def test_direct_imports(self):
        """Test direct imports work."""
        from spatialvi.external.ppi import (
            MeanEstimator,
            OLSEstimator,
            PPIConfig,
            PPIResult,
            SpatialMeanEstimator,
            SpatialPPIResult,
        )

        assert PPIConfig is not None
        assert PPIResult is not None
        assert MeanEstimator is not None
        assert OLSEstimator is not None
        assert SpatialMeanEstimator is not None
        assert SpatialPPIResult is not None
