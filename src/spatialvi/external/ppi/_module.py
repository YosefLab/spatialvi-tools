"""PPI (Prediction-Powered Inference) module components.

Prediction-Powered Inference is a statistical framework for combining
machine learning predictions with a small labeled dataset to make
statistically valid inferences. Unlike neural network modules, this
provides statistical estimators and confidence intervals.

This module provides supporting components for PPI analysis including
estimator classes and result containers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class PPIConfig:
    """Configuration for PPI analysis.

    Parameters
    ----------
    alpha
        Error rate (confidence level is 1-alpha).
    method
        Estimation method to use.
    bootstrap_samples
        Number of bootstrap samples for variance estimation.
    use_classical
        Whether to also compute classical (non-PPI) estimates.
    """

    alpha: float = 0.1
    method: Literal["mean", "ols", "quantile", "logistic"] = "mean"
    bootstrap_samples: int = 1000
    use_classical: bool = True


@dataclass
class PPIResult:
    """Result of a PPI analysis.

    Parameters
    ----------
    estimate
        Point estimate of the parameter.
    ci_lower
        Lower bound of confidence interval.
    ci_upper
        Upper bound of confidence interval.
    se
        Standard error of the estimate.
    classical_estimate
        Classical (non-PPI) estimate for comparison.
    classical_ci
        Classical confidence interval.
    n_labeled
        Number of labeled samples used.
    n_unlabeled
        Number of unlabeled samples used.
    """

    estimate: float
    ci_lower: float
    ci_upper: float
    se: float = 0.0
    classical_estimate: float | None = None
    classical_ci: tuple[float, float] | None = None
    n_labeled: int = 0
    n_unlabeled: int = 0

    @property
    def ci(self) -> tuple[float, float]:
        """Get confidence interval as tuple."""
        return (self.ci_lower, self.ci_upper)

    @property
    def ci_width(self) -> float:
        """Get width of confidence interval."""
        return self.ci_upper - self.ci_lower

    @property
    def efficiency_gain(self) -> float | None:
        """Get efficiency gain compared to classical estimate."""
        if self.classical_ci is None:
            return None
        classical_width = self.classical_ci[1] - self.classical_ci[0]
        if classical_width == 0:
            return None
        return (classical_width - self.ci_width) / classical_width


class MeanEstimator:
    """PPI estimator for the population mean.

    This estimator uses prediction-powered inference to estimate
    the population mean with fewer labeled samples.

    Parameters
    ----------
    config
        PPI configuration.
    """

    def __init__(self, config: PPIConfig | None = None):
        self.config = config or PPIConfig()

    def fit(
        self,
        y_labeled: "NDArray",
        yhat_labeled: "NDArray",
        yhat_unlabeled: "NDArray",
    ) -> PPIResult:
        """Compute PPI estimate for the mean.

        Parameters
        ----------
        y_labeled
            True values for labeled samples.
        yhat_labeled
            Predictions for labeled samples.
        yhat_unlabeled
            Predictions for unlabeled samples.

        Returns
        -------
        PPIResult with estimate and confidence interval.
        """
        try:
            from ppi_py import ppi_mean_ci, ppi_mean_pointestimate
        except ImportError:
            raise ImportError(
                "The 'ppi_py' package is not installed. "
                "Install it with: pip install ppi-python"
            )

        y = np.asarray(y_labeled).ravel()
        yhat = np.asarray(yhat_labeled).ravel()
        yhat_unlab = np.asarray(yhat_unlabeled).ravel()

        # PPI estimate
        estimate = ppi_mean_pointestimate(y, yhat, yhat_unlab)
        ci = ppi_mean_ci(y, yhat, yhat_unlab, alpha=self.config.alpha)

        # Classical estimate (optional)
        classical_estimate = None
        classical_ci = None
        if self.config.use_classical:
            classical_estimate = np.mean(y)
            se = np.std(y) / np.sqrt(len(y))
            from scipy import stats

            z = stats.norm.ppf(1 - self.config.alpha / 2)
            classical_ci = (classical_estimate - z * se, classical_estimate + z * se)

        return PPIResult(
            estimate=float(estimate),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            n_labeled=len(y),
            n_unlabeled=len(yhat_unlab),
            classical_estimate=classical_estimate,
            classical_ci=classical_ci,
        )


class OLSEstimator:
    """PPI estimator for OLS regression coefficients.

    This estimator uses prediction-powered inference for
    ordinary least squares regression.

    Parameters
    ----------
    config
        PPI configuration.
    """

    def __init__(self, config: PPIConfig | None = None):
        self.config = config or PPIConfig()

    def fit(
        self,
        X_labeled: "NDArray",
        y_labeled: "NDArray",
        yhat_labeled: "NDArray",
        X_unlabeled: "NDArray",
        yhat_unlabeled: "NDArray",
    ) -> list[PPIResult]:
        """Compute PPI estimates for OLS coefficients.

        Parameters
        ----------
        X_labeled
            Features for labeled samples.
        y_labeled
            True values for labeled samples.
        yhat_labeled
            Predictions for labeled samples.
        X_unlabeled
            Features for unlabeled samples.
        yhat_unlabeled
            Predictions for unlabeled samples.

        Returns
        -------
        List of PPIResult, one per coefficient.
        """
        try:
            from ppi_py import ppi_ols_ci, ppi_ols_pointestimate
        except ImportError:
            raise ImportError(
                "The 'ppi_py' package is not installed. "
                "Install it with: pip install ppi-python"
            )

        X = np.asarray(X_labeled)
        y = np.asarray(y_labeled).ravel()
        yhat = np.asarray(yhat_labeled).ravel()
        X_unlab = np.asarray(X_unlabeled)
        yhat_unlab = np.asarray(yhat_unlabeled).ravel()

        estimates = ppi_ols_pointestimate(X, y, yhat, X_unlab, yhat_unlab)
        cis = ppi_ols_ci(X, y, yhat, X_unlab, yhat_unlab, alpha=self.config.alpha)

        results = []
        for i, (est, ci) in enumerate(zip(estimates, cis)):
            results.append(
                PPIResult(
                    estimate=float(est),
                    ci_lower=float(ci[0]),
                    ci_upper=float(ci[1]),
                    n_labeled=len(y),
                    n_unlabeled=len(yhat_unlab),
                )
            )

        return results


@dataclass
class SpatialPPIResult:
    """Result of spatial PPI analysis.

    Parameters
    ----------
    gene
        Gene name.
    spatial_region
        Spatial region identifier.
    mean_result
        PPI result for mean expression.
    proportion_result
        PPI result for proportion of expressing cells.
    """

    gene: str
    spatial_region: str
    mean_result: PPIResult
    proportion_result: PPIResult | None = None


class SpatialMeanEstimator:
    """PPI estimator for spatial transcriptomics.

    This estimator computes PPI estimates for gene expression
    within spatial regions.

    Parameters
    ----------
    config
        PPI configuration.
    """

    def __init__(self, config: PPIConfig | None = None):
        self.config = config or PPIConfig()
        self._mean_estimator = MeanEstimator(config)

    def fit(
        self,
        expression_labeled: "NDArray",
        predicted_labeled: "NDArray",
        predicted_unlabeled: "NDArray",
        genes: list[str] | None = None,
    ) -> dict[str, PPIResult]:
        """Compute PPI estimates for each gene.

        Parameters
        ----------
        expression_labeled
            True expression for labeled cells (cells x genes).
        predicted_labeled
            Predicted expression for labeled cells.
        predicted_unlabeled
            Predicted expression for unlabeled cells.
        genes
            Gene names.

        Returns
        -------
        Dictionary mapping gene names to PPIResult.
        """
        expr = np.asarray(expression_labeled)
        pred_lab = np.asarray(predicted_labeled)
        pred_unlab = np.asarray(predicted_unlabeled)

        n_genes = expr.shape[1]
        if genes is None:
            genes = [f"gene_{i}" for i in range(n_genes)]

        results = {}
        for i, gene in enumerate(genes):
            try:
                result = self._mean_estimator.fit(
                    expr[:, i],
                    pred_lab[:, i],
                    pred_unlab[:, i],
                )
                results[gene] = result
            except Exception as e:
                logger.warning(f"Could not compute PPI for gene {gene}: {e}")

        return results
