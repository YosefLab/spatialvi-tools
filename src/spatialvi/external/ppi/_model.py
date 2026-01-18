"""Prediction-powered inference utilities.

The :class:`PPIInference` class provides static methods for computing
prediction-powered confidence intervals and point estimates using the
``ppi_py`` package. Prediction-powered inference allows researchers to
leverage machine learning predictions to reduce the number of labelled
examples required for statistical inference while maintaining rigorous
error control.

This is particularly useful in spatial transcriptomics when you have:
- A small number of cells with gold-standard labels
- A large number of cells with model predictions
- Want to make statistically valid inferences about population parameters

References
----------
Angelopoulos et al. (2023) "Prediction-Powered Inference"
https://arxiv.org/abs/2301.09633
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ppi_py import (
        ppi_mean_ci,
        ppi_mean_pointestimate,
        ppi_ols_ci,
        ppi_ols_pointestimate,
    )
except ImportError:
    ppi_mean_ci = None
    ppi_mean_pointestimate = None
    ppi_ols_ci = None
    ppi_ols_pointestimate = None


class PPIInference:
    """Static methods for prediction-powered inference.

    This class does not derive from the spatial model base because PPI
    operates on numerical arrays rather than AnnData objects. It exposes
    simple wrappers around commonly used estimands in the ``ppi_py``
    package, raising informative errors if the package is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from spatialvi.external import PPIInference
    >>> # Small labeled sample
    >>> y_labeled = np.array([1.2, 0.8, 1.1, 0.9, 1.0])
    >>> yhat_labeled = np.array([1.1, 0.9, 1.0, 0.85, 1.05])
    >>> # Large unlabeled sample with predictions
    >>> yhat_unlabeled = np.random.normal(1.0, 0.2, 1000)
    >>> # Compute confidence interval for mean
    >>> ci = PPIInference.mean_ci(y_labeled, yhat_labeled, yhat_unlabeled)
    >>> print(f"95% CI for mean: [{ci[0]:.3f}, {ci[1]:.3f}]")
    """

    @staticmethod
    def _require_ppi() -> None:
        if ppi_mean_ci is None:
            raise ImportError("The 'ppi_py' package is not installed. Install it with ``pip install ppi-python``.")

    @staticmethod
    def mean_ci(
        y: Iterable[float],
        yhat: Iterable[float],
        yhat_unlabeled: Iterable[float],
        alpha: float = 0.1,
    ) -> tuple[float, float]:
        """Compute a prediction-powered confidence interval for the mean.

        Parameters
        ----------
        y
            True labels for the labelled examples.
        yhat
            Predictions for the labelled examples.
        yhat_unlabeled
            Predictions for the unlabelled examples.
        alpha
            Error rate (the confidence level is 1-alpha).

        Returns
        -------
        ci_low, ci_high
            Lower and upper bounds of the confidence interval.
        """
        PPIInference._require_ppi()
        y_arr = np.asarray(y)
        yhat_arr = np.asarray(yhat)
        yhat_unlabeled_arr = np.asarray(yhat_unlabeled)
        return ppi_mean_ci(y_arr, yhat_arr, yhat_unlabeled_arr, alpha=alpha)

    @staticmethod
    def mean_pointestimate(
        y: Iterable[float],
        yhat: Iterable[float],
        yhat_unlabeled: Iterable[float],
    ) -> float:
        """Compute a prediction-powered point estimate for the mean.

        Parameters
        ----------
        y
            True labels for the labelled examples.
        yhat
            Predictions for the labelled examples.
        yhat_unlabeled
            Predictions for the unlabelled examples.

        Returns
        -------
        estimate
            The estimated mean of the outcome variable.
        """
        PPIInference._require_ppi()
        y_arr = np.asarray(y)
        yhat_arr = np.asarray(yhat)
        yhat_unlabeled_arr = np.asarray(yhat_unlabeled)
        return ppi_mean_pointestimate(y_arr, yhat_arr, yhat_unlabeled_arr)

    @staticmethod
    def ols_ci(
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        yhat: Iterable[float],
        X_unlabeled: Iterable[Iterable[float]],
        yhat_unlabeled: Iterable[float],
        alpha: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute prediction-powered CI for OLS regression coefficients.

        Parameters
        ----------
        X
            Labelled design matrix (n_l x p).
        y
            Labelled response vector.
        yhat
            Predictions for the labelled examples.
        X_unlabeled
            Unlabelled design matrix (n_u x p).
        yhat_unlabeled
            Predictions on the unlabelled data.
        alpha
            Error rate (confidence level is 1-alpha).

        Returns
        -------
        ci_low, ci_high
            Lower and upper bounds for each regression coefficient.
        """
        PPIInference._require_ppi()
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        yhat_arr = np.asarray(yhat)
        X_unlabeled_arr = np.asarray(X_unlabeled)
        yhat_unlabeled_arr = np.asarray(yhat_unlabeled)
        return ppi_ols_ci(X_arr, y_arr, yhat_arr, X_unlabeled_arr, yhat_unlabeled_arr, alpha=alpha)

    @staticmethod
    def ols_pointestimate(
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        yhat: Iterable[float],
        X_unlabeled: Iterable[Iterable[float]],
        yhat_unlabeled: Iterable[float],
    ) -> np.ndarray:
        """Compute prediction-powered point estimates for OLS coefficients.

        Parameters
        ----------
        X
            Labelled design matrix (n_l x p).
        y
            Labelled response vector.
        yhat
            Predictions for the labelled examples.
        X_unlabeled
            Unlabelled design matrix (n_u x p).
        yhat_unlabeled
            Predictions on the unlabelled data.

        Returns
        -------
        coefficients
            Point estimates for each regression coefficient.
        """
        PPIInference._require_ppi()
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        yhat_arr = np.asarray(yhat)
        X_unlabeled_arr = np.asarray(X_unlabeled)
        yhat_unlabeled_arr = np.asarray(yhat_unlabeled)
        return ppi_ols_pointestimate(X_arr, y_arr, yhat_arr, X_unlabeled_arr, yhat_unlabeled_arr)

    @staticmethod
    def classical_mean_ci(
        y: Iterable[float],
        alpha: float = 0.1,
    ) -> tuple[float, float]:
        """Compute classical confidence interval for the mean (no predictions).

        This is a utility method for comparison with PPI intervals.

        Parameters
        ----------
        y
            Sample values.
        alpha
            Error rate (confidence level is 1-alpha).

        Returns
        -------
        ci_low, ci_high
            Lower and upper bounds of the confidence interval.
        """
        from scipy import stats

        y_arr = np.asarray(y)
        n = len(y_arr)
        mean = np.mean(y_arr)
        se = np.std(y_arr, ddof=1) / np.sqrt(n)
        t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
        return (mean - t_val * se, mean + t_val * se)
