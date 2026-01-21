"""Prediction‑powered inference utilities.

The :class:`PPIInference` class provides static methods for computing
prediction‑powered confidence intervals and point estimates using the
``ppi_py`` package【569057324233913†L11-L14】.  Prediction‑powered inference allows
researchers to leverage machine learning predictions to reduce the number
of labelled examples required for statistical inference while maintaining
rigorous error control【569057324233913†L11-L14】.
"""

from __future__ import annotations

from typing import Iterable, Tuple

try:
    # Import the relevant functions from ppi_py
    from ppi_py import (
        ppi_mean_ci,
        ppi_mean_pointestimate,
        ppi_linear_regression_ci,
        ppi_linear_regression_pointestimate,
    )  # type: ignore
except ImportError:  # pragma: no cover
    ppi_mean_ci = None  # type: ignore
    ppi_mean_pointestimate = None  # type: ignore
    ppi_linear_regression_ci = None  # type: ignore
    ppi_linear_regression_pointestimate = None  # type: ignore


class PPIInference:
    """Static methods for prediction‑powered inference.

    This class does not derive from :class:`BaseSpatialModel` because PPI
    operates on numerical arrays rather than AnnData objects.  It exposes
    simple wrappers around commonly used estimands in the ``ppi_py``
    package, raising informative errors if the package is not installed.
    """

    @staticmethod
    def mean_ci(
        y: Iterable[float],
        yhat: Iterable[float],
        yhat_unlabeled: Iterable[float],
        alpha: float = 0.1,
    ) -> Tuple[float, float]:
        """Compute a prediction‑powered confidence interval for the mean.

        Parameters
        ----------
        y:
            True labels for the labelled examples.
        yhat:
            Predictions for the labelled examples.
        yhat_unlabeled:
            Predictions for the unlabelled examples.
        alpha:
            Error rate (the confidence level is 1‑alpha).

        Returns
        -------
        ci_low, ci_high:
            Lower and upper bounds of the confidence interval.
        """
        if ppi_mean_ci is None:
            raise ImportError(
                "The 'ppi_py' package is not installed. Install it with ``pip install ppi-python``."
            )
        return ppi_mean_ci(y, yhat, yhat_unlabeled, alpha=alpha)

    @staticmethod
    def mean_pointestimate(
        y: Iterable[float],
        yhat: Iterable[float],
        yhat_unlabeled: Iterable[float],
    ) -> float:
        """Compute a prediction‑powered point estimate for the mean.

        Returns
        -------
        estimate:
            The estimated mean of the outcome variable.
        """
        if ppi_mean_pointestimate is None:
            raise ImportError(
                "The 'ppi_py' package is not installed. Install it with ``pip install ppi-python``."
            )
        return ppi_mean_pointestimate(y, yhat, yhat_unlabeled)

    @staticmethod
    def linear_regression_ci(
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        Xhat: Iterable[Iterable[float]],
        Xhat_unlabeled: Iterable[Iterable[float]],
        yhat_unlabeled: Iterable[float],
        alpha: float = 0.1,
    ) -> Tuple[Iterable[float], Iterable[float]]:
        """Compute a prediction‑powered confidence interval for linear regression coefficients.

        Parameters
        ----------
        X:
            Labelled design matrix (n_l x p).
        y:
            Labelled response vector.
        Xhat:
            Predictions of the response on the labelled data (typically not
            required for linear regression in ppi_py but included for symmetry).
        Xhat_unlabeled:
            Predictions of the response on the unlabelled data.
        yhat_unlabeled:
            Predictions of the outcome on the unlabelled data.
        alpha:
            Error rate.

        Returns
        -------
        ci_low, ci_high:
            Lower and upper bounds for each regression coefficient.
        """
        if ppi_linear_regression_ci is None:
            raise ImportError(
                "The 'ppi_py' package is not installed. Install it with ``pip install ppi-python``."
            )
        return ppi_linear_regression_ci(X, y, Xhat, Xhat_unlabeled, yhat_unlabeled, alpha=alpha)

    @staticmethod
    def linear_regression_pointestimate(
        X: Iterable[Iterable[float]],
        y: Iterable[float],
        Xhat: Iterable[Iterable[float]],
        Xhat_unlabeled: Iterable[Iterable[float]],
        yhat_unlabeled: Iterable[float],
    ) -> Iterable[float]:
        """Compute prediction‑powered point estimates for linear regression coefficients.
        """
        if ppi_linear_regression_pointestimate is None:
            raise ImportError(
                "The 'ppi_py' package is not installed. Install it with ``pip install ppi-python``."
            )
        return ppi_linear_regression_pointestimate(X, y, Xhat, Xhat_unlabeled, yhat_unlabeled)