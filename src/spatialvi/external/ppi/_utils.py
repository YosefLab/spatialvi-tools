"""Utility functions for Prediction-Powered Inference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_rectifier(
    y: NDArray,
    yhat: NDArray,
    yhat_unlabeled: NDArray,
    alpha: float = 0.1,
) -> float:
    """Compute rectifier for PPI.

    Parameters
    ----------
    y
        Labeled outcomes.
    yhat
        Predictions for labeled data.
    yhat_unlabeled
        Predictions for unlabeled data.
    alpha
        Significance level.

    Returns
    -------
    Optimal rectifier value.
    """
    n = len(y)
    N = len(yhat_unlabeled)

    # Estimate bias
    bias = np.mean(yhat) - np.mean(y)

    # Estimate variance reduction
    var_yhat = np.var(yhat_unlabeled)
    cov_y_yhat = np.cov(y, yhat)[0, 1]

    # Optimal lambda
    if var_yhat > 0:
        lambda_opt = cov_y_yhat / var_yhat
    else:
        lambda_opt = 0

    return lambda_opt


def compute_variance_reduction(
    y: NDArray,
    yhat: NDArray,
    yhat_unlabeled: NDArray,
) -> float:
    """Compute variance reduction factor from PPI.

    Parameters
    ----------
    y
        Labeled outcomes.
    yhat
        Predictions for labeled data.
    yhat_unlabeled
        Predictions for unlabeled data.

    Returns
    -------
    Variance reduction factor (0 to 1, lower is better).
    """
    n = len(y)
    N = len(yhat_unlabeled)

    # Classical variance
    var_classical = np.var(y) / n

    # PPI variance
    residuals = y - yhat
    var_residual = np.var(residuals) / n
    var_prediction = np.var(yhat_unlabeled) / N

    # Total PPI variance
    var_ppi = var_residual + var_prediction

    if var_classical > 0:
        return var_ppi / var_classical
    return 1.0


def stratified_sample(
    labels: NDArray,
    n_labeled: int,
    seed: int | None = None,
) -> NDArray:
    """Generate stratified sample indices.

    Parameters
    ----------
    labels
        Array of labels/strata.
    n_labeled
        Total number of labeled samples.
    seed
        Random seed.

    Returns
    -------
    Indices of selected samples.
    """
    if seed is not None:
        np.random.seed(seed)

    unique_labels = np.unique(labels)
    n_per_stratum = n_labeled // len(unique_labels)

    selected = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        n_select = min(n_per_stratum, len(indices))
        selected.extend(np.random.choice(indices, size=n_select, replace=False))

    return np.array(selected)


def compute_power_analysis(
    effect_size: float,
    var_y: float,
    var_yhat: float,
    corr_y_yhat: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict[str, int]:
    """Compute sample size requirements for PPI vs classical.

    Parameters
    ----------
    effect_size
        Expected effect size.
    var_y
        Variance of outcome.
    var_yhat
        Variance of predictions.
    corr_y_yhat
        Correlation between y and yhat.
    alpha
        Significance level.
    power
        Desired power.

    Returns
    -------
    Dictionary with sample sizes for classical and PPI.
    """
    from scipy import stats

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Classical sample size
    n_classical = int(np.ceil((z_alpha + z_beta) ** 2 * var_y / effect_size ** 2))

    # PPI sample size (accounting for variance reduction)
    var_reduction = 1 - corr_y_yhat ** 2
    n_ppi = int(np.ceil((z_alpha + z_beta) ** 2 * var_y * var_reduction / effect_size ** 2))

    return {
        "n_classical": n_classical,
        "n_ppi": n_ppi,
        "reduction_factor": n_ppi / n_classical if n_classical > 0 else 1.0,
    }


def bootstrap_ci(
    y: NDArray,
    yhat: NDArray,
    yhat_unlabeled: NDArray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for PPI estimate.

    Parameters
    ----------
    y
        Labeled outcomes.
    yhat
        Predictions for labeled data.
    yhat_unlabeled
        Predictions for unlabeled data.
    n_bootstrap
        Number of bootstrap samples.
    alpha
        Significance level.
    seed
        Random seed.

    Returns
    -------
    Tuple of (lower, upper) confidence bounds.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(y)
    N = len(yhat_unlabeled)

    estimates = []
    for _ in range(n_bootstrap):
        # Bootstrap labeled data
        idx_labeled = np.random.choice(n, size=n, replace=True)
        y_boot = y[idx_labeled]
        yhat_boot = yhat[idx_labeled]

        # Bootstrap unlabeled predictions
        idx_unlabeled = np.random.choice(N, size=N, replace=True)
        yhat_unlabeled_boot = yhat_unlabeled[idx_unlabeled]

        # PPI estimate
        rectifier = compute_rectifier(y_boot, yhat_boot, yhat_unlabeled_boot)
        estimate = np.mean(yhat_unlabeled_boot) + rectifier * (np.mean(y_boot) - np.mean(yhat_boot))
        estimates.append(estimate)

    lower = np.percentile(estimates, 100 * alpha / 2)
    upper = np.percentile(estimates, 100 * (1 - alpha / 2))

    return lower, upper
