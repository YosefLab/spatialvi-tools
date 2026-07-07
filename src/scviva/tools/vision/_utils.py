"""Small shared statistics helper.

Adapted from visionpy (MIT licence, https://github.com/yoseflab/visionpy).

Kept in its own module (rather than inlined into ``_normalization.py``) to
avoid a circular import: ``filters.py`` needs ``_get_mean_var`` but
``_normalization.py`` imports ``log2p1`` from ``projections.py``, which in
turn imports ``apply_filters`` from ``filters.py``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _get_mean_var(X: np.ndarray | sparse.spmatrix, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance of *X* along *axis*, supporting sparse input.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Dense or sparse expression matrix.
    axis : int, optional
        Axis along which to compute the statistics, by default 0.

    Returns
    -------
    mean : np.ndarray
        Mean along *axis*.
    var : np.ndarray
        Variance along *axis*.
    """
    if sparse.issparse(X):
        mean = np.array(X.mean(axis=axis)).ravel()
        var = np.array(X.power(2).mean(axis=axis)).ravel() - mean**2
    else:
        mean = np.mean(X, axis=axis)
        var = np.var(X, axis=axis, ddof=1)
    return mean, var
