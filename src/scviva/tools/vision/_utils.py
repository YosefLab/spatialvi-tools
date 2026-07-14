"""Small shared statistics/array helpers.

Adapted from visionpy (MIT licence, https://github.com/yoseflab/visionpy).

Kept in their own module (rather than inlined into ``preprocessing/filters.py``
or ``preprocessing/normalization.py``) to avoid circular imports:
``preprocessing/filters.py`` needs ``_get_mean_var`` but
``preprocessing/normalization.py`` needs ``log2p1``, which
``tools/projections.py`` also needs, and ``tools/projections.py`` in turn
imports ``apply_filters`` from ``preprocessing/filters.py``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd


def log2p1(
    X: np.ndarray | sparse.spmatrix,
) -> np.ndarray | sparse.spmatrix:
    """Sparse-preserving log2(x + 1) transform.

    For sparse matrices only the stored (non-zero) values are transformed,
    keeping the sparsity pattern intact.  This is valid because
    ``log2(0 + 1) == 0``.

    Parameters
    ----------
    X : array-like of shape (n_cells, n_genes)
        Expression matrix, cells × genes.  Not modified in-place.

    Returns
    -------
    array-like of shape (n_cells, n_genes)
        Log-transformed matrix in the same format as the input.
    """
    if sparse.issparse(X):
        X = X.copy()
        X.data = np.log2(X.data + 1)
    else:
        X = np.log2(np.asarray(X, dtype=float) + 1)
    return X


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
        n = X.shape[axis]
        mean = np.array(X.mean(axis=axis)).ravel()
        # Population variance (E[X^2] - mean^2), Bessel-corrected to sample
        # variance (ddof=1) to match the dense path below.
        var = np.array(X.power(2).mean(axis=axis)).ravel() - mean**2
        if n > 1:
            var *= n / (n - 1)
    else:
        mean = np.mean(X, axis=axis)
        var = np.var(X, axis=axis, ddof=1)
    return mean, var


def gene_centered_svd(
    X: np.ndarray, n_components: int, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gene-mean-center a dense (cells x genes) matrix and run randomized SVD.

    Shared core of :func:`~scviva.tools.vision.tools.projections.apply_pca`
    and micro-clustering's internal PCA — both center by per-gene means
    before a truncated randomized SVD.

    Parameters
    ----------
    X : ndarray of shape (n_cells, n_genes)
        Dense expression matrix.
    n_components : int
        Number of singular components to compute.
    random_state : int, optional
        Seed for :func:`sklearn.utils.extmath.randomized_svd`, by default 0.

    Returns
    -------
    U : ndarray of shape (n_cells, n_components)
    S : ndarray of shape (n_components,)
    Vt : ndarray of shape (n_components, n_genes)
    """
    mu = X.mean(axis=0)
    X_c = X - mu
    U, S, Vt = randomized_svd(X_c, n_components=n_components, random_state=random_state)
    return U, S, Vt
