"""PCA and ZCA whitening transforms.

Whitening linearly transforms data to have identity covariance.  Given the
centred covariance ``Sigma = U Lambda U'``:

- **PCA whitening** uses ``W = Lambda^{-1/2} U'`` (decorrelates then scales).
- **ZCA whitening** uses the symmetric ``W = U Lambda^{-1/2} U'``, the whitening
  transform closest to the identity (keeps whitened data maximally similar to
  the original).

Reference: A. Kessy, A. Lewin, K. Strimmer (2018), "Optimal whitening and
decorrelation", The American Statistician.  Fail-closed on non-finite input or
degenerate shapes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check(x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < arr.shape[1] + 2 or not np.isfinite(arr).all():
        raise ValueError("x must be a finite (n, d) matrix with n >= d + 2")
    return arr


def _eig(arr: Array, eps: float) -> tuple[Array, Array, Array]:
    mean = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    return mean, vals, vecs


def pca_whiten(x: Array, eps: float = 1e-8) -> dict[str, Array]:
    """PCA whitening; returns whitened data and the whitening matrix."""
    arr = _check(x)
    mean, vals, vecs = _eig(arr, eps)
    w = (vecs / np.sqrt(vals)).T  # Lambda^{-1/2} U'
    return {"whitened": (arr - mean) @ w.T, "matrix": w, "mean": mean}


def zca_whiten(x: Array, eps: float = 1e-8) -> dict[str, Array]:
    """ZCA (symmetric) whitening; returns whitened data and the whitening matrix."""
    arr = _check(x)
    mean, vals, vecs = _eig(arr, eps)
    w = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return {"whitened": (arr - mean) @ w.T, "matrix": w, "mean": mean}
