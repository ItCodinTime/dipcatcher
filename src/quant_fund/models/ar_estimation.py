"""Autoregressive parameter estimation: Yule-Walker, Levinson-Durbin, Burg.

For an AR(p) process ``x_t = sum_{k=1}^p phi_k x_{t-k} + e_t`` this module
provides three classical estimators:

- Yule-Walker: solve the sample normal equations ``R phi = r`` (Yule 1927;
  Walker 1931).
- Levinson-Durbin: the O(p^2) recursion that also returns the reflection
  (partial autocorrelation) coefficients and the error variance at each order
  (Levinson 1947; Durbin 1960).
- Burg: minimises the sum of forward and backward prediction errors, giving a
  stable, high-resolution estimator (Burg 1975).

Fail-closed on non-finite input, non-positive order, or too little history.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check(x: Array, p: int) -> Array:
    arr = np.asarray(x, dtype=float).ravel()
    if p < 1:
        raise ValueError("order p must be >= 1")
    if arr.size < p + 5 or not np.isfinite(arr).all():
        raise ValueError("series must be finite with enough observations for order p")
    return arr


def autocovariance(x: Array, maxlag: int) -> Array:
    """Biased sample autocovariances for lags ``0..maxlag``."""
    arr = np.asarray(x, dtype=float).ravel()
    n = arr.size
    if maxlag < 0 or maxlag >= n:
        raise ValueError("maxlag must be in [0, n)")
    d = arr - arr.mean()
    return np.array([float(np.dot(d[: n - k], d[k:]) / n) for k in range(maxlag + 1)])


def levinson_durbin(acov: Array, p: int) -> dict[str, Array | float]:
    """Levinson-Durbin recursion from autocovariances ``acov[0..p]``."""
    r = np.asarray(acov, dtype=float).ravel()
    if r.size < p + 1 or p < 1:
        raise ValueError("need acov of length >= p + 1")
    if r[0] <= 0.0:
        raise ValueError("acov[0] must be positive")
    a = np.zeros(p + 1)
    a[0] = 1.0
    e = float(r[0])
    reflection = np.zeros(p)
    phi = np.zeros(p)
    for k in range(1, p + 1):
        acc = r[k] - float(np.dot(phi[: k - 1], r[k - 1 : 0 : -1]))
        kref = acc / e
        reflection[k - 1] = kref
        new_phi = phi.copy()
        new_phi[k - 1] = kref
        for j in range(k - 1):
            new_phi[j] = phi[j] - kref * phi[k - 2 - j]
        phi = new_phi
        e *= 1.0 - kref**2
        if e <= 0.0:
            e = 1e-12
    return {"ar": phi[:p], "sigma2": float(e), "reflection": reflection}


def yule_walker(x: Array, p: int) -> dict[str, Array | float]:
    """Yule-Walker AR(p) estimation via the Levinson-Durbin recursion."""
    arr = _check(x, p)
    r = autocovariance(arr, p)
    out = levinson_durbin(r, p)
    return {"ar": out["ar"], "sigma2": out["sigma2"], "order": float(p)}


def burg(x: Array, p: int) -> dict[str, Array | float]:
    """Burg (1975) maximum-entropy AR(p) estimation."""
    arr = _check(x, p)
    f = arr.copy()
    b = arr.copy()
    a = np.zeros(p + 1)
    a[0] = 1.0
    e = float(np.dot(arr, arr) / arr.size)
    reflection = np.zeros(p)
    for m in range(p):
        fp = f[m + 1 :]
        bp = b[m:-1]
        denom = float(np.dot(fp, fp) + np.dot(bp, bp))
        if denom <= 0.0:
            break
        k = 2.0 * float(np.dot(fp, bp)) / denom
        reflection[m] = k
        a_prev = a.copy()
        for i in range(1, m + 2):
            a[i] = a_prev[i] - k * a_prev[m + 1 - i]
        f_new = fp - k * bp
        b_new = bp - k * fp
        f[m + 1 :] = f_new
        b[m:-1] = b_new
        e *= 1.0 - k**2
    ar = -a[1 : p + 1]
    return {"ar": ar, "sigma2": float(e), "reflection": reflection}
