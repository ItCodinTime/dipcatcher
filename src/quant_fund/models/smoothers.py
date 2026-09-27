"""Nonparametric regression smoothers.

- ``nadaraya_watson``: local-constant kernel regression
- ``local_linear``: local-polynomial (degree 1) kernel regression
  (Fan & Gijbels) -- corrects NW boundary bias
- ``smoothing_spline``: cubic Reinsch smoothing spline via the
  (n x n) hat matrix S = (I + lam K)^{-1} on a uniform design;
  also provides a GCV criterion for bandwidth/lambda choice

Gaussian and Epanechnikov kernels supported. Fail-closed: non-finite
data, h <= 0, mismatched lengths, fewer than 3 distinct design points.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _kern(u: Array, kernel: str) -> Array:
    if kernel == "gauss":
        return np.asarray(np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi), dtype=float)
    if kernel == "epa":
        return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)
    raise ValueError("kernel must be gauss|epa")


def _check(x: Array, y: Array) -> tuple[Array, Array]:
    xx = np.asarray(x, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    if xx.size != yy.size or xx.size < 5:
        raise ValueError("x, y must match with >= 5 obs")
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("non-finite input")
    if np.unique(xx).size < 3:
        raise ValueError("need >= 3 distinct design points")
    return xx, yy


def nadaraya_watson(
    x: Array, y: Array, h: float, grid: Array | None = None, kernel: str = "gauss"
) -> Array:
    """NW estimate m(x) = sum K((x-x_i)/h) y_i / sum K(...)."""
    xx, yy = _check(x, y)
    if h <= 0:
        raise ValueError("h must be positive")
    gx = np.asarray(grid, dtype=float) if grid is not None else xx
    out = np.empty(gx.size)
    for i, g in enumerate(gx):
        w = _kern((g - xx) / h, kernel)
        ws = w.sum()
        out[i] = float(w @ yy / ws) if ws > 1e-14 else np.nan
    return out


def local_linear(
    x: Array, y: Array, h: float, grid: Array | None = None, kernel: str = "gauss"
) -> dict[str, Array]:
    """Local-linear fit: returns fitted values and slope estimates."""
    xx, yy = _check(x, y)
    if h <= 0:
        raise ValueError("h must be positive")
    gx = np.asarray(grid, dtype=float) if grid is not None else xx
    fit = np.empty(gx.size)
    slope = np.empty(gx.size)
    for i, g in enumerate(gx):
        u = (xx - g) / h
        w = _kern(u, kernel)
        wmat = w[:, None] * np.column_stack([np.ones(xx.size), xx - g])
        xtw = wmat.T @ np.column_stack([np.ones(xx.size), xx - g])
        xty = wmat.T @ yy
        try:
            b = np.linalg.solve(xtw + 1e-10 * np.eye(2), xty)
        except np.linalg.LinAlgError:
            b = np.array([np.nan, np.nan])
        fit[i], slope[i] = float(b[0]), float(b[1] / 1.0)
    return {"fit": fit, "slope": slope, "grid": gx}


def smoothing_spline(x: Array, y: Array, lam: float) -> dict[str, Array | float]:
    """Cubic smoothing spline on the observed (sorted) design.

    Solves min sum (y_i - f_i)^2 + lam * f' K f via the closed form
    f = (I + lam K)^{-1} y, where K is the standard Reinsch penalty
    matrix built from the second-difference operator weighted by the
    design spacings.
    """
    xx, yy = _check(x, y)
    if lam < 0:
        raise ValueError("lam must be >= 0")
    order = np.argsort(xx)
    xs, ys = xx[order], yy[order]
    n = xs.size
    # Reinsch: K = Q' W^{-1} Q with Q the (n-2, n) second-diff operator
    # scaled by successive gaps; W tridiagonal of gap sums.
    hvec = np.diff(xs)
    if (hvec <= 0).any():
        raise ValueError("x must contain distinct values")
    qm = np.zeros((n - 2, n))
    wm = np.zeros((n - 2, n - 2))
    for i in range(n - 2):
        h1, h2 = hvec[i], hvec[i + 1]
        qm[i, i] = 1.0 / h1
        qm[i, i + 1] = -1.0 / h1 - 1.0 / h2
        qm[i, i + 2] = 1.0 / h2
        wm[i, i] = (h1 + h2) / 3.0
        if i + 1 < n - 2:
            wm[i, i + 1] = wm[i + 1, i] = h2 / 6.0
    kmat = qm.T @ np.linalg.solve(wm, qm)
    smat = np.linalg.inv(np.eye(n) + lam * kmat)
    f = smat @ ys
    edf = float(np.trace(smat))
    return {"fit_sorted": f, "x_sorted": xs, "edf": edf, "hat": smat}


def gcv_score(x: Array, y: Array, lam: float) -> float:
    """Generalized cross-validation score for ``smoothing_spline``."""
    xx, yy = _check(x, y)
    out = smoothing_spline(xx, yy, lam)
    f = np.asarray(out["fit_sorted"])
    ys = yy[np.argsort(xx)]
    n = xx.size
    edf = float(out["edf"])
    rss = float(np.sum((ys - f) ** 2)) / n
    denom = (1.0 - edf / n) ** 2
    return float(rss / max(denom, 1e-12))
