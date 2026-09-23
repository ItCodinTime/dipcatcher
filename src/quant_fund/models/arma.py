"""ARMA(p,q) estimation: conditional sum of squares + Hannan-Rissanen.

- ``arma_css``: minimize the conditional SSR (e_t = 0 for t <= maxlag)
  over (phi, theta) by least squares on the filter recursion.
- ``hannan_rissanen``: two-stage estimator -- long AR(m) pre-whitens to
  residuals, then OLS of y_t on its own lags and lagged residuals
  (Hannan & Rissanen 1982).
- ``arma_select``: AIC grid search over (p, q) using CSS.

Fail-closed: negative orders, non-finite input, insufficient data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

Array = NDArray[np.float64]


def _css_residuals(y: Array, phi: Array, theta: Array) -> Array:
    """Conditional residuals with e_t = 0 for t <= max(p, q)."""
    n = y.size
    p, q = phi.size, theta.size
    m = max(p, q)
    e = np.zeros(n)
    for t in range(m, n):
        ar = float(np.dot(phi, y[t - p : t][::-1])) if p else 0.0
        ma = float(np.dot(theta, e[t - q : t][::-1])) if q else 0.0
        e[t] = y[t] - ar - ma
    return e


def arma_css(y: Array, p: int, q: int) -> dict[str, Array | float]:
    """ARMA(p,q) by conditional sum of squares. Includes intercept c."""
    yy = np.asarray(y, dtype=float).ravel()
    n = yy.size
    m = max(p, q)
    if p < 0 or q < 0 or p + q == 0:
        raise ValueError("need p + q >= 1 with nonnegative orders")
    if n < 4 * (p + q) + 20 or not np.isfinite(yy).all():
        raise ValueError("insufficient or non-finite data")
    mu = float(yy.mean())
    yd = yy - mu

    def ssr(par: Array) -> float:
        phi = par[:p]
        theta = par[p : p + q]
        e = _css_residuals(yd, phi, theta)
        if not np.isfinite(e).all():
            return 1e14
        return float(e[m:] @ e[m:])

    par0 = np.zeros(p + q)
    res = optimize.minimize(ssr, par0, method="BFGS")
    phi = res.x[:p]
    theta = res.x[p : p + q]
    e = _css_residuals(yd, phi, theta)[m:]
    dof = n - m - p - q - 1
    sigma2 = float(e @ e / max(dof, 1))
    aic = float((n - m) * np.log(max(sigma2, 1e-12)) + 2 * (p + q + 1))
    return {
        "phi": phi,
        "theta": theta,
        "mu": mu,
        "sigma2": sigma2,
        "aic": aic,
        "resid": e,
    }


def hannan_rissanen(
    y: Array, p: int, q: int, ar_order: int | None = None
) -> dict[str, Array | float]:
    """Two-stage Hannan-Rissanen ARMA(p,q) estimator."""
    yy = np.asarray(y, dtype=float).ravel()
    n = yy.size
    if p + q == 0 or p < 0 or q < 0:
        raise ValueError("need p + q >= 1")
    if ar_order is None:
        ar_order = int(min(max(p + q + 5, int(np.log(n) ** 1.5)), n // 3))
    if n < ar_order + 2 * (p + q) + 10 or not np.isfinite(yy).all():
        raise ValueError("insufficient or non-finite data")
    yd = yy - yy.mean()
    # stage 1: long AR(ar_order)
    x_ar = np.column_stack([yd[ar_order - 1 - i : n - 1 - i] for i in range(ar_order)])
    y_ar = yd[ar_order:]
    ar_coef = np.linalg.lstsq(x_ar, y_ar, rcond=None)[0]
    e_hat = y_ar - x_ar @ ar_coef  # residuals aligned with t = ar_order..n-1
    # stage 2: y_t on y-lags and e-lags
    start = ar_order + max(p, q) - p  # earliest usable t index offset
    rows_x = []
    rows_y = []
    for t in range(ar_order + q, n):
        row = []
        for i in range(1, p + 1):
            row.append(yd[t - i])
        for j in range(1, q + 1):
            row.append(e_hat[t - j - ar_order])
        rows_x.append(row)
        rows_y.append(yd[t])
    x2 = np.asarray(rows_x)
    y2 = np.asarray(rows_y)
    coef = np.linalg.lstsq(x2, y2, rcond=None)[0]
    phi = coef[:p]
    theta = coef[p : p + q]
    resid = y2 - x2 @ coef
    sigma2 = float(resid @ resid / resid.size)
    return {
        "phi": phi,
        "theta": theta,
        "sigma2": sigma2,
        "ar_order": float(ar_order),
        "resid": resid,
        "start": float(start),
    }


def arma_select(y: Array, max_p: int = 3, max_q: int = 3) -> dict[str, float]:
    """AIC grid selection over ARMA(p, q). Returns best orders + AIC table."""
    yy = np.asarray(y, dtype=float).ravel()
    best: dict[str, float] = {"p": -1.0, "q": -1.0, "aic": np.inf}
    table = np.full((max_p + 1, max_q + 1), np.nan)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p + q == 0:
                sigma2 = float(yy.var())
                aic = yy.size * np.log(max(sigma2, 1e-12)) + 2.0
            else:
                try:
                    aic = float(arma_css(yy, p, q)["aic"])
                except (ValueError, np.linalg.LinAlgError):
                    continue
            table[p, q] = aic
            if aic < float(best["aic"]):
                best = {"p": float(p), "q": float(q), "aic": aic}
    best["aic_table"] = table  # type: ignore[assignment]
    return best
