"""Forecast-combination methods.

References:
- Bates, Granger (1969). The combination of forecasts.  *Operations
  Research Quarterly* 20 — inverse-MSE weighting and covariance weighting.
- Granger, Ramanathan (1984). Improved methods of combining forecasts.
  *Journal of Forecasting* 3 — regression-based combination.
- Diebold, Mariano / standard practice: simple averages are hard to beat —
  implemented as the ``equal`` baseline and the median/trimmed variants.
- Stock, Watson (2004). Combination forecasts of output growth — empirical
  horserace evidence for trimmed/median combinations.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt

Array = NDArray[np.float64]


def _as_forecasts(f: Array, name: str = "forecasts") -> Array:
    m = np.asarray(f, dtype=float)
    if m.ndim != 2 or not np.all(np.isfinite(m)):
        raise ValueError(f"{name} must be a finite T x K matrix")
    if m.shape[0] < 5 or m.shape[1] < 2:
        raise ValueError(f"{name} must have >= 5 rows and >= 2 forecasts")
    return m


def _as_target(y: Array, n: int) -> Array:
    v = np.asarray(y, dtype=float).reshape(-1)
    if v.size != n or not np.all(np.isfinite(v)):
        raise ValueError("y must be a finite vector matching forecast rows")
    return v


def equal_weights(n_models: int) -> Array:
    """Uniform combination weights (the forecast-combination benchmark)."""
    if isinstance(n_models, bool) or not isinstance(n_models, int) or n_models < 1:
        raise ValueError("n_models must be a positive integer")
    return np.full(n_models, 1.0 / n_models)


def inverse_mse_weights(forecasts: Array, y: Array) -> Array:
    """Bates–Granger inverse-MSE weights ``w_k = (1/mse_k) / sum_j(1/mse_j)``."""
    f = _as_forecasts(forecasts)
    v = _as_target(y, f.shape[0])
    mse = np.mean((f - v[:, None]) ** 2, axis=0)
    if np.any(mse <= 0.0):
        raise ValueError("a forecast has zero MSE — perfect fit, cannot weight")
    iv = 1.0 / mse
    return iv / iv.sum()


def covariance_weights(forecasts: Array, y: Array) -> Array:
    """Bates–Granger covariance-optimal weights ``w = Sigma_e^{-1} 1 / (1' Sigma_e^{-1} 1)``.

    Uses the full error covariance, so correlated forecast errors are
    down-weighted.  Falls back to inverse-MSE when Sigma_e is singular.
    """
    f = _as_forecasts(forecasts)
    v = _as_target(y, f.shape[0])
    e = f - v[:, None]
    sigma_e = np.cov(e.T)
    if sigma_e.ndim == 0:
        sigma_e = sigma_e.reshape(1, 1)
    ones = np.ones(f.shape[1])
    try:
        inv = np.linalg.pinv(sigma_e + np.eye(f.shape[1]) * 1e-12)
        w = inv @ ones
        s = float(ones @ w)
        if not np.isfinite(s) or abs(s) < 1e-12:
            raise ValueError("degenerate")
        w = w / s
    except (np.linalg.LinAlgError, ValueError):
        w = inverse_mse_weights(f, v)
    return w


def granger_ramanathan(
    forecasts: Array, y: Array, *, constrain: bool = True
) -> tuple[Array, float]:
    """Granger–Ramanathan (1984) regression weights.

    Fits ``y = a + F w``; with ``constrain=True`` solves the restricted
    regression ``w >= 0, sum w = 1`` (GR variant B — weights as shares,
    intercept kept).  Returns ``(weights, intercept)``.
    """
    f = _as_forecasts(forecasts)
    v = _as_target(y, f.shape[0])
    t, k = f.shape
    if not constrain:
        x = np.column_stack([np.ones(t), f])
        coef, *_ = np.linalg.lstsq(x, v, rcond=None)
        return coef[1:], float(coef[0])

    # Constrained: minimize ||y - a - Fw|| s.t. w>=0, sum w = 1.
    def _obj(w: Array) -> float:
        a = float(np.mean(v - f @ w))
        r = v - a - f @ w
        return float(r @ r)

    res = opt.minimize(
        _obj,
        np.full(k, 1.0 / k),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
        options={"maxiter": 400, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    w = w / s if s > 0.0 else np.full(k, 1.0 / k)
    a = float(np.mean(v - f @ w))
    return w, a


def median_combination(forecasts: Array) -> Array:
    """Pointwise median across the K forecasts (robust combination)."""
    f = _as_forecasts(forecasts)
    return np.median(f, axis=1)


def trimmed_mean_combination(forecasts: Array, trim: float = 0.1) -> Array:
    """Pointwise trimmed mean across forecasts (drops extreme fraction)."""
    f = _as_forecasts(forecasts)
    if not np.isfinite(trim) or not (0.0 <= trim < 0.5):
        raise ValueError("trim must be in [0, 0.5)")
    k_drop = int(math.floor(trim * f.shape[1]))
    if k_drop == 0:
        return f.mean(axis=1)
    s = np.sort(f, axis=1)
    return s[:, k_drop : f.shape[1] - k_drop].mean(axis=1)


def combine(forecasts: Array, weights: Array) -> Array:
    """Apply combination weights: ``f_combined = F @ w``."""
    f = _as_forecasts(forecasts)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != f.shape[1] or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a finite vector matching forecasts")
    return f @ w


def rolling_inverse_mse(forecasts: Array, y: Array, window: int = 60) -> Array:
    """Expanding/windowed Bates–Granger weights, causal (row t uses < t).

    Returns a T x K weight matrix; rows before ``window`` are equal weights.
    """
    f = _as_forecasts(forecasts)
    v = _as_target(y, f.shape[0])
    t, k = f.shape
    if isinstance(window, bool) or not isinstance(window, int) or window < 5:
        raise ValueError("window must be an integer >= 5")
    out = np.full((t, k), 1.0 / k)
    for i in range(window, t):
        lo = max(0, i - window)
        mse = np.mean((f[lo:i] - v[lo:i, None]) ** 2, axis=0)
        mse = np.maximum(mse, 1e-18)
        iv = 1.0 / mse
        out[i] = iv / iv.sum()
    return out
