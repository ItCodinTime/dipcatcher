"""Mixed-frequency nowcasting: MIDAS and bridge equations.

References:
- Ghysels, Sinko & Valkanov (2007): MIDAS regressions with beta
  polynomial weights.
- Almon (1965): polynomial distributed lags.
- Baffigi, Golinelli & Parigi (2004): bridge equations for nowcasting.
- Clements & Galvao (2008): MIDAS with leads.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt

Array = NDArray[np.float64]


def _v(x: Array, n: int = 20) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"series must be finite with length >= {n}")
    return v


def beta_weights(k: int, theta1: float, theta2: float) -> Array:
    """Ghysels two-parameter beta polynomial weights over ``k`` lags.

    ``w_j = f(j/k; t1, t2) / sum f``, f = x^{t1-1} (1-x)^{t2-1}."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if theta1 <= 0 or theta2 <= 0:
        raise ValueError("beta shape parameters must be positive")
    x = (np.arange(1, k + 1) - 0.5) / k
    logw = (theta1 - 1.0) * np.log(np.maximum(x, 1e-12)) + (theta2 - 1.0) * np.log(
        np.maximum(1.0 - x, 1e-12)
    )
    w = np.exp(logw - logw.max())
    s = w.sum()
    if s <= 0 or not np.all(np.isfinite(w)):
        raise ValueError("degenerate beta weights")
    return np.asarray(w / s, dtype=float)


def fit_midas(
    y_low: Array,
    X_high: Array,
    k_lags: int,
    ar_lag: bool = True,
    max_iter: int = 150,
) -> dict[str, Array | float]:
    """MIDAS-beta regression (Ghysels–Sinko–Valkanov 2007).

    ``y_t = a + b * sum_j w_j(theta) x_{t-j} [+ c*y_{t-1}] + e_t``.
    ``X_high`` is (T, k_lags) — already-aligned high-frequency lags
    (caller aligns; we do NOT shift inside, preserving causality).
    Estimates (a, b, theta1, theta2 [, c]) by nonlinear LS.
    """
    y = _v(y_low)
    X = np.asarray(X_high, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.size or X.shape[1] != k_lags:
        raise ValueError("X_high must be (T, k_lags) aligned to y")
    if not np.all(np.isfinite(X)) or k_lags < 1:
        raise ValueError("X_high must be finite")
    n = y.size
    ylag = np.concatenate([[np.nan], y[:-1]])
    rows = np.isfinite(ylag) if ar_lag else np.ones(n, dtype=bool)
    yr = y[rows]
    Xr = X[rows]
    yl = ylag[rows]

    def unpack(theta: Array):
        if ar_lag:
            a, b, t1, t2, c = theta
        else:
            a, b, t1, t2 = theta
            c = 0.0
        return a, b, np.exp(t1), np.exp(t2), c

    def sse(theta: Array) -> float:
        a, b, t1, t2, c = unpack(theta)
        if t1 > 20 or t2 > 20:
            return 1e12
        try:
            w = beta_weights(k_lags, t1, t2)
        except ValueError:
            return 1e12
        mid = Xr @ w
        e = yr - (a + b * mid + c * yl)
        out = float(e @ e)
        return out if np.isfinite(out) else 1e12

    # Seed the slope from an equal-weight regression so the optimizer
    # does not sit at the b=0 ridge where the weights are unidentified.
    mid0 = Xr @ np.ones(k_lags) / k_lags
    b0 = float(np.cov(mid0, yr)[0, 1] / max(np.var(mid0), 1e-12))
    a0 = float(yr.mean() - b0 * mid0.mean())
    starts = [
        np.array([a0, b0, math.log(1.0), math.log(5.0), 0.0]),
        np.array([a0, b0, math.log(2.0), math.log(2.0), 0.0]),
        np.array([a0, b0, math.log(1.0), math.log(1.0), 0.0]),
    ]
    if not ar_lag:
        starts = [s[:4] for s in starts]
    best = None
    for s0 in starts:
        res = opt.minimize(
            sse, s0, method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-6}
        )
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun):
        raise ValueError("MIDAS fit failed")
    a, b, t1, t2, c = unpack(best.x)
    w = beta_weights(k_lags, t1, t2)
    mid_full = X @ w
    pred = a + b * mid_full + (c * ylag if ar_lag else 0.0)
    resid = y - pred
    resid = resid[rows]
    return {
        "intercept": float(a),
        "slope": float(b),
        "theta1": float(t1),
        "theta2": float(t2),
        "ar_coef": float(c),
        "weights": w,
        "fitted": pred,
        "resid": resid,
        "sse": float(best.fun),
    }


def almon_weights(k: int, degree: int, coefs: Array) -> Array:
    """Almon (1965) polynomial distributed-lag weights.

    ``w_j = sum_{d=0}^{deg} coefs_d * j^d`` for j = 0..k-1."""
    c = np.asarray(coefs, dtype=float).reshape(-1)
    if c.size != degree + 1:
        raise ValueError("coefs must have degree+1 entries")
    if k < 1 or not np.all(np.isfinite(c)):
        raise ValueError("invalid k or coefs")
    j = np.arange(k, dtype=float)
    w = np.zeros(k)
    for d_i, cd in enumerate(c):
        w += cd * j**d_i
    return w


def fit_almon(y: Array, X_lags: Array, degree: int = 2) -> dict[str, Array | float]:
    """Almon distributed-lag regression by restricted OLS.

    ``X_lags`` is (T, k) aligned lag matrix; the polynomial constraint
    reduces the k lag coefficients to degree+1 free parameters via the
    Almon transform Z = X @ P where P[j, d] = j^d.
    """
    yv = _v(y)
    X = np.asarray(X_lags, dtype=float)
    if X.ndim != 2 or X.shape[0] != yv.size or not np.all(np.isfinite(X)):
        raise ValueError("X_lags must be finite (T, k) aligned to y")
    n, k = X.shape
    if not (1 <= degree < k):
        raise ValueError("degree must be in [1, k)")
    j = np.arange(k, dtype=float)
    P = np.stack([j**d for d in range(degree + 1)], axis=1)
    Z = X @ P
    Zx = np.column_stack([np.ones(n), Z])
    beta, *_ = np.linalg.lstsq(Zx, yv, rcond=None)
    u = yv - Zx @ beta
    dof = max(n - degree - 2, 1)
    s2 = float(u @ u / dof)
    try:
        cov = s2 * np.linalg.inv(Zx.T @ Zx)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(degree + 2, np.nan)
    w = P @ beta[1:]
    return {
        "weights": w,
        "poly_coefs": beta[1:],
        "intercept": float(beta[0]),
        "se": se,
        "resid": u,
        "r2": float(1.0 - (u @ u) / max(float(((yv - yv.mean()) ** 2).sum()), 1e-20)),
    }


def bridge_regression(
    y_low: Array, monthly_means: Array, growth_terms: Array
) -> dict[str, Array | float]:
    """Baffigi–Golinelli–Parigi (2004) bridge equation.

    Regresses the low-frequency target on high-frequency indicators
    aggregated to the target frequency (means + optional growth terms).
    ``monthly_means`` and ``growth_terms`` are (T_low, m) matrices."""
    y = _v(y_low)
    M = np.asarray(monthly_means, dtype=float)
    G = np.asarray(growth_terms, dtype=float)
    if M.ndim != 2 or M.shape[0] != y.size or not np.all(np.isfinite(M)):
        raise ValueError("monthly_means must be finite (T_low, m)")
    if G.size == 0:
        Z = M
    else:
        if G.ndim != 2 or G.shape[0] != y.size or not np.all(np.isfinite(G)):
            raise ValueError("growth_terms must be finite (T_low, m2)")
        Z = np.column_stack([M, G])
    n, m = Z.shape
    Zx = np.column_stack([np.ones(n), Z])
    beta, *_ = np.linalg.lstsq(Zx, y, rcond=None)
    u = y - Zx @ beta
    dof = max(n - m - 1, 1)
    s2 = float(u @ u / dof)
    try:
        cov = s2 * np.linalg.inv(Zx.T @ Zx)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(m + 1, np.nan)
    return {
        "coefs": beta,
        "se": se,
        "resid": u,
        "r2": float(1.0 - (u @ u) / max(float(((y - y.mean()) ** 2).sum()), 1e-20)),
        "t_stats": beta / np.maximum(se, 1e-12),
    }
