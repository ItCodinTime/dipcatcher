"""MIDAS regression: mixed-frequency nowcasting with lag polynomials.

Ghysels, Sinko & Valkanov (2007): regress a low-frequency target on
high-frequency covariates aggregated through a parsimonious lag kernel,

    y_t = a + b * sum_k w_k(theta) * x_{t, k} + e_t,

where ``x_{t,k}`` are the K most recent high-frequency observations visible
at low-frequency date t. Two weight schemes:

- ``exp_almon``: w_k propto exp(theta_1 k + theta_2 k^2) — the standard
  two-parameter exponential Almon lag polynomial.
- ``umidas``: unrestricted per-lag coefficients (plain OLS — the
  parameterization-free limit, useful when K is small).

``midas_fit`` estimates by nonlinear least squares (BFGS on the SSE in
theta); ``midas_forecast`` maps a new (T, K) high-frequency lag matrix to
predictions. Fail-closed: shape/finite checks, K >= 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

Array = NDArray[np.float64]


def _check_xy(y: Array, x_lags: Array) -> tuple[Array, Array, int]:
    yy = np.asarray(y, dtype=float).ravel()
    xx = np.asarray(x_lags, dtype=float)
    if xx.ndim != 2:
        raise ValueError("x_lags must be (T, K)")
    t_len, k = xx.shape
    if yy.size != t_len or t_len < k + 3:
        raise ValueError("y and x_lags must match with T >= K + 3")
    if not np.isfinite(yy).all() or not np.isfinite(xx).all():
        raise ValueError("non-finite input")
    return yy, xx, k


def almon_weights(k: int, theta: Array) -> Array:
    """Normalized exponential Almon weights w_k for k = 0..K-1."""
    if k < 2:
        raise ValueError("k must be >= 2")
    th = np.asarray(theta, dtype=float).ravel()
    if th.size != 2 or not np.isfinite(th).all():
        raise ValueError("theta must be a finite length-2 vector")
    kk = np.arange(k, dtype=float)
    # normalize k to [0,1] to keep theta well-scaled
    z = kk / max(k - 1, 1)
    logits = th[0] * z + th[1] * z * z
    logits -= logits.max()
    w = np.exp(logits)
    return np.asarray(w / w.sum())


def midas_fit(
    y: Array,
    x_lags: Array,
    scheme: str = "exp_almon",
    ridge: float = 1e-6,
) -> dict[str, Array | float | str]:
    """Fit MIDAS: exp-Almon NLS, or unrestricted (UMIDAS) OLS per lag."""
    yy, xx, k = _check_xy(y, x_lags)
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be non-negative and finite")
    if scheme == "umidas":
        design = np.column_stack([np.ones(yy.size), xx])
        g = design.T @ design + ridge * np.eye(k + 1)
        g[0, 0] -= ridge  # don't penalize intercept
        beta = np.linalg.solve(g, design.T @ yy)
        fitted = design @ beta
        resid = yy - fitted
        return {
            "scheme": "umidas",
            "beta": beta,
            "intercept": float(beta[0]),
            "lag_coefs": beta[1:],
            "fitted": fitted,
            "resid": resid,
            "sse": float(resid @ resid),
            "r2": float(1.0 - resid @ resid / np.sum((yy - yy.mean()) ** 2)),
        }
    if scheme != "exp_almon":
        raise ValueError("scheme must be 'exp_almon' or 'umidas'")

    def _sse(theta: Array) -> float:
        a, b, th1, th2 = theta
        w = almon_weights(k, np.array([th1, th2]))
        resid = yy - (a + b * xx @ w)
        return float(resid @ resid)

    theta0 = np.array([yy.mean(), 1.0, 0.0, -2.0])
    res = minimize(_sse, theta0, method="BFGS")
    a, b, th1, th2 = res.x
    w = almon_weights(k, np.array([th1, th2]))
    fitted = a + b * xx @ w
    resid = yy - fitted
    return {
        "scheme": "exp_almon",
        "theta": np.array([th1, th2]),
        "weights": w,
        "intercept": float(a),
        "beta": float(b),
        "fitted": fitted,
        "resid": resid,
        "sse": float(resid @ resid),
        "r2": float(1.0 - resid @ resid / np.sum((yy - yy.mean()) ** 2)),
        "converged": bool(res.success),
    }


def midas_forecast(fit: dict[str, Array | float | str], x_lags: Array) -> Array:
    """Apply a fitted MIDAS to new high-frequency lag rows."""
    xx = np.asarray(x_lags, dtype=float)
    if xx.ndim == 1:
        xx = xx[None, :]
    if not np.isfinite(xx).all():
        raise ValueError("non-finite x_lags")
    scheme = fit["scheme"]
    a = float(fit["intercept"])
    if scheme == "umidas":
        coefs = np.asarray(fit["lag_coefs"], dtype=float)
        if xx.shape[1] != coefs.size:
            raise ValueError("x_lags width must match fitted lag count")
        return np.asarray(a + xx @ coefs)
    w = np.asarray(fit["weights"], dtype=float)
    if xx.shape[1] != w.size:
        raise ValueError("x_lags width must match fitted lag count")
    return np.asarray(a + float(fit["beta"]) * xx @ w)
