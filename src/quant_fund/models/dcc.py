"""Engle (2002) DCC(1,1) dynamic conditional correlation.

Two-stage QMLE:
  Stage 1: univariate GARCH(1,1) per series -> standardized residuals z_t.
  Stage 2: Q_t = (1-a-b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1};
           R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2};
           stage-2 loglik = -0.5 sum (ln|R_t| + z_t' R_t^{-1} z_t).

Fail-closed: a + b >= 1, non-PD Q_t, non-finite input, insufficient obs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

Array = NDArray[np.float64]


def _garch11(r: Array) -> tuple[float, float, float, Array]:
    """Scalar GARCH(1,1) QMLE -> (omega, alpha, beta, sig2 path)."""
    v0 = float(r.var())

    def nll(th: Array) -> float:
        w, a, b = float(th[0]), float(th[1]), float(th[2])
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        s2 = np.empty(r.size)
        s2[0] = v0
        for t in range(1, r.size):
            s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
            if s2[t] <= 0 or not np.isfinite(s2[t]):
                return 1e12
        return float(0.5 * np.sum(np.log(2 * np.pi * s2) + r * r / s2))

    res = optimize.minimize(
        nll, np.array([0.05 * v0, 0.05, 0.9]), method="Nelder-Mead", options={"maxiter": 3000}
    )
    w, a, b = (float(v) for v in res.x)
    s2 = np.empty(r.size)
    s2[0] = v0
    for t in range(1, r.size):
        s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
    return w, a, b, s2


def dcc_fit(returns: Array) -> dict[str, Array | float]:
    """DCC(1,1) fit on (T, k) returns.

    Returns per-series GARCH params, sig2 paths, a, b, R path (T,k,k),
    Qbar, stage-2 loglik.
    """
    rr = np.asarray(returns, dtype=float)
    if rr.ndim != 2 or rr.shape[0] < 100 or rr.shape[1] < 2:
        raise ValueError("returns must be (T, k), T >= 100, k >= 2")
    if not np.isfinite(rr).all():
        raise ValueError("non-finite input")
    t_n, k = rr.shape
    rc = rr - rr.mean(axis=0)
    z = np.empty_like(rc)
    sig2 = np.empty_like(rc)
    garch_params = np.empty((k, 3))
    for j in range(k):
        w, a, b, s2 = _garch11(rc[:, j])
        garch_params[j] = (w, a, b)
        sig2[:, j] = s2
        z[:, j] = rc[:, j] / np.sqrt(s2)
    qbar = np.corrcoef(z.T)

    def stage2_nll(par: Array) -> float:
        a, b = float(par[0]), float(par[1])
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        q = qbar.copy()
        ll = 0.0
        for t in range(1, t_n):
            q = (1 - a - b) * qbar + a * np.outer(z[t - 1], z[t - 1]) + b * q
            d = np.sqrt(np.diag(q))
            if (d <= 0).any():
                return 1e12
            r_t = q / np.outer(d, d)
            sign, logdet = np.linalg.slogdet(r_t)
            if sign <= 0:
                return 1e12
            try:
                quad = float(z[t] @ np.linalg.solve(r_t, z[t]))
            except np.linalg.LinAlgError:
                return 1e12
            ll += -0.5 * (logdet + quad)
        return -ll

    res = optimize.minimize(
        stage2_nll, np.array([0.04, 0.93]), method="Nelder-Mead", options={"maxiter": 2000}
    )
    a, b = float(res.x[0]), float(res.x[1])
    if not np.isfinite(res.fun) or a + b >= 0.999:
        raise ValueError("DCC stage-2 fit failed")
    # final paths
    q = qbar.copy()
    q_path = np.empty((t_n, k, k))
    r_path = np.empty((t_n, k, k))
    q_path[0] = q
    d0 = np.sqrt(np.diag(q))
    r_path[0] = q / np.outer(d0, d0)
    for t in range(1, t_n):
        q = (1 - a - b) * qbar + a * np.outer(z[t - 1], z[t - 1]) + b * q
        q_path[t] = q
        d = np.sqrt(np.maximum(np.diag(q), 1e-14))
        r_path[t] = q / np.outer(d, d)
    return {
        "a": a,
        "b": b,
        "persistence": a + b,
        "qbar": qbar,
        "garch_params": garch_params,
        "sig2": sig2,
        "z": z,
        "Q": q_path,
        "R": r_path,
        "loglik": float(-res.fun),
    }


def dcc_forecast(fit: dict[str, Array | float]) -> Array:
    """One-step-ahead correlation forecast R_{T+1}."""
    z = np.asarray(fit["z"], dtype=float)
    q_last = np.asarray(fit["Q"], dtype=float)[-1]
    qbar = np.asarray(fit["qbar"], dtype=float)
    a, b = float(fit["a"]), float(fit["b"])
    q_next = (1 - a - b) * qbar + a * np.outer(z[-1], z[-1]) + b * q_last
    d = np.sqrt(np.maximum(np.diag(q_next), 1e-14))
    return np.asarray(q_next / np.outer(d, d), dtype=float)
