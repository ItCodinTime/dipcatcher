"""Dynamic factor model: two-step Doz, Giannone & Reichlin (2012)
estimator for the approximate DFM

    x_t = Lambda F_t + e_t,      F_t = A F_{t-1} + u_t,

Step 1: static PCA of the standardized panel gives F0, Lambda0.
Step 2: a VAR(1) on F0 gives A; a single Kalman smoother pass with
(Lambda0, A, diag(idio variances)) refines the factor path.

Returns loadings, smoothed factors, VAR matrix A, common component,
idiosyncratic R2, eigenvalue spectrum for factor-count inspection.
Fail-closed: non-finite panel, degenerate columns, r out of range.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_lyapunov

Array = NDArray[np.float64]


def _check(x: Array) -> Array:
    xx = np.asarray(x, dtype=float)
    if xx.ndim != 2 or xx.shape[0] < 40 or xx.shape[1] < 5:
        raise ValueError("x must be (T, N), T >= 40, N >= 5")
    if not np.isfinite(xx).all():
        raise ValueError("non-finite input")
    if (xx.std(axis=0) <= 0).any():
        raise ValueError("degenerate column")
    return xx


def _kalman_smooth(x: Array, lam: Array, a: Array, r_diag: Array, q: Array) -> Array:
    """Kalman smoother for x_t = Lam F_t + e, F_t = A F_{t-1} + u."""
    t_n, n = x.shape
    r = lam.shape[1]
    f_pred = np.zeros(r)
    p_pred = solve_discrete_lyapunov(a, q)
    p_pred = (p_pred + p_pred.T) / 2.0
    f_filt = np.empty((t_n, r))
    p_filt = np.empty((t_n, r, r))
    p_preds = np.empty((t_n, r, r))
    f_preds = np.empty((t_n, r))
    for t in range(t_n):
        f_preds[t] = f_pred
        p_preds[t] = p_pred
        resid = x[t] - lam @ f_pred
        s = lam @ p_pred @ lam.T + np.diag(r_diag)
        k_gain = np.linalg.solve(s.T, (p_pred @ lam.T).T).T  # p_pred lam' S^{-1}
        f_upd = f_pred + k_gain @ resid
        p_upd = p_pred - k_gain @ lam @ p_pred
        f_filt[t] = f_upd
        p_filt[t] = (p_upd + p_upd.T) / 2.0
        f_pred = a @ f_upd
        p_pred = a @ p_filt[t] @ a.T + q
    # RTS smoother
    f_smooth = f_filt.copy()
    for t in range(t_n - 2, -1, -1):
        jmat = np.linalg.solve(p_preds[t + 1].T, (p_filt[t] @ a.T).T).T
        f_smooth[t] = f_filt[t] + jmat @ (f_smooth[t + 1] - f_preds[t + 1])
    return f_smooth


def dfm_fit(x: Array, r: int) -> dict[str, Array | float]:
    """Two-step DFM fit with r factors."""
    xx = _check(x)
    t_n, n = xx.shape
    if not 1 <= r < min(t_n, n):
        raise ValueError("r out of range")
    xs = (xx - xx.mean(axis=0)) / xx.std(axis=0)
    cov = xs.T @ xs / t_n
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1][:r]
    lam = vecs[:, idx] * np.sqrt(vals[idx])  # (N, r) loadings
    f0 = xs @ vecs[:, idx] / np.sqrt(vals[idx])[None, :]  # PC scores
    # sign normalization: make loadings' largest-abs entry positive
    for j in range(r):
        k = int(np.argmax(np.abs(lam[:, j])))
        if lam[k, j] < 0:
            lam[:, j] *= -1
            f0[:, j] *= -1
    # VAR(1) on factors
    a = np.linalg.lstsq(f0[:-1], f0[1:], rcond=None)[0].T  # F_t = A F_{t-1}
    u = f0[1:] - f0[:-1] @ a.T
    q = u.T @ u / max(u.shape[0] - 1, 1)
    # idiosyncratic variances
    resid0 = xs - f0 @ lam.T
    r_diag = np.maximum(resid0.var(axis=0), 1e-6)
    # Kalman smoothing pass
    f_smooth = _kalman_smooth(xs, lam, a, r_diag, q)
    common = f_smooth @ lam.T
    ss_tot = float((xs**2).sum())
    r2 = 1.0 - float(((xs - common) ** 2).sum()) / ss_tot
    return {
        "loadings": lam,
        "factors": f_smooth,
        "A": a,
        "Q": q,
        "R_diag": r_diag,
        "common": common,
        "r2": r2,
        "eigenvalues": vals[::-1],
    }


def dfm_forecast(fit: dict[str, Array | float]) -> Array:
    """One-step-ahead common-component forecast Lambda A f_T."""
    lam = np.asarray(fit["loadings"], dtype=float)
    a = np.asarray(fit["A"], dtype=float)
    f_last = np.asarray(fit["factors"], dtype=float)[-1]
    return np.asarray(lam @ (a @ f_last), dtype=float)
