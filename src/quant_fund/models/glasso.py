"""Graphical lasso: sparse precision matrix estimation
(Friedman, Hastie & Tibshirani 2008).

Maximizes  logdet Theta - tr(S Theta) - rho ||Theta||_1  over PD
Theta by the block coordinate descent algorithm: sweep columns, solve
a box-constrained lasso on each column block, update the working
covariance W, then extract Theta from the final blocks.

Also provides:
- ``partial_correlation``: P_ij = -Theta_ij / sqrt(Theta_ii Theta_jj)
- ``select_rho_bic``: extended-BIC grid search over rho

Fail-closed: non-finite data, rho < 0, singular results, T < 20.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _soft(x: float, a: float) -> float:
    return float(np.sign(x) * max(abs(x) - a, 0.0))


def glasso(
    s: Array, rho: float, max_iter: int = 200, tol: float = 1e-6
) -> dict[str, Array | float]:
    """Graphical lasso on sample covariance matrix S. Returns Theta, W."""
    ss = np.asarray(s, dtype=float)
    if ss.ndim != 2 or ss.shape[0] != ss.shape[1]:
        raise ValueError("s must be square")
    if not np.isfinite(ss).all() or rho < 0:
        raise ValueError("non-finite s or negative rho")
    p = ss.shape[0]
    if p < 3:
        raise ValueError("p must be >= 3")
    w = ss + rho * np.eye(p)  # working covariance W
    beta = np.zeros((p, p))
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            idx = np.delete(np.arange(p), j)
            w11 = w[np.ix_(idx, idx)]
            s12 = ss[idx, j]
            b = beta[idx, j].copy()
            # inner coordinate-descent lasso: min .5 b'W11 b - s12'b + rho|b|
            for _inner in range(100):
                b_old = b.copy()
                for i in range(idx.size):
                    b[i] = _soft(s12[i] - (w11[i] @ b - w11[i, i] * b[i]), rho) / w11[i, i]
                if np.abs(b - b_old).max() < 1e-7:
                    break
            beta[idx, j] = b
            w[idx, j] = w11 @ b
            w[j, idx] = w[idx, j]
        if np.abs(w - w_old).max() < tol:
            break
    # extract precision blocks
    theta = np.zeros((p, p))
    for j in range(p):
        idx = np.delete(np.arange(p), j)
        b = beta[idx, j]
        w12 = w[idx, j]
        th22 = 1.0 / max(ss[j, j] + rho - float(w12 @ b), 1e-12)
        theta[idx, j] = -b * th22
        theta[j, j] = th22
    ev = np.linalg.eigvalsh(theta)
    if ev.min() <= 0:
        raise ValueError("glasso produced non-PD precision")
    return {"theta": theta, "W": w, "min_eig": float(ev.min())}


def glasso_from_data(returns: Array, rho: float) -> dict[str, Array | float]:
    """Convenience wrapper: glasso on the sample covariance of returns."""
    rr = np.asarray(returns, dtype=float)
    if rr.ndim != 2 or rr.shape[0] < 20:
        raise ValueError("returns must be (T, p), T >= 20")
    if not np.isfinite(rr).all():
        raise ValueError("non-finite input")
    s = np.cov(rr.T)
    out = glasso(s, rho)
    out["S"] = s
    return out


def partial_correlation(theta: Array) -> Array:
    """Partial correlation matrix from a precision matrix."""
    th = np.asarray(theta, dtype=float)
    d = np.sqrt(np.maximum(np.diag(th), 1e-14))
    p = -th / np.outer(d, d)
    np.fill_diagonal(p, 1.0)
    return np.asarray(np.clip(p, -1.0, 1.0), dtype=float)


def select_rho_bic(returns: Array, rhos: Array) -> dict[str, Array | float]:
    """Extended-BIC-style selection: maximize ll - df*ln(T) - df*ln(p).

    ll = T/2 * (logdet Theta - tr(S Theta)); df = # nonzero off-diagonal.
    """
    rr = np.asarray(returns, dtype=float)
    if rr.ndim != 2 or rr.shape[0] < 20:
        raise ValueError("returns must be (T, p), T >= 20")
    rr = np.asarray(rr)
    t, p = rr.shape
    s = np.cov(rr.T)
    rhos = np.asarray(rhos, dtype=float).ravel()
    if rhos.size == 0 or (rhos < 0).any():
        raise ValueError("rhos must be nonnegative")
    best_rho, best_ebic = -1.0, -np.inf
    ebics = np.empty(rhos.size)
    for i, rho in enumerate(rhos):
        try:
            th = np.asarray(glasso(s, float(rho))["theta"])
        except ValueError:
            ebics[i] = np.inf
            continue
        sign, logdet = np.linalg.slogdet(th)
        if sign <= 0:
            ebics[i] = np.inf
            continue
        ll = 0.5 * t * (logdet - float(np.sum(s * th)))
        df = float((np.abs(th - np.diag(np.diag(th))) > 1e-8).sum() // 2)
        ebics[i] = -2.0 * ll + df * np.log(t) + 2.0 * df * np.log(p) * 0.5
        if ebics[i] < best_ebic or best_ebic == -np.inf:
            best_ebic = ebics[i]
            best_rho = float(rho)
    return {"rho": best_rho, "ebic": ebics, "rhos": rhos}
