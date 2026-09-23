"""POET covariance: Principal Orthogonal complEment Thresholding
(Fan, Liao & Mincheva 2013).

Approximate factor model R = F Lambda' + U. Decompose the p x p
covariance as Sigma = Lambda Lambda' + Sigma_u, where Sigma_u is made
sparse by thresholding the residual correlation off-diagonals:

    tau_ij = c * sqrt(ln p / T)  (universal threshold on correlation)

Estimated factor space uses the leading k principal components of the
sample covariance; k may be chosen by the Ahn-Horenstein eigenvalue
ratio. Thresholded Sigma_u is renormalized to the residual variances
and re-diagonalized if needed to keep the result PSD.

Fail-closed: T < p cases still run (POET is designed for them) but the
threshold must shrink toward diagonal dominance; invalid inputs raise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check(r: Array) -> Array:
    rr = np.asarray(r, dtype=float)
    if rr.ndim != 2 or rr.shape[0] < 30 or rr.shape[1] < 5:
        raise ValueError("returns must be (T, p), T >= 30, p >= 5")
    if not np.isfinite(rr).all():
        raise ValueError("non-finite input")
    if (rr.std(axis=0) <= 0).any():
        raise ValueError("degenerate column")
    return rr


def poet_select_k(returns: Array, kmax: int = 10) -> int:
    """Ahn-Horenstein (2013) eigenvalue-ratio factor count."""
    rr = _check(returns)
    t, p = rr.shape
    kmax = int(min(kmax, min(t, p) - 2))
    ev = np.linalg.eigvalsh(np.cov(rr.T))[::-1]
    ratios = ev[: kmax + 1] / np.maximum(ev[1 : kmax + 2], 1e-14)
    return int(np.argmax(ratios) + 1)


def poet_cov(
    returns: Array,
    k: int | None = None,
    thresh: float | None = None,
) -> dict[str, Array | float]:
    """POET covariance estimator.

    Returns sigma (p, p), loadings (p, k), sigma_u, n_factors, and the
    fraction of residual off-diagonals that were thresholded to zero.
    """
    rr = _check(returns)
    t, p = rr.shape
    if k is None:
        k = poet_select_k(rr)
    if not 1 <= k < min(t, p):
        raise ValueError("k out of range")
    c_thresh = float(np.sqrt(np.log(p) / t)) if thresh is None else float(thresh)
    if c_thresh <= 0:
        raise ValueError("thresh must be positive")

    rc = rr - rr.mean(axis=0)
    s_cov = rc.T @ rc / t
    vals, vecs = np.linalg.eigh(s_cov)
    idx = np.argsort(vals)[::-1][:k]
    # factors via PCA: F = rc @ V, loadings = V * sqrt(t)? use least squares
    v_k = vecs[:, idx] * np.sqrt(vals[idx])  # (p, k) scaled eigenvectors = Lambda
    factors = rc @ vecs[:, idx]  # (t, k) PC scores
    # residual: project rc off the PC subspace
    u = rc - rc @ vecs[:, idx] @ vecs[:, idx].T
    s_u = u.T @ u / t
    d_u = np.sqrt(np.maximum(np.diag(s_u), 1e-14))
    corr_u = s_u / np.outer(d_u, d_u)
    # hard-threshold off-diagonal correlations
    mask = np.abs(corr_u) > c_thresh
    np.fill_diagonal(mask, True)
    s_u_thr = s_u * mask
    # enforce PSD by eigenvalue clipping
    ev_u, vv_u = np.linalg.eigh(s_u_thr)
    ev_u = np.maximum(ev_u, 0.0)
    s_u_thr = vv_u @ np.diag(ev_u) @ vv_u.T
    lam = v_k  # loadings: columns are sqrt(lambda_j) * eigenvector
    sigma = lam @ lam.T + s_u_thr
    sparse_share = 1.0 - float(mask[np.triu_indices(p, 1)].mean())
    return {
        "sigma": sigma,
        "loadings": lam,
        "sigma_u": s_u_thr,
        "factors": factors,
        "n_factors": float(k),
        "sparsity": sparse_share,
        "eigenvalues": vals[idx],
    }
