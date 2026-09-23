"""High-breakdown covariance estimators.

- FastMCD (Rousseeuw & Van Driessen 1999): concentration-step MCD with
  random starts + classical reweighting at the chi2 0.975 cutoff.
- OGK (Maronna & Zamar 2002): orthogonalized Gnanadesikan-Kettenring
  pairwise robust covariance with MAD scale.
- Stahel-Donoho outlyingness: projection-depth outlier score via random
  directions.

Fail-closed: n too small vs dimension, non-finite data, singular scales.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

Array = NDArray[np.float64]


def _check_x(x: Array) -> Array:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 5 or a.shape[1] < 1:
        raise ValueError("X must be 2-D with >= 5 rows")
    if a.shape[0] <= 2 * a.shape[1]:
        raise ValueError("need n > 2p for a meaningful robust covariance")
    if not np.isfinite(a).all():
        raise ValueError("X must be finite")
    return a


def _mad(v: Array) -> float:
    med = float(np.median(v))
    s = float(np.median(np.abs(v - med))) * 1.4826
    return s


def _mahalanobis(x: Array, center: Array, cov: Array) -> Array:
    cov_inv = np.linalg.pinv(cov)
    d = x - center
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, cov_inv, d), 0.0))


def ogk_cov(x: Array) -> dict[str, Array | float]:
    """Orthogonalized Gnanadesikan-Kettenring estimator (Maronna-Zamar 2002).

    Step 1: robust column scales via MAD. Step 2: pairwise GK covariance
    cov(u,v) = (s(u+v)^2 - s(u-v)^2)/4 on scaled data. Step 3:
    orthogonalize by eigendecomposition and re-estimate variances.
    """
    xx = _check_x(x)
    n, p = xx.shape
    s = np.array([_mad(xx[:, j]) for j in range(p)])
    if (s <= 0).any():
        raise ValueError("zero MAD scale in a column")
    y = xx / s
    gamma = np.eye(p)
    for i in range(p):
        for j in range(p):
            if i == j:
                gamma[i, j] = _mad(y[:, i]) ** 2
            else:
                sp = _mad(y[:, i] + y[:, j])
                sm = _mad(y[:, i] - y[:, j])
                gamma[i, j] = 0.25 * (sp * sp - sm * sm)
    # orthogonalize
    eigval, u = np.linalg.eigh(gamma)
    z = y @ u
    lam = np.array([_mad(z[:, j]) ** 2 for j in range(p)])
    cov_scaled = (u * lam) @ u.T
    cov = np.diag(s) @ cov_scaled @ np.diag(s)
    center = np.array([float(np.median(xx[:, j])) for j in range(p)])
    return {
        "cov": np.asarray(cov, dtype=float),
        "center": center,
        "scales": s,
        "n": float(n),
        "p": float(p),
    }


def fast_mcd(
    x: Array,
    h: int | None = None,
    n_starts: int = 50,
    max_iter: int = 20,
    rng: np.random.Generator | None = None,
) -> dict[str, Array | float]:
    """FastMCD: h-subset MCD with C-steps, returns reweighted center/cov.

    ``h`` defaults to floor((n + p + 1)/2) (max breakdown). Objective is
    det(cov of h closest points); best over ``n_starts`` random starts.
    """
    xx = _check_x(x)
    n, p = xx.shape
    if h is None:
        h = (n + p + 1) // 2
    if not p + 1 <= h <= n:
        raise ValueError("h must satisfy p+1 <= h <= n")
    if n_starts < 1 or max_iter < 1:
        raise ValueError("n_starts, max_iter >= 1")
    gen = rng if rng is not None else np.random.default_rng(0)

    best_det = np.inf
    best_center = np.zeros(p)
    best_cov = np.eye(p)
    for _ in range(n_starts):
        idx = gen.choice(n, size=h, replace=False)
        center = xx[idx].mean(axis=0)
        cov = np.cov(xx[idx].T) + 1e-9 * np.eye(p)
        prev_det = np.inf
        for _ in range(max_iter):
            d = _mahalanobis(xx, center, cov)
            idx = np.argsort(d)[:h]
            center = xx[idx].mean(axis=0)
            cov = np.cov(xx[idx].T) + 1e-9 * np.eye(p)
            det = float(np.linalg.slogdet(cov)[1])
            if det >= prev_det - 1e-12:
                break
            prev_det = det
        if prev_det < best_det:
            best_det = prev_det
            best_center = center
            best_cov = cov

    # classical reweighting at chi2(p) 0.975
    d = _mahalanobis(xx, best_center, best_cov)
    keep = d**2 <= chi2.ppf(0.975, p)
    if keep.sum() <= p:
        raise ValueError("reweighting removed too many points")
    center_rw = xx[keep].mean(axis=0)
    cov_rw = np.cov(xx[keep].T)
    d_final = _mahalanobis(xx, center_rw, cov_rw)
    return {
        "center": center_rw,
        "cov": np.asarray(cov_rw, dtype=float),
        "distances": d_final,
        "raw_center": best_center,
        "raw_cov": best_cov,
        "h": float(h),
        "n_kept": float(keep.sum()),
    }


def stahel_donoho_outlyingness(
    x: Array,
    n_dirs: int = 250,
    rng: np.random.Generator | None = None,
) -> Array:
    """Stahel-Donoho outlyingness per row via random projection directions.

    o_i = max_a |x_i' a - med(X'a)| / MAD(X'a), a ~ unit sphere.
    """
    xx = _check_x(x)
    n, p = xx.shape
    if n_dirs < 10:
        raise ValueError("n_dirs >= 10")
    gen = rng if rng is not None else np.random.default_rng(0)
    dirs = gen.standard_normal((n_dirs, p))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    out = np.zeros(n)
    for a in dirs:
        proj = xx @ a
        med = float(np.median(proj))
        mad = _mad(proj)
        if mad <= 0.0:
            continue
        out = np.maximum(out, np.abs(proj - med) / mad)
    if (out == 0.0).all():
        raise ValueError("all projection directions degenerate")
    return out
