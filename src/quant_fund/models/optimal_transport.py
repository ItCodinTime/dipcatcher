"""Optimal-transport primitives for distributional alignment.

- ``sinkhorn_plan``: entropic OT between two empirical measures on a cost
  matrix, computed in the log domain for numerical stability
  (Cuturi 2013, Peyre & Cuturi 2019 ch. 4). Returns the transport plan,
  regularized transport cost, and the dual potentials.
- ``wasserstein_barycenter_1d``: exact W2 barycenter of one-dimensional
  measures via quantile averaging (Agueh & Carlier 2011): the barycenter's
  quantile function is the weighted mean of the input quantile functions.
- ``gaussian_bures``: fixed-point Frechet mean of multivariate Gaussians
  under W2 (Alvarez-Esteban et al. 2016), iterating the Bures map.

Fail-closed: negative marginals, non-PSD covariances, non-finite input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import fractional_matrix_power, sqrtm

Array = NDArray[np.float64]


def _check_marginals(a: Array, b: Array) -> tuple[Array, Array]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("marginals must be finite and non-empty")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("marginals must be non-negative")
    if abs(a.sum() - 1.0) > 1e-6 or abs(b.sum() - 1.0) > 1e-6:
        raise ValueError("marginals must each sum to 1")
    return a, b


def sinkhorn_plan(
    a: Array,
    b: Array,
    cost: Array,
    eps: float,
    n_iter: int = 500,
    tol: float = 1e-9,
) -> dict[str, Array | float | int]:
    """Entropic OT plan via log-stabilized Sinkhorn iterations."""
    a, b = _check_marginals(a, b)
    c = np.asarray(cost, dtype=float)
    if c.shape != (a.size, b.size) or not np.isfinite(c).all():
        raise ValueError("cost must be a finite (len(a), len(b)) matrix")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    k = np.exp(-c / eps)
    u = np.zeros(a.size)
    v = np.zeros(b.size)
    log_a = np.log(np.clip(a, 1e-300, None))
    log_b = np.log(np.clip(b, 1e-300, None))
    it = 0
    err = np.inf
    for it in range(1, n_iter + 1):
        # u_i = log a_i - logsumexp_j(-c/eps + v_j); mirror for v
        m1 = (-c / eps + v[None, :]).max(axis=1)
        u_new = log_a - (m1 + np.log(np.exp(-c / eps + v[None, :] - m1[:, None]).sum(axis=1)))
        m2 = (-c / eps + u_new[:, None]).max(axis=0)
        v = log_b - (m2 + np.log(np.exp(-c / eps + u_new[:, None] - m2[None, :]).sum(axis=0)))
        u = u_new
        if it % 10 == 0:
            plan_col = np.exp(u[:, None] - c / eps + v[None, :])
            err = float(np.abs(plan_col.sum(axis=1) - a).max())
            if err < tol:
                break
    plan = np.diag(np.exp(u)) @ k @ np.diag(np.exp(v))
    return {
        "plan": plan,
        "cost": float(np.sum(plan * c)),
        "iters": it,
        "dual_u": u,
        "dual_v": v,
        "residual": float(err),
    }


def wasserstein_barycenter_1d(
    supports: Array,
    probs: Array,
    weights: Array,
    n_grid: int = 256,
) -> dict[str, Array]:
    """Exact W2 barycenter of 1-D empirical measures by quantile averaging.

    ``supports`` (K, n) support points, ``probs`` (K, n) masses (each sums
    to 1), ``weights`` (K,) barycenter weights (sums to 1). Returns the
    barycenter quantile function on a uniform grid in (0, 1).
    """
    supports = np.asarray(supports, dtype=float)
    probs = np.asarray(probs, dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()
    if supports.ndim != 2 or probs.shape != supports.shape:
        raise ValueError("supports/probs must be matching (K, n) arrays")
    if weights.size != supports.shape[0] or (weights < 0).any():
        raise ValueError("weights must be non-negative, one per measure")
    if abs(weights.sum() - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1")
    if not np.isfinite(supports).all() or not np.isfinite(probs).all():
        raise ValueError("non-finite input")
    if (probs < 0).any() or np.abs(probs.sum(axis=1) - 1.0).max() > 1e-6:
        raise ValueError("each probs row must be a probability vector")
    if n_grid < 8:
        raise ValueError("n_grid must be >= 8")
    u = (np.arange(n_grid) + 0.5) / n_grid
    qs = np.empty((supports.shape[0], n_grid))
    for k in range(supports.shape[0]):
        cdf = np.concatenate([[0.0], np.cumsum(probs[k])])
        sup = np.concatenate([[supports[k, 0]], supports[k]])
        qs[k] = np.interp(u, cdf, sup)
    return {"u": u, "quantile": weights @ qs, "members": qs}


def gaussian_bures(
    means: Array,
    covs: Array,
    weights: Array,
    n_iter: int = 60,
    tol: float = 1e-10,
) -> dict[str, Array | float | int]:
    """W2 barycenter of Gaussians N(m_k, S_k) (Alvarez-Esteban et al. 2016)."""
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()
    if means.ndim != 2 or covs.ndim != 3 or covs.shape[0] != means.shape[0]:
        raise ValueError("means (K, d), covs (K, d, d)")
    d = means.shape[1]
    if covs.shape[1:] != (d, d):
        raise ValueError("covs must be (K, d, d)")
    if weights.size != means.shape[0] or (weights < 0).any() or abs(weights.sum() - 1.0) > 1e-6:
        raise ValueError("weights must be a probability vector over measures")
    if not np.isfinite(means).all() or not np.isfinite(covs).all():
        raise ValueError("non-finite input")
    for k in range(covs.shape[0]):
        if np.linalg.eigvalsh(covs[k]).min() <= 0:
            raise ValueError("every covariance must be positive definite")
    s = np.mean(covs, axis=0)
    it = 0
    err = np.inf
    while it < n_iter:
        it += 1
        s_half = sqrtm(s).real
        s_inv_half = fractional_matrix_power(s, -0.5).real
        acc = np.zeros_like(s)
        for k in range(covs.shape[0]):
            inner = fractional_matrix_power(s_half @ covs[k] @ s_half, 0.5).real
            acc += weights[k] * inner
        s_new = s_inv_half @ (acc @ acc) @ s_inv_half
        err = float(np.linalg.norm(s_new - s, "fro") / max(np.linalg.norm(s, "fro"), 1e-12))
        s = (s_new + s_new.T) / 2.0
        if err < tol:
            break
    return {
        "mean": weights @ means,
        "cov": s,
        "iters": it,
        "residual": float(err),
    }
