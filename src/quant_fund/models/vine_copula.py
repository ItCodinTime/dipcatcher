"""C-vine copula with Gaussian pair copulas.

A vine decomposes a d-dimensional copula density into d(d-1)/2 bivariate
pair copulas on a tree of nested conditionals.  The canonical (C-) vine
fits star trees: level 1 pairs (1, j), level l pairs (l, j | 1..l-1) via
the h-function recursion

    h(u | v; rho) = Phi( (Phi^{-1}(u) - rho * Phi^{-1}(v)) / sqrt(1 - rho^2) )

with each pair correlation estimated by Kendall's-tau inversion
``rho = sin(pi * tau / 2)`` (Genest & Favre 2007 for the IFM baseline).

References: T. Bedford & R. M. Cooke (2002), Annals of Statistics 30(4);
K. Aas et al. (2009), Insurance: Mathematics and Economics 44(2).
Fail-closed on non-finite input, d < 2, or observations outside (0,1).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from quant_fund.models.copula import kendall_tau

Array = NDArray[np.float64]

_EPS = 1e-10


def _as_pseudo(u: Array) -> Array:
    m = np.asarray(u, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2 or m.shape[0] < 10:
        raise ValueError("u must be an (n, d) pseudo-sample, d >= 2, n >= 10")
    if not np.all(np.isfinite(m)):
        raise ValueError("u must be finite")
    if (m <= 0.0).any() or (m >= 1.0).any():
        raise ValueError("u must lie strictly inside (0, 1)")
    return m


def to_pseudo(x: Array) -> Array:
    """Rank-transform an (n, d) sample to (0,1) via ranks/(n+1)."""
    m = np.asarray(x, dtype=float)
    if m.ndim != 2 or m.shape[0] < 10:
        raise ValueError("x must be a finite (n, d) sample, n >= 10")
    if not np.all(np.isfinite(m)):
        raise ValueError("x must be finite")
    n = m.shape[0]
    out = np.empty_like(m)
    for j in range(m.shape[1]):
        out[:, j] = stats.rankdata(m[:, j]) / (n + 1.0)
    return out


def _h(u: Array, v: Array, rho: float) -> Array:
    """Conditional CDF of U | V = v under the Gaussian pair copula."""
    r = min(max(rho, -0.999), 0.999)
    zu = stats.norm.ppf(np.clip(u, _EPS, 1 - _EPS))
    zv = stats.norm.ppf(np.clip(v, _EPS, 1 - _EPS))
    return np.asarray(stats.norm.cdf((zu - r * zv) / math.sqrt(1 - r * r)), dtype=float)


def _h_inv(u: Array, v: Array, rho: float) -> Array:
    """Inverse h: u | v quantile under the Gaussian pair copula."""
    r = min(max(rho, -0.999), 0.999)
    zu = stats.norm.ppf(np.clip(u, _EPS, 1 - _EPS))
    zv = stats.norm.ppf(np.clip(v, _EPS, 1 - _EPS))
    return np.asarray(stats.norm.cdf(r * zv + math.sqrt(1 - r * r) * zu), dtype=float)


def _pair_loglik(u: Array, v: Array, rho: float) -> float:
    r = min(max(rho, -0.999), 0.999)
    zu = stats.norm.ppf(np.clip(u, _EPS, 1 - _EPS))
    zv = stats.norm.ppf(np.clip(v, _EPS, 1 - _EPS))
    return float(
        np.sum(
            -0.5 * math.log(1 - r * r)
            - (r * r * (zu**2 + zv**2) - 2 * r * zu * zv) / (2 * (1 - r * r))
        )
    )


def cvine_fit(u: Array) -> dict[str, Array]:
    """Fit a canonical vine: tree l pairs (l, j | 1..l-1), tau-inverted rho.

    Returns the lower-triangular ``theta`` matrix (``theta[l, j]`` is the
    pair correlation at tree ``l`` for column ``j``), total loglik, and the
    pseudo-observation count.
    """
    m = _as_pseudo(u)
    n, d = m.shape
    theta, loglik = _cvine_exact(m)
    return {
        "theta": theta,
        "loglik": np.array([loglik]),
        "n": np.array([n]),
        "d": np.array([d]),
    }


def _cvine_exact(m: Array) -> tuple[Array, float]:
    """Canonical C-vine fit: at tree l the pivot is column l conditioned on
    columns 0..l-1 via the h-recursion."""
    n, d = m.shape
    theta = np.zeros((d - 1, d))
    total = 0.0
    # cond[l, j] = value of column j conditioned on columns 0..l-1
    cond = m.copy()
    for level in range(d - 1):
        pivot = cond[:, level]
        nxt = np.empty((n, d))
        nxt[:] = np.nan
        for j in range(level + 1, d):
            vj = cond[:, j]
            tau = kendall_tau(np.column_stack([pivot, vj]))
            rho = math.sin(0.5 * math.pi * tau)
            theta[level, j] = rho
            total += _pair_loglik(vj, pivot, rho)
            nxt[:, j] = _h(vj, pivot, rho)
        cond = np.hstack([cond[:, : level + 1], nxt[:, level + 1 :]])
    return theta, total


def cvine_loglik(fit: dict[str, Array], u: Array) -> float:
    """Evaluate the fitted vine density on new pseudo-observations."""
    m = _as_pseudo(u)
    theta = np.asarray(fit["theta"], dtype=float)
    n, d = m.shape
    total = 0.0
    cond = m.copy()
    for level in range(d - 1):
        pivot = cond[:, level]
        nxt = np.full((n, d), np.nan)
        for j in range(level + 1, d):
            rho = theta[level, j]
            vj = cond[:, j]
            total += _pair_loglik(vj, pivot, rho)
            nxt[:, j] = _h(vj, pivot, rho)
        cond = np.hstack([cond[:, : level + 1], nxt[:, level + 1 :]])
    return float(total)


def cvine_simulate(fit: dict[str, Array], n: int = 500, *, seed: int = 0) -> dict[str, Array]:
    """Sample the fitted C-vine via sequential inverse-h recursion.

    Draws independent uniforms w_i; u_1 = w_1; u_j for j >= 2 is obtained
    by pushing w_j down the tree: at tree l, ``u_j = h^{-1}(v | u_l)`` in
    order l = j-1 .. 0 (C-vine inversion).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    theta = np.asarray(fit["theta"], dtype=float)
    d = theta.shape[1]
    rng = np.random.default_rng(seed)
    w = rng.uniform(_EPS, 1 - _EPS, size=(n, d))
    out = np.empty((n, d))
    out[:, 0] = w[:, 0]
    for j in range(1, d):
        v = w[:, j]
        # walk down the trees from top (j-1) to 0
        for level in range(j - 1, -1, -1):
            v = _h_inv(v, out[:, level], theta[level, j])
        out[:, j] = v
    return {"u": out}
