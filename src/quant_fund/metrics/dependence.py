"""Nonlinear dependence measures for paired samples.

Pearson correlation misses monotone-but-nonlinear and non-monotone
structure; these statistics capture general dependence, each with a
different strength.

References:
- Szekely, Rizzo & Bakirov (2007): distance correlation (dCor).
- Gretton et al. (2005): HSIC (Hilbert–Schmidt independence criterion).
- Gretton et al. (2012): MMD two-sample statistic.
- Chatterjee (2021): rank-based xi_n, asymptotically normal, detects
  any measurable dependence; power for monotone alternatives.
- Kraskov, Stogbauer & Grassberger (2004): kNN mutual information.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.spatial import cKDTree
from scipy.special import digamma

Array = NDArray[np.float64]


def _xy(x: Array, y: Array, n: int = 20) -> tuple[Array, Array]:
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    if a.size != b.size or a.size < n:
        raise ValueError(f"paired series must share length >= {n}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("series must be finite")
    if a.std() == 0 or b.std() == 0:
        raise ValueError("degenerate (constant) input")
    return a, b


def _dist(v: Array) -> Array:
    d = np.abs(v[:, None] - v[None, :])
    return d


def distance_correlation(x: Array, y: Array) -> dict[str, float]:
    """Szekely et al. (2007) distance correlation in [0, 1].

    dCor = 0 iff independence (for finite first moments). Returns the
    statistic and a permutation-free asymptotic p-value via the
    chi-square approximation ``n*R^2 ~ chi2(1)`` (conservative).
    """
    a, b = _xy(x, y)
    n = a.size
    A = _dist(a)
    B = _dist(b)
    Am = A - A.mean(axis=0)[None, :] - A.mean(axis=1)[:, None] + A.mean()
    Bm = B - B.mean(axis=0)[None, :] - B.mean(axis=1)[:, None] + B.mean()
    dcov2 = float((Am * Bm).mean())
    dvar_x = float((Am * Am).mean())
    dvar_y = float((Bm * Bm).mean())
    if dvar_x <= 0 or dvar_y <= 0:
        raise ValueError("degenerate distance variance")
    dcor = math.sqrt(max(dcov2, 0.0) / math.sqrt(dvar_x * dvar_y))
    # Szekely approximate test: n*dcor^2 ~ chi2(1) (upper bound).
    stat = n * dcor * dcor
    p = float(1.0 - stats.chi2.cdf(stat, 1))
    return {"dcor": float(dcor), "statistic": stat, "pvalue": p, "n": float(n)}


def _rbf(v: Array, sigma: float) -> Array:
    d2 = (v[:, None] - v[None, :]) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma))


def _median_bandwidth(v: Array) -> float:
    d = np.abs(v[:, None] - v[None, :])
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 0.0
    return max(med, 1e-8)


def hsic(x: Array, y: Array, sigma: float | None = None, n_perm: int = 200,
         seed: int = 0) -> dict[str, float]:
    """Gretton et al. (2005) HSIC with median-heuristic RBF kernels.

    HSIC = tr(KHLH)/n^2; the p-value is a seeded permutation test on the
    y-ordering (exact under independence, no parametric null needed)."""
    a, b = _xy(x, y)
    n = a.size
    sx = sigma if sigma is not None else _median_bandwidth(a)
    sy = sigma if sigma is not None else _median_bandwidth(b)
    K = _rbf(a, sx)
    L = _rbf(b, sy)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H

    def stat_of(M: Array) -> float:
        return float(np.trace(Kc @ (H @ M @ H)) / (n * n))

    stat = stat_of(L)
    rng = np.random.default_rng(seed)
    count = 1.0  # observed counts as one permutation
    for _ in range(n_perm):
        Lp = L[rng.permutation(n)][:, rng.permutation(n)]
        # Same permutation on rows and cols preserves the kernel geometry.
        count += float(stat_of(Lp) >= stat)
    p = count / (n_perm + 1.0)
    return {"hsic": stat, "pvalue": p, "n": float(n), "sigma_x": sx, "sigma_y": sy}


def mmd(x: Array, y: Array, sigma: float | None = None) -> dict[str, float]:
    """Gretton et al. (2012) two-sample MMD with RBF kernel (biased est.).

    Tests whether x and y are drawn from the same distribution.
    ``MMD^2 = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]``.
    """
    a, b = _xy(x, y)
    n = a.size
    s = sigma if sigma is not None else _median_bandwidth(np.concatenate([a, b]))
    Kxx = _rbf(a, s)
    Kyy = _rbf(b, s)
    Kxy = _rbf2(a, b, s)
    stat = float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
    # Null variance (Gretton 2012, linear-time bound): ~ 4 sigma_k^2/n.
    var = max(4.0 * float(np.var(Kxy)) / n, 1e-20)
    p = float(1.0 - stats.norm.cdf(stat / math.sqrt(var)))
    return {"mmd2": stat, "pvalue": p, "n": float(n), "sigma": s}


def _rbf2(a: Array, b: Array, sigma: float) -> Array:
    d2 = (a[:, None] - b[None, :]) ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma))


def chatterjee_xi(x: Array, y: Array) -> dict[str, float]:
    """Chatterjee (2021) xi_n rank correlation.

    Sorts by x, then measures how much the y-ranks jump between
    consecutive x-values. xi in [~0, 1]; ~0 under independence, ~1 under
    noiseless functional dependence. Asymptotic: sqrt(n)*xi -> N(0, 2/5)
    under independence.
    """
    a, b = _xy(x, y, n=30)
    n = a.size
    order = np.argsort(a, kind="stable")
    # Ranks of y with average tie handling.
    ry = stats.rankdata(b, method="average")
    r_sorted = ry[order]
    num = float(np.sum(np.abs(np.diff(r_sorted))))
    # Max possible for random: n^2/3; normalization: xi = 1 - num/(n^2/3).
    xi = 1.0 - num * 3.0 / (n * n - 1.0)
    var = 2.0 / 5.0 / n
    z = float(xi / math.sqrt(var))
    p = float(1.0 - stats.norm.cdf(z))
    return {"xi": float(xi), "statistic": z, "pvalue": p, "n": float(n)}


def mutual_information_knn(x: Array, y: Array, k: int = 5) -> dict[str, float]:
    """Kraskov–Stogbauer–Grassberger (2004) kNN mutual information.

    ``I = psi(k) - <psi(nx+1) + psi(ny+1)> + psi(n)`` using max-norm
    joint-space neighbor counts. Positive values indicate dependence.
    """
    a, b = _xy(x, y, n=30)
    n = a.size
    if not (1 <= k < n):
        raise ValueError("k must be in [1, n)")
    pts = np.column_stack([a, b])
    # Scale to comparable units.
    pts = pts / pts.std(axis=0)
    tree = cKDTree(pts, leafsize=16)
    # For each point, distance to k-th neighbor in joint space (max norm ~
    # Chebyshev): use p=inf.
    dists, _ = tree.query(pts, k=k + 1, p=np.inf)
    eps = np.maximum(dists[:, k], 1e-12)
    tx = cKDTree(a[:, None], leafsize=16)
    ty = cKDTree(b[:, None], leafsize=16)
    nx = np.array([len(tx.query_ball_point(a[i], eps[i] - 1e-15, p=np.inf)) - 1 for i in range(n)])
    ny = np.array([len(ty.query_ball_point(b[i], eps[i] - 1e-15, p=np.inf)) - 1 for i in range(n)])
    mi = float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1.0) + digamma(ny + 1.0)))
    return {"mi": mi, "k": float(k), "n": float(n)}
