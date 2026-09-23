"""Bootstrap and subsampling schemes for time-series inference.

Complements ``metrics/inference.py`` (which has generic block-bootstrap
CI helpers) with the resampling plans themselves, each appropriate to a
different dependence/heteroskedasticity structure.

References:
- Wu (1986) / Mammen (1993): wild bootstrap for heteroskedastic errors.
- Freedman (1981) / Efron (1979): pairs (iid case) bootstrap.
- Buhlmann (1997): sieve bootstrap via AR approximation.
- Politis & Romano (1994): subsampling for generic statistics.
- MacKinnon (2009): bootstrap hypothesis testing practice.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _v(x: Array, n: int = 10) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"series must be finite with length >= {n}")
    return v


def wild_bootstrap_residuals(residuals: Array, n_boot: int = 500, seed: int = 0) -> Array:
    """Mammen (1993) two-point wild bootstrap: ``e*_t = e_t * v_t`` where
    ``v_t`` takes {-1, +1} scaled to reproduce the first three moments:
    P(v = (1+sqrt5)/2) = (sqrt5-1)/(2sqrt5). Returns (n_boot, n).
    """
    e = _v(residuals, n=5)
    n = e.size
    rng = np.random.default_rng(seed)
    a = (1.0 + math.sqrt(5.0)) / 2.0
    b = (1.0 - math.sqrt(5.0)) / 2.0
    p = (math.sqrt(5.0) - 1.0) / (2.0 * math.sqrt(5.0))
    v = np.where(rng.random((n_boot, n)) < p, a, b)
    return e[None, :] * v


def rademacher_bootstrap(residuals: Array, n_boot: int = 500, seed: int = 0) -> Array:
    """Rademacher wild bootstrap: e*_t = e_t * {+-1} with equal probability."""
    e = _v(residuals, n=5)
    n = e.size
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    return e[None, :] * signs


def pairs_bootstrap(y: Array, X: Array, n_boot: int = 500, seed: int = 0) -> Array:
    """Freedman/Efron pairs bootstrap for regression: resample (y_i, x_i)
    jointly, refit OLS, return (n_boot, k) coefficient paths."""
    yv = _v(y, n=10)
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim != 2 or Xa.shape[0] != yv.size or not np.all(np.isfinite(Xa)):
        raise ValueError("X must be finite (n, k) matching y")
    n, k = Xa.shape
    if k >= n:
        raise ValueError("underdetermined design")
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, k))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        beta, *_ = np.linalg.lstsq(Xa[idx], yv[idx], rcond=None)
        out[b] = beta
    return out


def sieve_bootstrap_residuals(
    residuals: Array, order: int = 1, n_boot: int = 500, seed: int = 0
) -> Array:
    """Buhlmann (1997) sieve bootstrap: fit AR(p) to the residuals, then
    resample the AR innovations iid and regenerate AR paths. Preserves
    linear autocorrelation. Returns (n_boot, n).
    """
    e = _v(residuals, n=20)
    n = e.size
    p = max(1, int(order))
    if p >= n // 4:
        raise ValueError("order too large for series length")
    ec = e - e.mean()
    # Yule–Walker AR(p) fit on sample autocovariances.
    acov = np.array([np.dot(ec[: n - h], ec[h:]) / n for h in range(p + 1)])
    R = np.array([[acov[abs(i - j)] for j in range(p)] for i in range(p)])
    try:
        phi = np.linalg.solve(R + 1e-10 * np.eye(p), acov[1:])
    except np.linalg.LinAlgError as exc:
        raise ValueError("AR fit singular") from exc
    innov = ec[p:] - np.sum(
        np.stack([phi[k] * ec[p - 1 - k : n - 1 - k] for k in range(p)]), axis=0
    )
    if innov.size < 5 or not np.all(np.isfinite(innov)):
        raise ValueError("degenerate AR innovations")
    innov = innov - innov.mean()
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, n))
    for b in range(n_boot):
        u = rng.choice(innov, size=n, replace=True)
        path = np.empty(n)
        path[:p] = ec[:p]
        for t in range(p, n):
            path[t] = float(np.dot(phi, path[t - p : t][::-1])) + u[t]
        out[b] = path + e.mean()
    return out


def subsample_statistic(
    series: Array,
    stat_fn,
    block_len: int,
    overlapping: bool = True,
) -> dict[str, Array | float]:
    """Politis–Romano (1994) subsampling distribution of a statistic.

    Evaluates ``stat_fn`` on every contiguous block of length
    ``block_len`` (all n-b+1 blocks if overlapping, else the floor
    partition). Returns the empirical distribution and the subsampling
    CI percentiles. The caller uses ``quantiles`` for intervals; the
    statistic on the full sample is also returned.
    """
    v = _v(series, n=10)
    n = v.size
    if not (2 <= block_len < n):
        raise ValueError("block_len must be in [2, n)")
    if overlapping:
        blocks = [v[i : i + block_len] for i in range(n - block_len + 1)]
    else:
        m = n // block_len
        blocks = [v[i * block_len : (i + 1) * block_len] for i in range(m)]
    if len(blocks) < 3:
        raise ValueError("too few subsample blocks")
    stats_b = np.array([float(stat_fn(b)) for b in blocks])
    if not np.all(np.isfinite(stats_b)):
        raise ValueError("statistic produced nonfinite values on blocks")
    return {
        "sub_stats": stats_b,
        "full_stat": float(stat_fn(v)),
        "q025": float(np.quantile(stats_b, 0.025)),
        "q975": float(np.quantile(stats_b, 0.975)),
        "n_blocks": float(len(blocks)),
    }


def circular_block_indices(n: int, block_len: int, n_boot: int, seed: int = 0) -> NDArray[np.int64]:
    """Politis–Romano (1994) circular block bootstrap indices: wrap-around
    blocks, returns (n_boot, n) index matrix."""
    if not (1 <= block_len <= n):
        raise ValueError("block_len must be in [1, n]")
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_len))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % n
    return idx.reshape(n_boot, n_blocks * block_len)[:, :n]
