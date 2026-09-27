"""Nonparametric two-sample tests.

- Two-sample Kolmogorov-Smirnov test (Smirnov 1939): the maximum gap between the
  empirical CDFs, with the asymptotic Kolmogorov p-value.
- Energy distance and its permutation test (Szekely & Rizzo 2004, 2013):
  ``E = 2 A - B - C`` with ``A = mean|x_i - y_j|`` and within-sample averages
  ``B``, ``C``; the scaled statistic is zero iff the distributions match.
- Two-sample Cramer-von Mises test (Anderson 1962) via permutation.
- A generic two-sample permutation test.

Fail-closed on non-finite input or too little data.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import kolmogorov

Array = NDArray[np.float64]


def _two(x: Array, y: Array) -> tuple[Array, Array]:
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.size < 5 or ya.size < 5 or not (np.isfinite(xa).all() and np.isfinite(ya).all()):
        raise ValueError("x and y must be finite with >= 5 observations each")
    return xa, ya


def ks_two_sample(x: Array, y: Array) -> dict[str, float]:
    """Two-sample Kolmogorov-Smirnov statistic with asymptotic p-value."""
    xa, ya = _two(x, y)
    nx, ny = xa.size, ya.size
    grid = np.concatenate([xa, ya])
    cdf_x = np.searchsorted(np.sort(xa), grid, side="right") / nx
    cdf_y = np.searchsorted(np.sort(ya), grid, side="right") / ny
    d = float(np.max(np.abs(cdf_x - cdf_y)))
    ne = nx * ny / (nx + ny)
    p = float(kolmogorov((np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d))
    return {"statistic": d, "pvalue": min(max(p, 0.0), 1.0), "n_x": float(nx), "n_y": float(ny)}


def energy_distance(x: Array, y: Array) -> float:
    """Szekely-Rizzo energy distance E = 2A - B - C (>= 0)."""
    xa, ya = _two(x, y)
    a = float(np.mean(np.abs(xa[:, None] - ya[None, :])))
    b = float(np.mean(np.abs(xa[:, None] - xa[None, :])))
    c = float(np.mean(np.abs(ya[:, None] - ya[None, :])))
    return max(2.0 * a - b - c, 0.0)


def permutation_test(
    x: Array,
    y: Array,
    statistic: Callable[[Array, Array], float],
    n_perm: int = 999,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Generic two-sample permutation test (larger statistic = more extreme)."""
    xa, ya = _two(x, y)
    gen = np.random.default_rng() if rng is None else rng
    observed = statistic(xa, ya)
    pooled = np.concatenate([xa, ya])
    nx = xa.size
    count = 1
    for _ in range(n_perm):
        gen.shuffle(pooled)
        if statistic(pooled[:nx], pooled[nx:]) >= observed:
            count += 1
    return {"statistic": float(observed), "pvalue": count / (n_perm + 1)}


def energy_test(
    x: Array, y: Array, n_perm: int = 499, rng: np.random.Generator | None = None
) -> dict[str, float]:
    """Permutation test based on the energy distance."""
    return permutation_test(x, y, energy_distance, n_perm=n_perm, rng=rng)


def _cvm_statistic(x: Array, y: Array) -> float:
    nx, ny = x.size, y.size
    pooled = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(pooled)) + 1
    rx = ranks[:nx]
    ry = ranks[nx:]
    i = np.arange(1, nx + 1)
    j = np.arange(1, ny + 1)
    u = nx * float(np.sum((np.sort(rx) - i) ** 2)) + ny * float(np.sum((np.sort(ry) - j) ** 2))
    n = nx + ny
    return u / (nx * ny * n) - (4.0 * nx * ny - 1.0) / (6.0 * n)


def cramer_von_mises_2samp(
    x: Array, y: Array, n_perm: int = 499, rng: np.random.Generator | None = None
) -> dict[str, float]:
    """Two-sample Cramer-von Mises test (Anderson 1962) via permutation."""
    return permutation_test(x, y, _cvm_statistic, n_perm=n_perm, rng=rng)
