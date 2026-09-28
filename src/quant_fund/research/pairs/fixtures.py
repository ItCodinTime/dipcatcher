"""Labeled SYNTHETIC fixtures for the pairs/stat-arb research pack.

All generators are seeded and deterministic. Nothing here is market data;
every payload is correctness evidence only (``data_label == "SYNTHETIC"``).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def ar1_spread(n: int = 600, rho: float = 0.90, sigma: float = 0.15, seed: int = 0) -> Array:
    """Stationary AR(1) / discrete-OU spread with a known half-life.

    ``s_t = rho * s_{t-1} + e_t``, ``e ~ N(0, sigma^2)`` — the known answer
    for the OU half-life estimator is ``-ln(2) / ln(rho)``.
    """
    n_i = int(n)
    if n_i < 20:
        raise ValueError("n must be >= 20")
    if not 0.0 < float(rho) < 1.0 or float(sigma) <= 0:
        raise ValueError("need 0 < rho < 1 and sigma > 0")
    rng = np.random.default_rng(int(seed))
    eps = rng.normal(0.0, float(sigma), size=n_i)
    s = np.empty(n_i)
    s[0] = 0.0
    for t in range(1, n_i):
        s[t] = float(rho) * s[t - 1] + eps[t]
    return s


def planted_pair_panel(
    n_assets: int = 8,
    n_dates: int = 600,
    seed: int = 0,
    *,
    pair: tuple[int, int] = (0, 1),
    spread_rho: float = 0.90,
    spread_sigma: float = 0.15,
    idio_sigma: float = 0.02,
) -> dict[str, object]:
    """Panel with ONE planted cointegrated pair on a shared I(1) factor.

    ``logA = f + eps_A`` and ``logB = f + s + eps_B`` where ``f`` is a common
    random walk and ``s`` a stationary AR(1) — so ``logA - logB`` is
    stationary with known half-life and hedge ratio 1. Every other column
    is an independent random walk. Returns the price panel plus the planted
    ground truth; labeled SYNTHETIC.
    """
    n_a, n_t = int(n_assets), int(n_dates)
    i, j = int(pair[0]), int(pair[1])
    if n_a < 3 or n_t < 120 or not (0 <= i < n_a and 0 <= j < n_a) or i == j:
        raise ValueError("bad panel dims or pair indices")
    rng = np.random.default_rng(int(seed))
    factor = np.cumsum(rng.normal(0.0, 1.0, size=n_t))
    spread = ar1_spread(n_t, spread_rho, spread_sigma, seed=int(seed) + 1)
    logs = np.empty((n_t, n_a))
    for k in range(n_a):
        if k == i:
            logs[:, k] = factor + rng.normal(0.0, idio_sigma, size=n_t)
        elif k == j:
            logs[:, k] = factor + spread + rng.normal(0.0, idio_sigma, size=n_t)
        else:
            logs[:, k] = np.cumsum(rng.normal(0.0, 1.0, size=n_t))
    return {
        "prices": np.exp(logs),
        "log_prices": logs,
        "pair": (i, j),
        "common_factor": factor,
        "planted_spread": spread,
        "spread_rho": float(spread_rho),
        "true_half_life": -math.log(2.0) / math.log(float(spread_rho)),
        "seed": int(seed),
        "data_label": "SYNTHETIC",
    }


def independent_walks_panel(
    n_assets: int = 6,
    n_dates: int = 600,
    seed: int = 0,
) -> dict[str, object]:
    """Null panel: all columns independent random walks (no cointegration).

    Used to sanity-check familywise/false-discovery behavior — nothing
    here is truly cointegrated, so any BH rejection is a false positive by
    construction. Labeled SYNTHETIC.
    """
    n_a, n_t = int(n_assets), int(n_dates)
    if n_a < 2 or n_t < 120:
        raise ValueError("bad panel dims")
    rng = np.random.default_rng(int(seed))
    logs = np.cumsum(rng.normal(0.0, 1.0, size=(n_t, n_a)), axis=0)
    return {
        "prices": np.exp(logs),
        "log_prices": logs,
        "seed": int(seed),
        "data_label": "SYNTHETIC",
    }
