"""Michaud (1998) resampled efficiency for portfolio construction.

Plug-in mean-variance optimisation over-fits estimation error and produces
concentrated, unstable weights.  Resampled efficiency averages the optimiser
over Monte-Carlo resamples of the estimated return distribution:

1. draw ``n_obs`` synthetic returns from ``N(mu, cov)``;
2. re-estimate ``mu_hat``, ``cov_hat`` and solve the mean-variance problem;
3. repeat and average the optimal weights.

The averaged weights are more diversified and more stable out of sample.

References: R. Michaud (1998), *Efficient Asset Management*; R. & R. Michaud
(2008).  Long-only (simplex) portfolios by default.  Fail-closed on shape
mismatch or non-finite input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

Array = NDArray[np.float64]


def _check(mu: Array, cov: Array) -> tuple[Array, Array]:
    m = np.asarray(mu, dtype=float).ravel()
    c = np.asarray(cov, dtype=float)
    n = m.size
    if c.shape != (n, n) or not np.isfinite(m).all() or not np.isfinite(c).all():
        raise ValueError("mu (n,) and cov (n, n) must be finite and conformable")
    if n < 2:
        raise ValueError("need at least two assets")
    return m, c


def _bounds(n: int, long_only: bool) -> list[tuple[float, float]]:
    return [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n


def max_sharpe_weights(mu: Array, cov: Array, long_only: bool = True) -> Array:
    """Single mean-variance (max-Sharpe) solution on the simplex."""
    m, c = _check(mu, cov)
    n = m.size
    x0 = np.full(n, 1.0 / n)
    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)

    def neg_sharpe(w: Array) -> float:
        var = float(w @ c @ w)
        if var <= 0.0:
            return 0.0
        return float(-float(w @ m) / np.sqrt(var))

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=_bounds(n, long_only), constraints=cons)
    w = np.asarray(res.x, dtype=float)
    return np.asarray(w / w.sum(), dtype=float)


def min_variance_weights(cov: Array, long_only: bool = True) -> Array:
    """Single global minimum-variance solution on the simplex."""
    n = np.asarray(cov, dtype=float).shape[0]
    _check(np.zeros(n), cov)
    x0 = np.full(n, 1.0 / n)
    c = np.asarray(cov, dtype=float)
    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    res = minimize(
        lambda w: float(w @ c @ w),
        x0,
        method="SLSQP",
        bounds=_bounds(n, long_only),
        constraints=cons,
    )
    w = np.asarray(res.x, dtype=float)
    return np.asarray(w / w.sum(), dtype=float)


def resampled_weights(
    mu: Array,
    cov: Array,
    n_obs: int = 120,
    n_sims: int = 100,
    objective: str = "max_sharpe",
    long_only: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, Array]:
    """Michaud resampled weights averaged over ``n_sims`` return resamples."""
    m, c = _check(mu, cov)
    if objective not in {"max_sharpe", "min_variance"}:
        raise ValueError("objective must be 'max_sharpe' or 'min_variance'")
    if n_obs < m.size + 2 or n_sims < 2:
        raise ValueError("n_obs must exceed the asset count and n_sims >= 2")
    gen = np.random.default_rng() if rng is None else rng
    acc = np.zeros(m.size)
    for _ in range(n_sims):
        sample = gen.multivariate_normal(m, c, size=n_obs)
        mu_s = sample.mean(axis=0)
        cov_s = np.cov(sample, rowvar=False)
        if objective == "max_sharpe":
            acc += max_sharpe_weights(mu_s, cov_s, long_only)
        else:
            acc += min_variance_weights(cov_s, long_only)
    w = acc / n_sims
    return {"weights": w / w.sum(), "n_sims": np.array([n_sims], dtype=float)}
