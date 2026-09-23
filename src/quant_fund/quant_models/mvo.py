"""Simple long-only mean-variance (quant-models ``mvo-portfolio-optimization``).

Does not replace ``quant_fund.portfolio.optimizer.optimize_mean_variance``,
which is the live-constrained CVXPY book. This is the unconstrained-style
research QP from the notebooks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

Array = NDArray[np.float64]


def long_only_mean_variance(
    mu: ArrayLike,
    cov: ArrayLike,
    *,
    risk_aversion: float = 1.0,
) -> Array:
    """Max ``μ'w − (λ/2) w'Σw`` subject to ``w>=0``, ``1'w=1``."""
    m = np.asarray(mu, dtype=float).reshape(-1)
    v = np.asarray(cov, dtype=float)
    n = m.size
    if v.shape != (n, n):
        raise ValueError("cov must match mu")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")

    def obj(w: np.ndarray) -> float:
        return float(-m @ w + 0.5 * risk_aversion * (w @ v @ w))

    res = minimize(
        obj,
        np.repeat(1.0 / n, n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},),
        options={"ftol": 1e-12, "maxiter": 400},
    )
    if not res.success:
        raise RuntimeError(f"MVO failed: {res.message}")
    w = np.maximum(res.x, 0.0)
    return np.asarray(w / w.sum(), dtype=float)
