"""Risk-based weights: inverse-vol, equal-risk contribution.

davidalmeida90/quant-models ``risk-based-allocation``. Long-only.
Research allocation; the live optimizer remains ``optimize_mean_variance``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

Array = NDArray[np.float64]


def inverse_vol_weights(cov: ArrayLike) -> Array:
    v = np.asarray(cov, dtype=float)
    vol = np.sqrt(np.diag(v))
    if np.any(vol <= 0) or not np.isfinite(vol).all():
        raise ValueError("vols must be positive and finite")
    w = 1.0 / vol
    return np.asarray(w / w.sum(), dtype=float)


def equal_risk_contribution(cov: ArrayLike) -> Array:
    """Long-only ERC: ``w_i (Σ w)_i`` equal across names."""
    v = np.asarray(cov, dtype=float)
    n = v.shape[0]
    if n < 2:
        return np.ones(n, dtype=float)

    def obj(w: np.ndarray) -> float:
        rc = w * (v @ w)
        target = rc.mean()
        return float(np.sum((rc - target) ** 2))

    res = minimize(
        obj,
        np.repeat(1.0 / n, n),
        method="SLSQP",
        bounds=[(1e-8, 1.0)] * n,
        constraints=({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},),
        options={"ftol": 1e-14, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"ERC failed: {res.message}")
    w = np.maximum(res.x, 0.0)
    return np.asarray(w / w.sum(), dtype=float)


def risk_contributions(weights: ArrayLike, cov: ArrayLike) -> Array:
    w = np.asarray(weights, dtype=float)
    v = np.asarray(cov, dtype=float)
    rc = w * (v @ w)
    port_var = float(w @ v @ w)
    if port_var <= 0:
        raise ValueError("portfolio variance must be positive")
    return np.asarray(rc / port_var, dtype=float)
