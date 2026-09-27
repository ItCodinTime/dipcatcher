"""Entropic risk measure and Entropic Value-at-Risk (EVaR).

The entropic risk measure is the exponential-utility certainty equivalent

    rho_theta(L) = (1 / theta) * log E[exp(theta L)],   theta > 0,

for a loss ``L`` (positive-is-loss).  It is convex and law-invariant but not
coherent (not positively homogeneous).

Ahmadi-Javid (2012) introduced the Entropic Value-at-Risk, the tightest
coherent upper bound on both VaR and CVaR derived from the Chernoff bound:

    EVaR_{1-alpha}(L) = inf_{z > 0} (1/z) * log( E[exp(z L)] / alpha ),

where ``alpha`` is the tail probability (confidence ``1 - alpha``).  It
satisfies ``EVaR >= CVaR >= VaR``.  The infimum is found by bounded scalar
minimisation of a quasi-convex objective.

References: H. Follmer, A. Schied (2011), *Stochastic Finance*; A. Ahmadi-Javid
(2012), J. Optimization Theory and Applications.  Fail-closed on non-finite
input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

Array = NDArray[np.float64]


def _as_losses(losses: Array, min_obs: int = 5) -> Array:
    arr = np.asarray(losses, dtype=float).ravel()
    if arr.size < min_obs or not np.isfinite(arr).all():
        raise ValueError(f"losses must be finite with >= {min_obs} observations")
    return arr


def _log_mgf(losses: Array, z: float) -> float:
    """Numerically stable log E[exp(z L)] using the log-sum-exp shift."""
    a = z * losses
    amax = float(a.max())
    return amax + float(np.log(np.mean(np.exp(a - amax))))


def entropic_risk_measure(losses: Array, theta: float = 1.0) -> float:
    """Entropic (exponential-utility) risk measure, ``theta > 0``."""
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    arr = _as_losses(losses)
    return _log_mgf(arr, theta) / theta


def entropic_value_at_risk(losses: Array, alpha: float = 0.95) -> dict[str, float]:
    """Ahmadi-Javid (2012) EVaR at confidence ``alpha`` (tail prob ``1-alpha``)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    arr = _as_losses(losses)
    tail = 1.0 - alpha

    def obj(log_z: float) -> float:
        z = float(np.exp(log_z))
        return (_log_mgf(arr, z) - float(np.log(tail))) / z

    res = minimize_scalar(obj, bounds=(-12.0, 12.0), method="bounded")
    z_star = float(np.exp(res.x))
    return {"evar": float(res.fun), "z": z_star, "alpha": float(alpha)}
