"""Merton (1974) / KMV distance-to-default.

Equity is a call option on firm asset value V struck at the face
value of debt D:

    E = V N(d1) - D e^{-rT} N(d2)
    sigma_E = (V / E) N(d1) sigma_V        (Ito leverage relation)

with d1 = [ln(V/D) + (r + sigma_V^2/2) T] / (sigma_V sqrt(T)).

Solve the 2x2 nonlinear system for (V, sigma_V) given observed
(E, sigma_E, D, r, T); then DD = (ln(V/D) + (mu_V - sigma_V^2/2)T)
/ (sigma_V sqrt(T)) and the risk-neutral-style PD = N(-DD_simple)
where DD_simple = ln(V/D)/(sigma_V sqrt(T)).

Fail-closed: non-positive inputs, solver non-convergence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

Array = NDArray[np.float64]


def merton_kmv_solve(
    equity: Array,
    sigma_equity: Array,
    debt: Array,
    r: float = 0.0,
    t_horizon: float = 1.0,
) -> dict[str, Array]:
    """Solve for asset value V and asset vol sigma_V per observation.

    All inputs may be scalars or equal-length arrays.
    """
    e = np.atleast_1d(np.asarray(equity, dtype=float))
    se = np.atleast_1d(np.asarray(sigma_equity, dtype=float))
    d = np.atleast_1d(np.asarray(debt, dtype=float))
    n = max(e.size, se.size, d.size)
    e = np.broadcast_to(e, (n,))
    se = np.broadcast_to(se, (n,))
    d = np.broadcast_to(d, (n,))
    if (e <= 0).any() or (se <= 0).any() or (d <= 0).any():
        raise ValueError("equity, sigma_equity, debt must be positive")
    if t_horizon <= 0 or not np.isfinite(r):
        raise ValueError("bad horizon/rate")

    v_out = np.empty(n)
    sv_out = np.empty(n)
    for i in range(n):
        ei, si, di = float(e[i]), float(se[i]), float(d[i])

        def eqs(th: Array, di: float = di, ei: float = ei, si: float = si) -> list[float]:
            v, sv = np.exp(th[0]), np.exp(th[1])  # positivity via log-params
            sqt = sv * np.sqrt(t_horizon)
            d1 = (np.log(v / di) + (r + 0.5 * sv * sv) * t_horizon) / sqt
            d2 = d1 - sqt
            f1 = v * stats.norm.cdf(d1) - di * np.exp(-r * t_horizon) * stats.norm.cdf(d2) - ei
            f2 = (v / ei) * stats.norm.cdf(d1) * sv - si
            return [float(f1), float(f2)]

        th0 = np.array([np.log(ei + di), np.log(max(si * ei / (ei + di), 1e-4))])
        res = optimize.root(eqs, th0, method="hybr")
        if not res.success:
            raise ValueError(f"KMV solve failed at obs {i}: {res.message}")
        v_out[i] = np.exp(res.x[0])
        sv_out[i] = np.exp(res.x[1])

    sqt = sv_out * np.sqrt(t_horizon)
    dd = np.log(v_out / d) / sqt
    pd = stats.norm.cdf(-dd)
    return {
        "V": v_out,
        "sigma_V": sv_out,
        "DD": dd,
        "PD": pd,
    }
