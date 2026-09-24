"""Model-free variance swap fair strike via option replication.

A variance swap's fair strike equals the risk-neutral expected realised
variance, which is replicated by a static portfolio of out-of-the-money options
(the "log contract").  Using the Demeterfi-Derman-Kamal-Zou (1999) discretisation
with strikes ``K_i``, spacings ``dK_i`` and the forward ``F`` (``K0`` the largest
strike at or below ``F``):

    K_var = (2/T) e^{rT} [ sum_{K_i < K0} (dK_i/K_i^2) P(K_i)
                          + sum_{K_i >= K0} (dK_i/K_i^2) C(K_i) ]
            - (1/T) (F/K0 - 1)^2,

with puts used below ``K0`` and calls at/above it (out-of-the-money).

References: A. Neuberger (1994); K. Demeterfi, E. Derman, M. Kamal, J. Zou
(1999); P. Carr, D. Madan (1998).  Fail-closed on invalid inputs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _strike_spacings(strikes: Array) -> Array:
    k = strikes
    dk = np.empty_like(k)
    dk[1:-1] = (k[2:] - k[:-2]) / 2.0
    dk[0] = k[1] - k[0]
    dk[-1] = k[-1] - k[-2]
    return dk


def variance_swap_fair_strike(
    strikes: Array,
    call_prices: Array,
    put_prices: Array,
    forward: float,
    r: float,
    maturity: float,
) -> dict[str, float]:
    """Demeterfi et al. (1999) fair variance strike from an OTM option strip."""
    k = np.asarray(strikes, dtype=float).ravel()
    calls = np.asarray(call_prices, dtype=float).ravel()
    puts = np.asarray(put_prices, dtype=float).ravel()
    if not (k.size == calls.size == puts.size) or k.size < 5:
        raise ValueError("strikes, calls, puts must share length >= 5")
    if not (np.all(np.diff(k) > 0) and np.isfinite(k).all()):
        raise ValueError("strikes must be strictly increasing and finite")
    if forward <= 0.0 or maturity <= 0.0:
        raise ValueError("forward and maturity must be positive")
    below = k[k <= forward]
    if below.size == 0:
        raise ValueError("need at least one strike at or below the forward")
    k0 = float(below.max())
    dk = _strike_spacings(k)
    otm = np.where(k < k0, puts, calls)
    weights = dk / k**2
    strip = float(np.sum(weights * otm))
    fair_var = (2.0 / maturity) * np.exp(r * maturity) * strip - (1.0 / maturity) * (
        forward / k0 - 1.0
    ) ** 2
    return {
        "fair_variance": float(fair_var),
        "fair_vol": float(np.sqrt(max(fair_var, 0.0))),
        "k0": k0,
    }
