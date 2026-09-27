"""Heston (1993) call via the Albrecher little-trap characteristic function.

davidalmeida90/quant-models ``heston-vol-surface/heston_vol_surface.py``.
Default parameters match that notebook's reference set. Research surface
engine; not a live vol mark.
"""

from __future__ import annotations

import numpy as np
from numpy import trapezoid
from numpy.typing import ArrayLike

from quant_fund.quant_models.black_scholes import implied_volatility

S0_REF = 100.0
R_REF = 0.03
Q_REF = 0.0
V0_REF = 0.019
KAPPA_REF = 2.5
THETA_REF = 0.07
XI_REF = 0.75
RHO_REF = -0.72


def heston_char(
    u: ArrayLike,
    T: float,
    S0: float,
    r: float,
    q: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> np.ndarray:
    """Albrecher 'little trap' Heston CF of ``log S_T``."""
    u_c = np.asarray(u, dtype=np.complex128)
    xi_h = kappa - rho * xi * 1j * u_c
    d = np.sqrt(xi_h**2 + xi**2 * (1j * u_c + u_c**2))
    g2 = (xi_h - d) / (xi_h + d)
    exp_dt = np.exp(-d * T)
    c = (kappa * theta / xi**2) * ((xi_h - d) * T - 2.0 * np.log((1.0 - g2 * exp_dt) / (1.0 - g2)))
    d_term = ((xi_h - d) / xi**2) * (1.0 - exp_dt) / (1.0 - g2 * exp_dt)
    return np.asarray(np.exp(c + d_term * v0 + 1j * u_c * (np.log(S0) + (r - q) * T)))


def heston_call(
    K: float,
    T: float,
    *,
    S0: float = S0_REF,
    r: float = R_REF,
    q: float = Q_REF,
    v0: float = V0_REF,
    kappa: float = KAPPA_REF,
    theta: float = THETA_REF,
    xi: float = XI_REF,
    rho: float = RHO_REF,
    u_max: float = 100.0,
    n_u: int = 1024,
) -> float:
    """European Heston call from P1/P2 recovered by a trapezoid CF integral."""
    if T <= 0 or K <= 0 or S0 <= 0:
        raise ValueError("S0, K, T must be positive")
    if xi <= 0:
        raise ValueError("xi (vol of vol) must be positive")
    args = dict(S0=S0, r=r, q=q, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
    u = np.linspace(1e-8, u_max, int(n_u))
    log_k = np.log(K)
    phi_m1 = heston_char(-1j, T, **args)
    integrand_p1 = np.real(
        np.exp(-1j * u * log_k) * heston_char(u - 1j, T, **args) / (1j * u * phi_m1)
    )
    integrand_p2 = np.real(np.exp(-1j * u * log_k) * heston_char(u, T, **args) / (1j * u))
    p1 = 0.5 + (1.0 / np.pi) * float(trapezoid(integrand_p1, u))
    p2 = 0.5 + (1.0 / np.pi) * float(trapezoid(integrand_p2, u))
    return float(S0 * np.exp(-q * T) * p1 - K * np.exp(-r * T) * p2)


def heston_put(
    K: float,
    T: float,
    **kwargs: float,
) -> float:
    """Put via put-call parity on ``heston_call``."""
    s0 = float(kwargs.get("S0", S0_REF))
    r = float(kwargs.get("r", R_REF))
    q = float(kwargs.get("q", Q_REF))
    call = heston_call(K, T, **kwargs)  # type: ignore[arg-type]
    return float(call - s0 * np.exp(-q * T) + K * np.exp(-r * T))


def heston_implied_vol(K: float, T: float, **kwargs: float) -> float:
    """Black implied vol of the Heston call."""
    s0 = float(kwargs.get("S0", S0_REF))
    r = float(kwargs.get("r", R_REF))
    q = float(kwargs.get("q", Q_REF))
    price = heston_call(K, T, **kwargs)  # type: ignore[arg-type]
    return implied_volatility(s0, K, T, r, q, price, "call")
