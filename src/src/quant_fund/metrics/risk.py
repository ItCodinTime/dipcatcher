"""VaR and Expected Shortfall. Loss L = -R. See docs/MATH_SPEC.md.

Empty / all-non-finite inputs → honest NaN. Alpha outside (0, 1) or
non-finite → fail-closed ValueError. Research / risk diagnostics only —
not a live P&L claim.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _require_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def losses_from_returns(returns: Array) -> Array:
    """Map returns to losses ``L = -R``. Empty input → empty array."""
    return -np.asarray(returns, dtype=float)


def historical_var(losses: Array, alpha: float = 0.95) -> float:
    """Empirical VaR at level ``alpha`` on losses. Empty/all-NaN → NaN."""
    _require_alpha(alpha)
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, alpha))


def historical_es(losses: Array, alpha: float = 0.95) -> float:
    """Expected loss in the upper empirical tail, including fractional mass.

    The empirical distribution assigns mass ``1 / n`` to each observation.  If
    the requested tail contains a non-integer number of observations, the
    boundary observation is weighted only by the remaining tail probability.
    This is the discrete Acerbi--Tasche definition and handles ties without
    silently changing the requested tail probability.

    Empty / all-non-finite losses → honest NaN. Bad alpha → ValueError.
    """
    _require_alpha(alpha)
    x = np.asarray(losses, dtype=float)
    x = np.sort(x[np.isfinite(x)])[::-1]
    if x.size == 0:
        return float("nan")
    tail_mass = (1.0 - alpha) * x.size
    remaining = tail_mass
    weighted_sum = 0.0
    for value in x:
        mass = min(1.0, remaining)
        weighted_sum += mass * float(value)
        remaining -= mass
        if remaining <= 1e-12:
            break
    return float(weighted_sum / tail_mass)


def gaussian_var(losses: Array, alpha: float = 0.95) -> float:
    """Parametric Gaussian VaR. Needs ≥2 finite losses; else honest NaN."""
    from scipy.stats import norm

    _require_alpha(alpha)
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    mu = float(np.mean(x))
    sig = float(np.std(x, ddof=1))
    return float(mu + sig * norm.ppf(alpha))


def gaussian_es(losses: Array, alpha: float = 0.95) -> float:
    """Parametric Gaussian ES. Needs ≥2 finite losses; else honest NaN."""
    from scipy.stats import norm

    _require_alpha(alpha)
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    mu = float(np.mean(x))
    sig = float(np.std(x, ddof=1))
    z = float(norm.ppf(alpha))
    return float(mu + sig * norm.pdf(z) / (1.0 - alpha))


def var_es_from_return_quantiles(
    quantile_map: dict[float, float], alpha: float = 0.95
) -> tuple[float, float]:
    """Map return quantiles to loss VaR/ES using quantile-function integration.

    ``quantile_map`` contains return quantiles ``q_tau``.  Loss VaR is
    ``-q_(1-alpha)`` and loss ES is the negative average of the return
    quantile function over ``[0, 1-alpha]``.  Integrating the piecewise-linear
    quantile curve respects unevenly spaced quantiles and fractional endpoints.

    Empty / all-non-finite / non-monotone tau → honest (NaN, NaN).
    Bad alpha → ValueError.
    """
    _require_alpha(alpha)
    if not quantile_map:
        return float("nan"), float("nan")
    taus = np.asarray(sorted(quantile_map), dtype=float)
    qs = np.asarray([quantile_map[float(t)] for t in taus], dtype=float)
    valid = np.isfinite(taus) & np.isfinite(qs) & (taus >= 0.0) & (taus <= 1.0)
    taus, qs = taus[valid], qs[valid]
    if taus.size == 0 or np.any(np.diff(taus) <= 0):
        return float("nan"), float("nan")

    ret_level = 1.0 - alpha
    var_return = float(np.interp(ret_level, taus, qs))
    if ret_level <= taus[0]:
        area = ret_level * qs[0]
    else:
        knots = [0.0]
        values = [float(qs[0])]
        knots.extend(float(t) for t in taus if 0.0 < t < ret_level)
        values.extend(float(q) for t, q in zip(taus, qs, strict=True) if 0.0 < t < ret_level)
        knots.append(ret_level)
        values.append(var_return)
        area = float(np.trapezoid(np.asarray(values), np.asarray(knots)))
    return -float(var_return), -area / ret_level
