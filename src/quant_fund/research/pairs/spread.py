"""Ornstein-Uhlenbeck spread diagnostics and z-score signal bands.

Pure trailing-window statistics. ``zscore_trailing`` and ``bands_position``
only ever read observations at or before the output index.

References:
- Vidyamurthy (2004): AR(1) half-life of the cointegrated spread.
- Ornstein & Uhlenbeck (1930); Chan (2013): AR(1) <-> OU bridge
  ``half_life = -ln(2) / ln(rho)``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _series(x: Array, name: str, minimum: int = 10) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < minimum or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite series with >= {minimum} obs")
    return v


def ar1_fit(spread: Array) -> tuple[float, float]:
    """AR(1) ``s_t = c + rho * s_{t-1} + e_t`` by OLS. Returns ``(c, rho)``."""
    v = _series(spread, "spread")
    design = np.column_stack([np.ones(v.size - 1), v[:-1]])
    b, *_ = np.linalg.lstsq(design, v[1:], rcond=None)
    return float(b[0]), float(b[1])


def ou_half_life(spread: Array) -> float:
    """Half-life of mean reversion ``-ln(2) / ln(rho)`` (Vidyamurthy 2004).

    Returns ``+inf`` when ``rho <= 0`` (oscillating) or ``rho >= 1``
    (non-stationary / explosive) — no finite mean-reversion horizon.
    """
    _, rho = ar1_fit(spread)
    if rho <= 0.0 or rho >= 1.0:
        return math.inf
    return -math.log(2.0) / math.log(rho)


def ou_params(spread: Array) -> dict[str, float]:
    """OU-consistent descriptors of the AR(1) fit (unit time step).

    Maps the discrete AR(1) onto ``dX = theta * (mu - X) dt + sigma dW``:
    ``theta = -ln(rho)``, ``mu = c / (1 - rho)``, ``sigma`` from the AR(1)
    innovation std under the Euler discretization. Descriptive only.
    """
    v = _series(spread, "spread")
    c, rho = ar1_fit(v)
    resid = v[1:] - (c + rho * v[:-1])
    sigma_e = float(np.std(resid))
    if rho <= 0.0 or rho >= 1.0:
        return {
            "rho": rho,
            "theta": math.inf,
            "mu": math.nan,
            "sigma": sigma_e,
            "half_life": math.inf,
        }
    theta = -math.log(rho)
    # Exact OU discretization: Var(e) = sigma^2 * (1 - rho^2) / (2 * theta).
    sigma = sigma_e * math.sqrt(2.0 * theta / max(1.0 - rho * rho, 1e-12))
    return {
        "rho": rho,
        "theta": theta,
        "mu": c / (1.0 - rho),
        "sigma": sigma,
        "half_life": -math.log(2.0) / math.log(rho),
    }


def zscore_trailing(spread: Array, window: int, min_periods: int | None = None) -> Array:
    """Trailing z-score: ``z_t = (s_t - mean_t) / std_t`` over ``[t-w+1, t]``.

    ``NaN`` while fewer than ``min_periods`` (default ``window``)
    observations are available. Strictly trailing: ``z_t`` never reads
    ``s_u`` for ``u > t``. Uses population std (ddof=0).
    """
    v = np.asarray(spread, dtype=float).reshape(-1)
    w = int(window)
    if w < 5 or v.size < 5 or not np.all(np.isfinite(v)):
        raise ValueError("need a finite spread and window >= 5")
    mp = w if min_periods is None else int(min_periods)
    if mp < 5 or mp > w:
        raise ValueError("min_periods must be in [5, window]")
    out = np.full(v.size, np.nan)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cs2 = np.concatenate([[0.0], np.cumsum(v * v)])
    for t in range(mp - 1, v.size):
        lo = max(0, t - w + 1)
        n = t - lo + 1
        m = (cs[t + 1] - cs[lo]) / n
        var = (cs2[t + 1] - cs2[lo]) / n - m * m
        sd = math.sqrt(max(var, 0.0))
        out[t] = 0.0 if sd <= 0.0 else (v[t] - m) / sd
    return out


def bands_position(z: Array, entry: float = 2.0, exit: float = 0.5) -> Array:
    """Z-score band state machine; position in ``{-1, 0, +1}``.

    ``+1`` = long the spread (entered on ``z < -entry``), ``-1`` = short the
    spread (entered on ``z > +entry``). Exit to flat when ``|z| <= exit``;
    opposite-side re-entry happens on a later bar, not the same one. NaN
    z-scores force flat. Causal state machine — ``out[t]`` depends only on
    ``z[<=t]``.
    """
    v = np.asarray(z, dtype=float).reshape(-1)
    e_in, e_out = float(entry), float(exit)
    if not np.isfinite(e_in) or e_in <= 0 or not np.isfinite(e_out) or e_out < 0:
        raise ValueError("need entry > 0 and exit >= 0")
    if e_out >= e_in:
        raise ValueError("exit must be < entry")
    out = np.zeros(v.size)
    pos = 0
    for t in range(v.size):
        zt = v[t]
        if not np.isfinite(zt):
            pos = 0
        elif pos == 0:
            if zt > e_in:
                pos = -1
            elif zt < -e_in:
                pos = 1
        elif pos == -1:
            if zt <= e_out:
                pos = 0
        elif zt >= -e_out:
            pos = 0
        out[t] = float(pos)
    return out
