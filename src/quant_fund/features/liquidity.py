"""Microstructure liquidity estimators.

Low- and high-frequency liquidity measures beyond the repo's existing
Corwin–Schultz/Abdi–Ranaldo spread estimators, Amihud ILLIQ, Kyle lambda,
and VPIN.

References:
- Amivest (unpublished)/Brennan-Subrahmanyam (1996) liquidity ratio.
- Lesmond, Ogden, Trzcinka (1999) LOT zeros spread estimator.
- Lesmond (2005) FHT: zeros-frequency imputed transaction cost.
- Pastor, Stambaugh (2003) gamma: return reversal on signed volume.
- Hasbrouck (1991/2009) lambda: price impact per signed volume (Bayesian
    version of the PS regression).
- Glosten, Harris (1988) trade-indicator spread components.
- Holden (2009) effective tick / price clustering estimator.
- Roll (1984) is already in ``microstructure.py`` (kept out here).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

Array = NDArray[np.float64]


def _v(x: Array, n: int = 10) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"series must be finite with length >= {n}")
    return v


def amivest_ratio(returns: Array, volumes: Array) -> float:
    """Amivest liquidity ratio: ``sum |volume| / sum |return|``.

    High values = liquid (large volume moves price little). The inverse is
    the Amihud ratio.
    """
    r = _v(returns)
    vol = _v(volumes, 1)
    if vol.size != r.size or np.any(vol < 0):
        raise ValueError("volumes must be non-negative and match returns")
    denom = float(np.abs(r).sum())
    if denom <= 0.0:
        raise ValueError("zero total absolute return")
    return float(np.abs(vol).sum() / denom)


def zero_freq(returns: Array, eps: float = 1e-8) -> float:
    """Proportion of (near-)zero returns — LOT/FHT input statistic."""
    r = _v(returns, 20)
    return float(np.mean(np.abs(r) <= eps))


def lot_spread(returns: Array, eps: float = 1e-8) -> dict[str, float]:
    """Lesmond–Ogden–Trzcinka (1999) limited-dependent-variable spread.

    Estimates the latent cost thresholds (a1<0, a2>0) of the LOT model
    ``r* = r + a1 * 1[r*<0] + a2 * 1[r*>0]`` via the method-of-moments
    approximation: zero-return frequency maps to the cost region width
    under normality. Returns spread estimate and zero share.
    """
    r = _v(returns, 30)
    z = zero_freq(r, eps)
    sig = float(np.std(r[r != 0])) if np.any(r != 0) else 0.0
    if sig <= 0.0 or not np.isfinite(sig):
        raise ValueError("no nonzero returns")
    # Approximate: central mass between -z1 and +z2 of a normal ~ zero freq.
    half_width = sstats.norm.ppf(0.5 + z / 2.0) * sig
    return {
        "spread": 2.0 * half_width,
        "zero_share": z,
        "sigma_nonzero": sig,
    }


def fht_cost(returns: Array, eps: float = 1e-8) -> dict[str, float]:
    """Lesmond (2005) FHT: imputed per-trade cost from zero-return frequency.

    ``FHT = 2 * sigma * N^{-1}( (1+Z)/2 )`` where Z is the zero-return
    share and sigma the nonzero-return std — a one-liner transaction-cost
    proxy widely used for low-frequency data.
    """
    r = _v(returns, 30)
    z = zero_freq(r, eps)
    nz = r[np.abs(r) > eps]
    if nz.size < 10:
        raise ValueError("too few nonzero returns")
    sig = float(np.std(nz))
    if sig <= 0.0:
        raise ValueError("zero nonzero-return variance")
    z = float(np.clip(z, 0.0, 0.98))
    fht = 2.0 * sig * sstats.norm.ppf((1.0 + z) / 2.0)
    return {"fht": fht, "zero_share": z, "sigma": sig}


def pastor_stambaugh_gamma(returns: Array, volumes: Array) -> dict[str, float]:
    """Pastor–Stambaugh (2003) liquidity gamma.

    Regresses ``r_{t+1}`` on ``[r_t, sign(r_t) * vol_t]``; the signed-volume
    coefficient (gamma) measures order-flow reversal — more negative/stronger
    magnitude = more illiquid. Returns gamma and its t-stat (OLS).
    """
    r = _v(returns, 30)
    vol = _v(volumes, 30)
    if vol.size != r.size or np.any(vol < 0):
        raise ValueError("volumes must be non-negative and match returns")
    y = r[1:]
    r_lag = r[:-1]
    signed_vol = np.sign(r_lag) * vol[:-1]
    x = np.column_stack([np.ones(y.size), r_lag, signed_vol])
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    e = y - x @ b
    n, k = x.shape
    s2 = float(e @ e) / (n - k)
    cov = np.linalg.pinv(x.T @ x) * s2
    se = math.sqrt(max(cov[2, 2], 0.0))
    gamma = float(b[2])
    return {"gamma": gamma, "t": gamma / se if se > 0 else np.nan}


def hasbrouck_lambda(price_changes: Array, signed_volumes: Array) -> dict[str, float]:
    """Hasbrouck (1991) lambda: price change on signed square-root volume.

    ``dp_t = c + lambda * sign(q_t)*sqrt(|q_t|) + e_t`` — the standard
    Hasbrouck impact regression. Returns lambda and t-stat.
    """
    dp = _v(price_changes, 30)
    q = _v(signed_volumes, 30)
    if q.size != dp.size:
        raise ValueError("signed_volumes must match price_changes")
    reg = np.sign(q) * np.sqrt(np.abs(q))
    x = np.column_stack([np.ones(dp.size), reg])
    b, *_ = np.linalg.lstsq(x, dp, rcond=None)
    e = dp - x @ b
    n = dp.size
    s2 = float(e @ e) / (n - 2)
    cov = np.linalg.pinv(x.T @ x) * s2
    se = math.sqrt(max(cov[1, 1], 0.0))
    lam = float(b[1])
    return {"lambda": lam, "t": lam / se if se > 0 else np.nan}


def glosten_harris(
    price_changes: Array, trade_signs: Array, volumes: Array | None = None
) -> dict[str, float]:
    """Glosten–Harris (1988) spread decomposition.

    ``dp_t = c0 * q_t + c1 * q_t * V_t + e_t`` where q_t in {-1,+1} is the
    trade direction; c0 is the transitory (spread) component and c1 the
    adverse-selection component. ``volumes`` optional — without them only
    the transitory component is identified.
    """
    dp = _v(price_changes, 30)
    q = _v(trade_signs, 30)
    if q.size != dp.size or not np.all(np.abs(np.abs(q) - 1.0) < 1e-6):
        raise ValueError("trade_signs must be +/-1 matching price_changes")
    cols = [np.ones(dp.size), q]
    if volumes is not None:
        vol = _v(volumes, 30)
        if vol.size != dp.size:
            raise ValueError("volumes must match price_changes")
        cols.append(q * vol)
    x = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(x, dp, rcond=None)
    e = dp - x @ b
    n, k = x.shape
    s2 = float(e @ e) / (n - k)
    cov = np.linalg.pinv(x.T @ x) * s2
    out = {
        "transitory": float(b[1]),
        "t_transitory": float(b[1] / math.sqrt(max(cov[1, 1], 1e-16))),
    }
    if volumes is not None:
        out["adverse"] = float(b[2])
        out["t_adverse"] = float(b[2] / math.sqrt(max(cov[2, 2], 1e-16)))
        out["spread_est"] = 2.0 * abs(float(b[1]))
    return out


def effective_tick(prices: Array) -> dict[str, Array]:
    """Holden (2009) effective tick: probability-weighted price-clustering.

    Estimates the implicit tick size from the frequency with which prices
    end in round increments. Returns the estimated tick and clustering
    probabilities for candidate grids {1, 0.5, 0.25, 0.1, 0.05, 0.01}.
    """
    p = _v(prices, 30)
    if np.any(p <= 0):
        raise ValueError("prices must be positive")
    cands = np.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.01])
    probs = np.zeros(cands.size)
    for i, g in enumerate(cands):
        frac = np.abs(p / g - np.round(p / g))
        probs[i] = float(np.mean(frac < 0.01))
    # Holden: effective tick = sum_i g_i * (prob_i - prob_{i-1}).
    prev = 0.0
    tick = 0.0
    for i, g in enumerate(cands):
        w = max(probs[i] - prev, 0.0)
        tick += g * w
        prev = probs[i]
    return {"tick": np.array([tick]), "probs": probs}


def turnover_volatility(volumes: Array) -> dict[str, float]:
    """Volume-based liquidity risk: turnover level + volatility of turnover."""
    vol = _v(volumes, 20)
    if np.any(vol < 0):
        raise ValueError("volumes must be non-negative")
    mu = float(vol.mean())
    if mu <= 0:
        raise ValueError("zero mean volume")
    return {
        "turnover_mean": mu,
        "turnover_cv": float(vol.std() / mu),
        "turnover_vol": float(np.std(np.log(np.maximum(vol, 1e-12)))),
    }
