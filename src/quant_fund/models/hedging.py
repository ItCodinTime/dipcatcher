"""Minimum-variance hedging and hedge-effectiveness measurement.

References:
- Ederington (1979): minimum-variance hedge ratio and effectiveness R2.
- Johnson (1960): variance-minimizing hedge.
- Kroner & Sultan (1993): time-varying (bivariate GARCH) hedge ratios —
  represented here by the rolling OLS variant.
- Myers & Thompson (1989): generalized hedge-ratio critique (conditional
  vs unconditional ratios) — we report both OLS and correlation forms.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _paired(x: Array, y: Array, n: int = 30) -> tuple[Array, Array]:
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    if a.size != b.size or a.size < n:
        raise ValueError(f"series must share length >= {n}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("series must be finite")
    return a, b


def mv_hedge_ratio(spot_ret: Array, hedge_ret: Array) -> dict[str, float]:
    """Ederington (1979) minimum-variance hedge ratio.

    OLS slope of spot changes on hedge changes: h* = Cov(s, h)/Var(h).
    Also reports the correlation-based ratio rho * (sigma_s / sigma_h)."""
    s, h = _paired(spot_ret, hedge_ret)
    vh = float(h.var())
    if vh <= 0:
        raise ValueError("hedge leg has zero variance")
    h_star = float(np.cov(s, h)[0, 1] / vh)
    rho = float(np.corrcoef(s, h)[0, 1])
    h_corr = rho * (s.std() / h.std()) if h.std() > 0 else np.nan
    return {
        "h_star": h_star,
        "h_corr": float(h_corr),
        "rho": rho,
        "n": float(s.size),
    }


def hedge_effectiveness(spot_ret: Array, hedge_ret: Array, h: float) -> dict[str, float]:
    """Ederington (1979) hedged-portfolio variance reduction.

    ``E = 1 - Var(s - h*d) / Var(s)`` — the hedging R-squared."""
    s, d = _paired(spot_ret, hedge_ret)
    vs = float(s.var())
    if vs <= 0:
        raise ValueError("spot leg has zero variance")
    hedged = s - h * d
    vh = float(hedged.var())
    e = 1.0 - vh / vs
    return {
        "effectiveness": float(e),
        "var_unhedged": vs,
        "var_hedged": vh,
        "h": float(h),
    }


def rolling_hedge_ratio(spot_ret: Array, hedge_ret: Array, window: int = 60) -> dict[str, Array | float]:
    """Rolling OLS hedge ratio — simple time-varying MV hedge."""
    s, h = _paired(spot_ret, hedge_ret)
    n = s.size
    if not (10 <= window <= n):
        raise ValueError("window must be in [10, n]")
    out = np.full(n, np.nan)
    for t in range(window, n):
        hs = h[t - window : t]
        ss = s[t - window : t]
        v = float(hs.var())
        if v > 0:
            out[t] = float(np.cov(ss, hs)[0, 1] / v)
    return {"ratio": out, "window": float(window)}


def basis_statistics(spot: Array, hedge: Array) -> dict[str, float]:
    """Basis (spot minus hedge) level statistics: mean, std, half-life of
    mean reversion via AR(1), and correlation of levels."""
    s, h = _paired(spot, hedge, n=40)
    basis = s - h
    mu = float(basis.mean())
    sd = float(basis.std())
    if sd <= 0:
        raise ValueError("degenerate basis")
    x = basis[:-1] - basis[:-1].mean()
    yv = basis[1:] - basis[1:].mean()
    den = float(x @ x)
    rho = float(x @ yv / den) if den > 0 else np.nan
    if np.isfinite(rho):
        if 0 < rho < 1:
            half = math.log(0.5) / math.log(rho)
        elif rho >= 1:
            half = math.inf
        else:
            half = 0.0
    else:
        half = np.nan
    return {
        "mean": mu,
        "std": sd,
        "ar1": float(rho),
        "half_life": float(half),
        "level_corr": float(np.corrcoef(s, h)[0, 1]),
    }
