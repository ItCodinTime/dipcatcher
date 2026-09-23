"""Drawdown-based risk measures beyond max drawdown.

``metrics.returns`` already has ``drawdown_series`` / ``max_drawdown`` /
``calmar_ratio``; ``metrics.analytics`` has duration stats.  This module adds
the distribution-level drawdown risk measures and downside-power ratios.

References:
- Checkalov, Uryasev, Zabarankin (2004). Drawdown measure in portfolio
  optimization (CDaR). *International Journal of Theoretical and Applied
  Finance* 7.
- Keating, Shadwick (2002). A universal performance measure (Omega).
  *Journal of Performance Measurement* 6.
- Kaplan, Knowles (2004). Kappa: a generalized downside risk-adjusted
  performance measure. *Journal of Performance Measurement* 8.
- Martin, McCann (1989). *The Investor's Guide to Fidelity Funds* — ulcer
  index.
- Zephyr Associates — pain index, tail ratio definitions.
- Magdon-Ismail, Atiya et al. (2004). On the maximum drawdown of a
  Brownian motion — expected max DD approximation.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _as_returns(r: Array, min_obs: int = 20) -> Array:
    v = np.asarray(r, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < min_obs:
        raise ValueError(f"returns must have >= {min_obs} finite observations")
    return v


def _require_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not (0.5 < a < 1.0):
        raise ValueError("alpha must be in (0.5, 1)")
    return a


def _dd_series(returns: Array) -> Array:
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(wealth)
    return wealth / peak - 1.0  # <= 0


def drawdown_at_risk(returns: Array, alpha: float = 0.95) -> float:
    """DaR: the alpha-quantile of the drawdown magnitude distribution.

    Positive number: ``quantile(|DD_t|, alpha)``.
    """
    v = _as_returns(returns)
    a = _require_alpha(alpha)
    dd = np.abs(_dd_series(v))
    return float(np.quantile(dd, a))


def conditional_drawdown_at_risk(returns: Array, alpha: float = 0.95) -> dict[str, float]:
    """CDaR (Checkalov–Uryasev 2004): mean of drawdowns beyond DaR."""
    v = _as_returns(returns)
    a = _require_alpha(alpha)
    dd = np.abs(_dd_series(v))
    dar = float(np.quantile(dd, a))
    tail = dd[dd >= dar]
    cdar = float(tail.mean()) if tail.size else dar
    return {"dar": dar, "cdar": cdar, "alpha": a}


def omega_ratio(returns: Array, threshold: float = 0.0) -> float:
    """Keating–Shadwick (2002) Omega ratio.

    ``Omega(tau) = E[max(r - tau, 0)] / E[max(tau - r, 0)]`` — ratio of
    gains above the threshold to losses below it.
    """
    v = _as_returns(returns)
    t = float(threshold)
    if not np.isfinite(t):
        raise ValueError("threshold must be finite")
    gains = np.maximum(v - t, 0.0).sum()
    losses = np.maximum(t - v, 0.0).sum()
    if losses <= 0.0:
        raise ValueError("no losses below threshold — Omega undefined")
    return float(gains / losses)


def kappa_ratio(returns: Array, order: int = 3, threshold: float = 0.0) -> float:
    """Kaplan–Knowles (2004) Kappa_l = (E[r]-tau) / LPM_l^{1/l}."""
    v = _as_returns(returns)
    t = float(threshold)
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    if not np.isfinite(t):
        raise ValueError("threshold must be finite")
    lpm = float(np.mean(np.maximum(t - v, 0.0) ** order))
    if lpm <= 0.0:
        raise ValueError("no downside below threshold — Kappa undefined")
    return float((v.mean() - t) / (lpm ** (1.0 / order)))


def ulcer_index(returns: Array) -> float:
    """Martin–McCann ulcer index: ``sqrt(mean(DD_pct^2))`` (DD in percent)."""
    v = _as_returns(returns)
    dd = _dd_series(v) * 100.0
    return float(np.sqrt(np.mean(dd**2)))


def pain_index(returns: Array) -> float:
    """Zephyr pain index: mean absolute drawdown."""
    v = _as_returns(returns)
    return float(np.mean(np.abs(_dd_series(v))))


def burke_ratio(returns: Array, periods_per_year: float = 252.0) -> float:
    """Burke ratio: annualized return / sqrt(sum of squared drawdowns).

    Uses the maximum drawdown of each underwater episode (not the full DD
    path) per the standard definition.
    """
    v = _as_returns(returns)
    if not np.isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")
    dd = _dd_series(v)
    # Episode maxima: local minima of the DD path between recoveries.
    mdds: list[float] = []
    cur = 0.0
    for d in dd:
        cur = min(cur, d)
        if d == 0.0 and cur < 0.0:
            mdds.append(cur)
            cur = 0.0
    if cur < 0.0:
        mdds.append(cur)
    denom = math.sqrt(sum(m * m for m in mdds))
    if denom <= 0.0:
        raise ValueError("no drawdown episodes — Burke undefined")
    ann = float((1.0 + v.mean()) ** periods_per_year - 1.0)
    return ann / denom


def tail_ratio(returns: Array, alpha: float = 0.95) -> float:
    """Zephyr tail ratio: ``|q_alpha| / |q_{1-alpha}|`` of the return dist."""
    v = _as_returns(returns)
    a = _require_alpha(alpha)
    hi = float(np.quantile(v, a))
    lo = float(np.quantile(v, 1.0 - a))
    if lo >= 0.0:
        raise ValueError("lower tail quantile non-negative — tail ratio undefined")
    return abs(hi) / abs(lo)


def expected_max_drawdown(
    mu: float, sigma: float, horizon: int, n_paths: int = 4000, seed: int = 0
) -> dict[str, float]:
    """Expected maximum drawdown via Monte Carlo (Magdon-Ismail et al. 2004).

    Simulates GBM paths with per-period mean ``mu`` and std ``sigma``;
    returns mean and quantiles of the max-drawdown distribution.
    """
    m = float(mu)
    s = float(sigma)
    if not np.isfinite(m) or not np.isfinite(s) or s <= 0.0:
        raise ValueError("mu finite, sigma positive required")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 2:
        raise ValueError("horizon must be an integer >= 2")
    if n_paths < 100:
        raise ValueError("n_paths must be >= 100")
    rng = np.random.default_rng(seed)
    rets = rng.normal(m, s, size=(n_paths, horizon))
    wealth = np.cumprod(1.0 + np.clip(rets, -0.9999, None), axis=1)
    peak = np.maximum.accumulate(wealth, axis=1)
    dd = 1.0 - wealth / peak
    mdd = dd.max(axis=1)
    return {
        "mean": float(mdd.mean()),
        "q50": float(np.quantile(mdd, 0.5)),
        "q95": float(np.quantile(mdd, 0.95)),
    }
