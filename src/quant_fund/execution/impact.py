"""Market-impact models and implementation-shortfall decomposition.

``costs.sqrt_impact`` already covers the instantaneous square-root cost
formula; this module adds the *dynamic* propagator view, the Perold (1988)
shortfall attribution, POV schedules, and benchmark slippage measures.

References:
- Perold (1988). The implementation shortfall: paper versus reality.
  *Journal of Portfolio Management* 14.
- Bouchaud et al. (2009). Markets as a dynamical ecology — the propagator
  model with transient impact ``G(tau) ~ tau^{-beta}``.
- Almgren et al. (2005). Direct estimation of equity market impact.
  *Risk* 18 — permanent vs temporary split ``I_perm ~ sigma (Q/V)^psi``.
- Gatheral (2010). No-dynamic-arbitrage and market impact.
- Kissell, Glantz (2003). *Optimal Trading Strategies* — IS components.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _finite_scalar(x: float, name: str, *, positive: bool = False) -> float:
    v = float(x)
    if not np.isfinite(v) or (positive and v <= 0.0):
        kind = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {kind}")
    return v


def _as_vector(x: Array, name: str, min_len: int = 1) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < min_len or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite vector of length >= {min_len}")
    return v


def sqrt_impact_bps(fraction_of_volume: float, sigma: float, upsilon: float = 0.6) -> float:
    """Almgren-style temporary impact in bps: ``upsilon * sigma * sqrt(q)``.

    ``fraction_of_volume`` = order size / market volume over the horizon;
    ``sigma`` = daily volatility in the same units (decimal or bps —
    consistent units in, same units out).
    """
    q = _finite_scalar(fraction_of_volume, "fraction_of_volume")
    s = _finite_scalar(sigma, "sigma", positive=True)
    u = _finite_scalar(upsilon, "upsilon", positive=True)
    if q < 0.0:
        raise ValueError("fraction_of_volume must be non-negative")
    return u * s * math.sqrt(q)


def permanent_impact(
    quantity: float, daily_volume: float, sigma: float, gamma: float = 0.5
) -> float:
    """Almgren et al. (2005) permanent impact fraction: ``gamma * sigma * Q/V``."""
    q = _finite_scalar(quantity, "quantity")
    v = _finite_scalar(daily_volume, "daily_volume", positive=True)
    s = _finite_scalar(sigma, "sigma", positive=True)
    g = _finite_scalar(gamma, "gamma", positive=True)
    if q < 0.0:
        raise ValueError("quantity must be non-negative")
    return g * s * (q / v)


def propagator_kernel(tau: Array, beta: float = 0.5) -> Array:
    """Bouchaud propagator decay ``G(tau) = (1 + tau)^{-beta}``.

    ``tau`` in bars; ``beta`` ≈ 0.4–0.6 empirical.  G(0) = 1.
    """
    t = np.asarray(tau, dtype=float)
    if not np.all(np.isfinite(t)) or np.any(t < 0.0):
        raise ValueError("tau must be a finite non-negative array")
    b = _finite_scalar(beta, "beta", positive=True)
    return (1.0 + t) ** (-b)


def propagator_price_path(
    signed_trades: Array, beta: float = 0.5, impact_scale: float = 1.0
) -> Array:
    """Price path under the transient propagator model.

    ``dp_t = impact_scale * sum_{s<=t} G(t-s) * q_s`` where ``q_s`` is the
    signed trade at bar s.  Returns cumulative price displacement per bar.
    """
    q = _as_vector(signed_trades, "signed_trades")
    b = _finite_scalar(beta, "beta", positive=True)
    a = _finite_scalar(impact_scale, "impact_scale")
    n = q.size
    taus = np.arange(n, dtype=np.float64)
    g = propagator_kernel(taus, b)
    # Convolution: dp_t = a * sum_s g[t-s] q[s].
    return a * np.array([np.dot(g[t - np.arange(t + 1)], q[: t + 1]) for t in range(n)])


def pow_law_total_impact(
    quantity: float, daily_volume: float, sigma: float, exponent: float = 0.5
) -> float:
    """General power-law impact: ``sigma * (Q/V)^exponent`` (fraction of price)."""
    q = _finite_scalar(quantity, "quantity")
    v = _finite_scalar(daily_volume, "daily_volume", positive=True)
    s = _finite_scalar(sigma, "sigma", positive=True)
    e = _finite_scalar(exponent, "exponent", positive=True)
    if q < 0.0:
        raise ValueError("quantity must be non-negative")
    return float(s * (q / v) ** e)


def pov_schedule(quantity: float, volume_forecast: Array, participation: float) -> Array:
    """Percent-of-volume schedule: trades ``p * V_t`` per bar until filled.

    Returns per-bar quantities summing to <= ``quantity``; the final bar is
    truncated so the schedule never exceeds the order.
    """
    q_total = _finite_scalar(quantity, "quantity", positive=True)
    v = _as_vector(volume_forecast, "volume_forecast")
    if np.any(v <= 0.0):
        raise ValueError("volume_forecast must be positive")
    p = _finite_scalar(participation, "participation")
    if not (0.0 < p <= 1.0):
        raise ValueError("participation must be in (0, 1]")
    raw = p * v
    cum = np.cumsum(raw)
    over = cum > q_total
    if np.any(over):
        first = int(np.argmax(over))
        raw[first] = max(0.0, q_total - (cum[first] - raw[first]))
        raw[first + 1 :] = 0.0
    return raw


def vwap_slippage(trade_prices: Array, trade_qty: Array, market_vwap: float) -> float:
    """Signed VWAP slippage: ``(exec_vwap - market_vwap) / market_vwap``.

    Positive for buys that paid up.  ``trade_qty`` carries the sign for
    buy(+)/sell(-); slippage is reported on the *buy-equivalent* side.
    """
    p = _as_vector(trade_prices, "trade_prices")
    q = _as_vector(trade_qty, "trade_qty")
    m = _finite_scalar(market_vwap, "market_vwap", positive=True)
    if p.size != q.size:
        raise ValueError("trade_prices and trade_qty must match length")
    if np.any(p <= 0.0):
        raise ValueError("trade_prices must be positive")
    q_abs = np.abs(q)
    if q_abs.sum() <= 0.0:
        raise ValueError("trade_qty must be non-zero")
    exec_vwap = float(np.sum(p * q_abs) / q_abs.sum())
    side = 1.0 if float(q.sum()) >= 0.0 else -1.0
    return side * (exec_vwap - m) / m


def arrival_price_slippage(trade_prices: Array, trade_qty: Array, arrival_price: float) -> float:
    """Signed slippage vs the arrival (decision) price, in return units."""
    p = _as_vector(trade_prices, "trade_prices")
    q = _as_vector(trade_qty, "trade_qty")
    a = _finite_scalar(arrival_price, "arrival_price", positive=True)
    if p.size != q.size:
        raise ValueError("trade_prices and trade_qty must match length")
    if np.any(p <= 0.0):
        raise ValueError("trade_prices must be positive")
    q_abs = np.abs(q)
    if q_abs.sum() <= 0.0:
        raise ValueError("trade_qty must be non-zero")
    exec_vwap = float(np.sum(p * q_abs) / q_abs.sum())
    side = 1.0 if float(q.sum()) >= 0.0 else -1.0
    return side * (exec_vwap - a) / a


def perold_shortfall(
    decision_price: float,
    exec_prices: Array,
    exec_qty: Array,
    unfilled_qty: float,
    cancel_price: float,
    fees: float = 0.0,
) -> dict[str, float]:
    """Perold (1988) implementation-shortfall decomposition (fractions).

    Splits IS into:
    - ``execution``: filled part — ``(exec_vwap - P_d) / P_d``
    - ``opportunity``: unfilled part — ``(P_cancel - P_d) / P_d`` weighted by
      the unfilled fraction of the order
    - ``fees``: explicit costs / (decision notional)
    - ``total``: their sum (paper return minus realized return, buys).

    All quantities share the same units; buys only (qty > 0 means buy).
    """
    pd_ = _finite_scalar(decision_price, "decision_price", positive=True)
    pc = _finite_scalar(cancel_price, "cancel_price", positive=True)
    f = _finite_scalar(fees, "fees")
    u = _finite_scalar(unfilled_qty, "unfilled_qty")
    if u < 0.0:
        raise ValueError("unfilled_qty must be non-negative")
    p = _as_vector(exec_prices, "exec_prices")
    q = _as_vector(exec_qty, "exec_qty")
    if p.size != q.size:
        raise ValueError("exec_prices and exec_qty must match length")
    if np.any(p <= 0.0) or np.any(q < 0.0):
        raise ValueError("exec_prices must be positive and exec_qty >= 0")
    filled = float(q.sum())
    total_qty = filled + u
    if total_qty <= 0.0:
        raise ValueError("order must have positive total quantity")
    exec_vwap = float(np.sum(p * q) / filled) if filled > 0.0 else pd_
    exec_part = ((exec_vwap - pd_) / pd_) * (filled / total_qty)
    opp_part = ((pc - pd_) / pd_) * (u / total_qty)
    fee_part = f / (pd_ * total_qty)
    return {
        "execution": exec_part,
        "opportunity": opp_part,
        "fees": fee_part,
        "total": exec_part + opp_part + fee_part,
        "fill_rate": filled / total_qty,
    }


def required_participation(
    quantity: float, volume_forecast: Array, max_bars: int | None = None
) -> float:
    """Minimum POV rate to finish ``quantity`` within ``max_bars``.

    Returns the participation fraction ``p`` such that
    ``p * sum(V_t) = Q`` over the allowed bars.  Fail-closed
    (``ValueError``) when the order exceeds total forecast volume.
    """
    q_total = _finite_scalar(quantity, "quantity", positive=True)
    v = _as_vector(volume_forecast, "volume_forecast")
    if np.any(v <= 0.0):
        raise ValueError("volume_forecast must be positive")
    if max_bars is not None and (int(max_bars) < 1 or int(max_bars) > v.size):
        raise ValueError("max_bars must be in [1, len(volume_forecast)]")
    v_use = v if max_bars is None else v[: int(max_bars)]
    total_v = float(v_use.sum())
    if q_total > total_v:
        raise ValueError(f"quantity {q_total} exceeds forecast volume {total_v} over horizon")
    return float(q_total / total_v)
