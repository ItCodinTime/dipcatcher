"""NumPy reference for the research backtest core.

Pipeline: signal, target weights, rebalance clock, no-trade band, turnover
costs, net returns, NAV.

This is the delay-1 close-to-close identity used by
``hedge_lab.directional`` (TSMOM, top-k 12–1, Antonacci dual momentum,
inverse-vol) and the rank-weight constructor ``research.net_replay._weights``
(momentum, reversal, equal weight). It is not ``backtest.engine`` and it is
not the tournament share/cash ledger: no next-open fills, participation
caps, borrow, financing, or kill switch.

``w_t`` is a function of prices through ``t - delay`` only. It earns the
simple return ``R_t = P_t / P_{t-1} - 1``. With ``delay >= 1`` that return
is not an input to the weight.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from quant_fund.diffbacktest.spec import (
    StrategyParams,
    as_int,
    flag,
    linear_cost_rate,
    validate_params,
)
from quant_fund.metrics.returns import max_drawdown, sharpe_ratio

Array = NDArray[np.float64]
_EPS = 1e-12


@dataclass(frozen=True)
class Simulation:
    """One hard (NumPy) backtest. Research diagnostic, not a live NAV."""

    strategy: str
    weights: Array
    asset_returns: Array
    turnover: Array
    costs: Array
    net: Array
    nav: Array
    initial_nav: float
    sigma: Array

    @property
    def terminal_pnl(self) -> float:
        return float(self.nav[-1] - self.initial_nav)


def simple_returns(prices: Array) -> Array:
    """Close-to-close simple returns. Row 0 is 0. Matches directional.py."""
    p = np.asarray(prices, dtype=np.float64)
    out = np.zeros_like(p, dtype=np.float64)
    if p.ndim == 1:
        out[1:] = p[1:] / np.maximum(p[:-1], _EPS) - 1.0
    else:
        out[1:] = p[1:] / np.maximum(p[:-1], _EPS) - 1.0
    out[~np.isfinite(out)] = 0.0
    return out


def validate_prices(prices: np.ndarray, *, min_rows: int = 8) -> Array:
    p = np.asarray(prices, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < min_rows or p.shape[1] < 1:
        raise ValueError(f"prices must have shape (T, N) with T >= {min_rows} and N >= 1")
    if not np.all(np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("prices must be finite and positive")
    return p


def apply_rebalance_band(weights: Array, band: float) -> Array:
    """No-trade band as a soft-threshold proximal map, hard version.

    ``held <- held + sign(target - held) * max(|target - held| - band, 0)``.
    ``band <= 0`` returns the target unchanged (bit-preserving).
    """
    w = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(band) or band < 0.0:
        raise ValueError("rebalance_band must be finite and non-negative")
    if band == 0.0:
        return np.array(w, dtype=np.float64, copy=True)
    held = np.zeros(w.shape[1], dtype=np.float64)
    out = np.zeros_like(w)
    for t in range(w.shape[0]):
        delta = w[t] - held
        trade = np.sign(delta) * np.maximum(np.abs(delta) - band, 0.0)
        held = held + trade
        out[t] = held
    return out


def trailing_sigma(returns: Array, vol_lookback: int, delay: int) -> Array:
    """Causal per-name std (ddof=1) of returns through ``t - delay``."""
    lb = max(int(vol_lookback), 2)
    t_len, n_names = returns.shape
    out = np.zeros((t_len, n_names), dtype=np.float64)
    for t in range(t_len):
        end = t - delay + 1
        if end < lb:
            continue
        window = returns[end - lb : end]
        sig = np.std(window, axis=0, ddof=1)
        ok = np.isfinite(sig) & (sig > _EPS)
        out[t, ok] = sig[ok]
    return out


def apply_costs(
    weights: Array,
    asset_returns: Array,
    params: StrategyParams,
    sigma: Array,
) -> tuple[Array, Array, Array]:
    """Return ``(net, l1_turnover, cost)``.

    Linear term: ``(one_way_cost + (commission_bps + half_spread_bps) / 1e4) * ||Δw||_1``.
    Impact term, off unless ``impact_y > 0``: ``impact_y * Σ_i σ_i |Δw_i|^{3/2}``,
    the weight-space planning cost (NAV fraction, not dollar ``sqrt_impact``).
    Turnover on bar 0 is 0, matching the directional books.
    """
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(asset_returns, dtype=np.float64)
    dw = np.zeros_like(w)
    dw[1:] = np.abs(w[1:] - w[:-1])
    l1 = np.sum(dw, axis=1)
    linear = linear_cost_rate(params) * l1
    if float(params.impact_y) == 0.0:
        impact = np.zeros(w.shape[0], dtype=np.float64)
    else:
        sig = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, 0.0)
        impact = float(params.impact_y) * np.sum(sig * np.power(dw, 1.5), axis=1)
    gross = np.sum(w * r, axis=1)
    net = gross - linear - impact
    return net, l1, linear + impact


def nav_path(net: Array, initial_nav: float = 1.0) -> Array:
    if not np.isfinite(initial_nav) or initial_nav <= 0.0:
        raise ValueError("initial_nav must be finite and positive")
    return float(initial_nav) * np.cumprod(1.0 + np.asarray(net, dtype=np.float64))


def terminal_pnl(net: Array, initial_nav: float = 1.0) -> float:
    """Compounded P&L on ``initial_nav`` (wealth at T minus starting wealth)."""
    path = nav_path(net, initial_nav)
    if path.size == 0 or not np.isfinite(path[-1]):
        return float("nan")
    return float(path[-1] - initial_nav)


def sum_pnl(net: Array) -> float:
    """Sum of net simple returns. Equals the sum of a directional book series."""
    r = np.asarray(net, dtype=np.float64).reshape(-1)
    if r.size == 0 or not np.all(np.isfinite(r)):
        return float("nan")
    return float(np.sum(r))


def sharpe(net: Array, periods_per_year: float = 252.0) -> float:
    """Annualized sample Sharpe. Delegates to ``metrics.returns.sharpe_ratio``."""
    return float(
        sharpe_ratio(np.asarray(net, dtype=float), periods_per_year=periods_per_year)["sharpe"]
    )


def drawdown(net: Array) -> float:
    """Max drawdown (≤ 0). Delegates to ``metrics.returns.max_drawdown``."""
    return float(max_drawdown(np.asarray(net, dtype=float)))


def objective_value(
    net: Array,
    name: str,
    *,
    initial_nav: float = 1.0,
    periods_per_year: float = 252.0,
    score_start: int = 0,
) -> float:
    scored = np.asarray(net, dtype=np.float64)[score_start:]
    if name == "pnl":
        return terminal_pnl(scored, initial_nav)
    if name == "pnl_sum":
        return sum_pnl(scored)
    if name == "sharpe":
        return sharpe(scored, periods_per_year)
    if name == "drawdown":
        return drawdown(scored)
    raise ValueError(f"unknown objective {name!r}")


def path_sigma(prices: np.ndarray, *, floor: float = 1e-4) -> Array:
    """Full-sample return std per name, floored. Frozen scale for adversarial shocks."""
    r = simple_returns(validate_prices(prices, min_rows=4))[1:]
    sig = np.std(r, axis=0, ddof=1)
    sig = np.where(np.isfinite(sig), sig, floor)
    return np.maximum(sig, floor).astype(np.float64)


def perturb_prices(
    prices: np.ndarray,
    eps: np.ndarray,
    sigma: np.ndarray,
    *,
    return_clip: tuple[float, float] = (-0.8, 2.0),
) -> Array:
    """Rebuild a price path after a volatility-scaled return shock.

    ``r'_t = clip(r_t + ε_t ⊙ σ, lo, hi)`` with ``r'_0 = 0`` and ``P_0`` fixed.
    ``σ`` is in return units per name, so ``ε`` is in "original volatility" units.
    Prices stay positive because ``1 + r' >= 0.2`` under the default clip.
    """
    px = np.asarray(prices, dtype=np.float64)
    shock = np.asarray(eps, dtype=np.float64) * np.asarray(sigma, dtype=np.float64).reshape(1, -1)
    if shock.shape != px.shape:
        raise ValueError("eps must have the same shape as prices")
    r = simple_returns(px)
    lo, hi = return_clip
    r2 = np.clip(r + shock, lo, hi)
    r2 = np.array(r2, copy=True)
    r2[0] = 0.0
    growth = np.cumprod(1.0 + r2, axis=0)
    return cast(Array, np.asarray(px[0] * growth, dtype=np.float64))


def synthetic_prices(
    n_steps: int,
    n_names: int,
    seed: int,
    *,
    mu: float = 0.0002,
    sigma: float = 0.01,
    s0: float = 100.0,
) -> Array:
    """I.i.d. Gaussian simple returns, constant mean and vol. SYNTHETIC.

    Not a market simulator. The clip at ±40% only keeps prices positive.
    """
    if n_steps < 8 or n_names < 1:
        raise ValueError("n_steps >= 8 and n_names >= 1 required")
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0 or s0 <= 0:
        raise ValueError("mu finite, sigma > 0, s0 > 0 required")
    rng = np.random.default_rng(seed)
    rets = mu + sigma * rng.normal(0.0, 1.0, size=(n_steps, n_names))
    rets = np.clip(rets, -0.4, 0.4)
    px = np.empty((n_steps, n_names), dtype=np.float64)
    px[0] = s0
    for t in range(1, n_steps):
        px[t] = px[t - 1] * (1.0 + rets[t])
    return px


def strong_trend_prices(n_steps: int = 80, seed: int = 0) -> Array:
    """Three names with stable formation-window signs. SYNTHETIC parity fixture.

    Drifts are large relative to the noise so a sharpened tanh agrees with
    ``sign``. This is not a market path and not a performance exhibit.
    """
    if n_steps < 16:
        raise ValueError("n_steps >= 16 required")
    drifts = np.array([0.01, -0.01, 0.008], dtype=np.float64)
    noise = np.random.default_rng(seed).normal(0.0, 0.002, size=(n_steps, 3))
    rets = np.clip(drifts + noise, -0.4, 0.4)
    px = np.empty((n_steps, 3), dtype=np.float64)
    px[0] = 100.0
    for t in range(1, n_steps):
        px[t] = px[t - 1] * (1.0 + rets[t])
    return px


def _tsmom_row(
    returns_through_end: Array,
    *,
    lookback: int,
    skip: int,
    vol_lookback: int,
    target_vol: float,
    periods_per_year: float,
    long_only: bool,
    max_gross: float,
) -> Array:
    """One-date Moskowitz weight. Same formula as ``quant_models.tsmom_weights``."""
    r = returns_through_end
    n_names = r.shape[1]
    if r.shape[0] < lookback + 1 or lookback <= skip:
        return np.zeros(n_names, dtype=np.float64)
    window = r[-lookback:-skip] if skip > 0 else r[-lookback:]
    signal = np.sign(np.nansum(window, axis=0))
    vol_win = r[-vol_lookback:]
    sigma = np.nanstd(vol_win, axis=0, ddof=1)
    sigma = np.where(sigma > _EPS, sigma, np.nan)
    size = target_vol / (sigma * np.sqrt(periods_per_year))
    w = np.where(np.isfinite(signal * size), signal * size, 0.0)
    if long_only:
        w = np.maximum(w, 0.0)
    gross = float(np.nansum(np.abs(w)))
    if gross > float(max_gross) > 0.0:
        w = w * (float(max_gross) / gross)
    return np.where(np.isfinite(w), w, 0.0)


def _scheduled(raw_at_rebalance: Array, start: int, step: int) -> Array:
    """Hold the last rebalance target. Zeros before ``start``."""
    t_len, n_names = raw_at_rebalance.shape
    w = np.zeros((t_len, n_names), dtype=np.float64)
    prev: Array | None = None
    for t in range(start, t_len):
        if prev is not None and (t - start) % step != 0:
            w[t] = prev
            continue
        prev = np.array(raw_at_rebalance[t], dtype=np.float64, copy=True)
        w[t] = prev
    return w


def _tsmom_weights(prices: Array, returns: Array, params: StrategyParams) -> Array:
    del prices
    lookback = as_int(params.lookback, name="lookback", lo=2, hi=10_000)
    skip = as_int(params.skip, name="skip", lo=0, hi=10_000)
    vol_lookback = as_int(params.vol_lookback, name="vol_lookback", lo=2, hi=10_000)
    delay = params.delay
    step = params.rebalance_every
    start = lookback + delay
    long_only = flag(params.long_only)
    t_len, n_names = returns.shape
    raw = np.zeros((t_len, n_names), dtype=np.float64)
    for t in range(start, t_len):
        end = t - delay + 1
        if end < lookback + 1:
            continue
        raw[t] = _tsmom_row(
            returns[:end],
            lookback=lookback,
            skip=skip,
            vol_lookback=vol_lookback,
            target_vol=float(params.target_vol),
            periods_per_year=float(params.periods_per_year),
            long_only=long_only,
            max_gross=float(params.max_gross),
        )
    return _scheduled(raw, start, step)


def _topk_weights(prices: Array, returns: Array, params: StrategyParams) -> Array:
    del returns
    lookback = as_int(params.lookback, name="lookback", lo=2, hi=10_000)
    skip = as_int(params.skip, name="skip", lo=0, hi=10_000)
    delay = params.delay
    step = params.rebalance_every
    start = lookback + delay
    k = max(1, as_int(params.top_k, name="top_k", lo=1, hi=10_000))
    require_positive = flag(params.require_positive)
    t_len, n_names = prices.shape
    k = min(k, n_names)
    w = np.zeros((t_len, n_names), dtype=np.float64)
    held: list[int] | None = None
    for t in range(start, t_len):
        end = t - delay
        if end < lookback or (held is not None and (t - start) % step != 0):
            picked = held or []
        else:
            past = end - lookback
            skip_i = end - skip
            if past < 0 or skip_i <= past:
                picked = []
            else:
                mom = prices[skip_i] / np.maximum(prices[past], _EPS) - 1.0
                mom = np.where(np.isfinite(mom), mom, -np.inf)
                if require_positive:
                    mom = np.where(mom > 0.0, mom, -np.inf)
                order = np.argsort(mom)[::-1]
                picked = [int(i) for i in order[:k] if np.isfinite(mom[int(i)])]
            held = picked
        alive = [i for i in picked if np.isfinite(prices[end, i]) and float(prices[end, i]) > _EPS]
        if alive:
            w[t, alive] = 1.0 / float(len(alive))
    return w


def _antonacci_weights(prices: Array, returns: Array, params: StrategyParams) -> Array:
    del returns
    if prices.shape[1] != 2:
        raise ValueError("antonacci requires N=2 columns (risky, defensive)")
    lookback = as_int(params.lookback, name="lookback", lo=2, hi=10_000)
    skip = as_int(params.skip, name="skip", lo=0, hi=10_000)
    delay = params.delay
    step = params.rebalance_every
    start = lookback + delay
    risky = prices[:, 0]
    defensive = prices[:, 1]
    t_len = prices.shape[0]
    w = np.zeros((t_len, 2), dtype=np.float64)
    last = (0.0, 0.0)
    for t in range(start, t_len):
        if t > start and (t - start) % step != 0:
            w[t, 0], w[t, 1] = last
            continue
        end = t - delay
        past = end - lookback
        skip_i = end - skip
        if past < 0 or skip_i <= past:
            continue
        if risky[past] <= _EPS or defensive[past] <= _EPS:
            last = (0.0, 0.0)
            w[t, 0], w[t, 1] = last
            continue
        m_risky = risky[skip_i] / risky[past] - 1.0
        m_def = defensive[skip_i] / defensive[past] - 1.0
        if (not np.isfinite(m_risky)) or (not np.isfinite(m_def)):
            last = (0.0, 0.0)
        elif m_risky > m_def and m_risky > 0.0:
            last = (1.0, 0.0)
        elif m_def > 0.0:
            last = (0.0, 1.0)
        else:
            last = (0.0, 0.0)
        w[t, 0], w[t, 1] = last
    return w


def _risk_parity_weights(prices: Array, returns: Array, params: StrategyParams) -> Array:
    del prices
    # Match risk_parity_blend: lookback is clamped to at least 8.
    lb = max(as_int(params.vol_lookback, name="vol_lookback", lo=2, hi=10_000), 8)
    delay = params.delay
    t_len, n_names = returns.shape
    w = np.zeros((t_len, n_names), dtype=np.float64)
    for t in range(lb + delay, t_len):
        end = t - delay + 1
        window = returns[end - lb : end]
        sig = np.std(window, axis=0, ddof=1)
        inv = np.zeros(n_names, dtype=np.float64)
        ok = np.isfinite(sig) & (sig > _EPS)
        inv[ok] = 1.0 / sig[ok]
        total = float(inv.sum())
        if total > _EPS:
            w[t] = inv / total
    # The published blend sums to 1. max_gross < 1 scales that book down.
    # max_gross >= 1 leaves it unchanged (no spurious rescale from 1+eps).
    if 0.0 < float(params.max_gross) < 1.0:
        w = w * float(params.max_gross)
    return w


def _rank_weights(prices: Array, params: StrategyParams, *, reverse: bool) -> Array:
    """Cross-sectional rank book. Same sizing as ``net_replay._weights``."""
    lookback = as_int(params.lookback, name="lookback", lo=2, hi=10_000)
    delay = params.delay
    step = params.rebalance_every
    start = lookback + delay
    long_short = flag(params.long_short)
    t_len, n_names = prices.shape
    if n_names < 2:
        raise ValueError("momentum and reversal require N >= 2")
    budget = float(params.gross_limit) * (1.0 - float(params.target_buffer))
    name_limit = float(params.max_name_weight) * (1.0 - float(params.target_buffer))
    count = max(1, int(n_names * float(params.fraction)))
    count = min(count, n_names)
    leg = budget / (2.0 if long_short else 1.0)
    size = min(leg / count, name_limit)
    w = np.zeros((t_len, n_names), dtype=np.float64)
    prev: Array | None = None
    for t in range(start, t_len):
        if prev is not None and (t - start) % step != 0:
            w[t] = prev
            continue
        end = t - delay
        past = end - lookback
        if past < 0:
            continue
        signal = prices[end] / np.maximum(prices[past], _EPS) - 1.0
        if reverse:
            signal = -signal
        signal = np.where(np.isfinite(signal), signal, -np.inf)
        ranked = np.argsort(signal, kind="stable")
        row = np.zeros(n_names, dtype=np.float64)
        row[ranked[-count:]] = size
        if long_short:
            row[ranked[:count]] = -size
        prev = row
        w[t] = row
    return w


def _equal_weights(prices: Array, params: StrategyParams) -> Array:
    n_names = prices.shape[1]
    budget = float(params.gross_limit) * (1.0 - float(params.target_buffer))
    name_limit = float(params.max_name_weight) * (1.0 - float(params.target_buffer))
    size = min(budget / n_names, name_limit)
    w = np.zeros_like(prices, dtype=np.float64)
    w[params.delay :] = size
    return w


_BUILDERS: dict[str, Callable[[Array, Array, StrategyParams], Array]] = {
    "tsmom": _tsmom_weights,
    "topk": _topk_weights,
    "antonacci": _antonacci_weights,
    "risk_parity": _risk_parity_weights,
    "momentum": lambda prices, returns, params: _rank_weights(prices, params, reverse=False),
    "reversal": lambda prices, returns, params: _rank_weights(prices, params, reverse=True),
    "equal_weight": lambda prices, returns, params: _equal_weights(prices, params),
}


def warmup_start(strategy: str, params: StrategyParams) -> int:
    """First bar that can hold a non-zero weight. Used to score in-sample fits."""
    validate_params(strategy, params)
    if strategy == "equal_weight":
        return params.delay
    if strategy == "risk_parity":
        lb = max(as_int(params.vol_lookback, name="vol_lookback", lo=2, hi=10_000), 8)
        return lb + params.delay
    lookback = as_int(params.lookback, name="lookback", lo=2, hi=10_000)
    return lookback + params.delay


def simulate(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams | None = None,
    *,
    initial_nav: float = 1.0,
) -> Simulation:
    """Hard research backtest. ``params`` default is the published-style short window."""
    book = params or StrategyParams()
    validate_params(strategy, book)
    px = validate_prices(prices)
    if strategy == "antonacci" and px.shape[1] != 2:
        raise ValueError("antonacci requires N=2 columns (risky, defensive)")
    needed = warmup_start(strategy, book) + 2
    if px.shape[0] < needed:
        raise ValueError(f"prices have {px.shape[0]} rows; {strategy} needs at least {needed}")
    returns = simple_returns(px)
    weights = _BUILDERS[strategy](px, returns, book)
    weights = apply_rebalance_band(weights, float(book.rebalance_band))
    vol_lb = as_int(book.vol_lookback, name="vol_lookback", lo=2, hi=10_000)
    sigma = trailing_sigma(returns, vol_lb, book.delay)
    net, turnover, costs = apply_costs(weights, returns, book, sigma)
    if not np.all(np.isfinite(net)):
        raise ValueError("non-finite net returns; check costs and prices")
    nav = nav_path(net, initial_nav)
    return Simulation(
        strategy=strategy,
        weights=weights,
        asset_returns=returns,
        turnover=turnover,
        costs=costs,
        net=net,
        nav=nav,
        initial_nav=float(initial_nav),
        sigma=sigma,
    )
