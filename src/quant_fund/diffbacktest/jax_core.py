"""JAX research backtest: hard parity, smooth relaxations, straight-through.

Hard mode matches ``numpy_core`` (and therefore the existing directional books
when the band is off and impact is off). Smooth mode replaces

* absolute value and the no-trade band with softplus / soft-threshold,
* sign, top-k, and the dual-momentum switch with smooth surrogates,

so P&L, Sharpe, and drawdown have exact reverse-mode gradients in every
continuous parameter and in the price path. STE mode uses the hard trade in
the forward pass and the smooth surrogate on the backward pass.

Requires the optional ``jax`` extra. The NumPy core does not import this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import numpy as np

from quant_fund.diffbacktest.numpy_core import validate_prices
from quant_fund.diffbacktest.spec import (
    GRAD_BETA,
    LIVE_PNL_CLAIM,
    RESEARCH_ONLY,
    StrategyParams,
    active_parameters,
    as_int,
    flag,
    harden,
    pack,
    validate_params,
)

_EPS = 1e-12
# Pairwise rank temperature, in 1/return. A 10bp score gap is order-1 at
# beta = 1, so PARITY_BETA resolves discrete top-k / momentum ties instead of
# freezing a fractional rank. Inclusion still sharpens with ``beta`` alone.
_RANK_SCALE = 1.0e3


def available() -> bool:
    """True when the optional JAX extra imports."""
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def _libs() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX backtest needs the optional 'jax' extra (CPU is enough): "
            "uv sync --extra jax. The NumPy core does not need JAX."
        ) from exc
    cast(Callable[[str, object], None], jax.config.update)("jax_enable_x64", True)
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX float64 could not be enabled; parity requires x64")
    return jax, jnp


def sign_ste(x: Any, beta: float) -> Any:
    """Straight-through sign: forward ``sign``, backward ``tanh(beta x)``."""
    jax, jnp = _libs()
    hard = jnp.sign(x)
    soft = jnp.tanh(beta * x)
    return soft + jax.lax.stop_gradient(hard - soft)


def soft_abs(x: Any, beta: float) -> Any:
    """``(softplus(beta x) + softplus(-beta x)) / beta`` → |x| as beta → ∞."""
    _, jnp = _libs()
    return (jnp.logaddexp(0.0, beta * x) + jnp.logaddexp(0.0, -beta * x)) / beta


def soft_max(a: Any, b: Any, beta: float) -> Any:
    _, jnp = _libs()
    m = jnp.maximum(a, b)
    return m + jnp.logaddexp(beta * (a - m), beta * (b - m)) / beta


def soft_min(a: Any, b: Any, beta: float) -> Any:
    return -soft_max(-a, -b, beta)


def soft_threshold(x: Any, band: float, beta: float) -> Any:
    """Smooth proximal map of the no-trade band.

    Hard map: ``sign(x) * max(|x| - band, 0)``. At ``band = 0`` the smooth
    map is the identity for every finite ``beta`` (the bias of ``soft_abs``
    cancels), so a zero band does not dead-zone small risk-parity trades.
    Positive bands sharpen to the hard threshold as ``beta`` grows.
    """
    _, jnp = _libs()
    absx = soft_abs(x, beta)
    gap = absx - band
    excess = 0.5 * (gap + soft_abs(gap, beta))
    # excess at band 0 is 0.5 * (|x|~ + soft_abs(|x|~)), not |x|~. Rescale so
    # that point is exactly x, and a positive band only shrinks the trade.
    excess0 = 0.5 * (absx + soft_abs(absx, beta))
    scale = excess / jnp.maximum(excess0, _EPS)
    return x * scale


@dataclass(frozen=True)
class JaxSimulation:
    strategy: str
    mode: str
    beta: float
    weights: np.ndarray
    asset_returns: np.ndarray
    turnover: np.ndarray
    costs: np.ndarray
    net: np.ndarray
    nav: np.ndarray
    initial_nav: float

    @property
    def terminal_pnl(self) -> float:
        return float(self.nav[-1] - self.initial_nav)


@dataclass(frozen=True)
class GradientResult:
    """Exact reverse-mode gradient of one objective. Research diagnostic."""

    objective: str
    value: float
    parameter_names: tuple[str, ...]
    parameter_gradient: np.ndarray
    price_gradient: np.ndarray
    mode: str
    beta: float
    research_only: bool = RESEARCH_ONLY
    live_pnl_claim: bool = LIVE_PNL_CLAIM


def _static_ints(params: StrategyParams) -> tuple[int, int, int, int, int, int, int]:
    return (
        as_int(params.lookback, name="lookback", lo=2, hi=10_000),
        as_int(params.skip, name="skip", lo=0, hi=10_000),
        as_int(params.vol_lookback, name="vol_lookback", lo=2, hi=10_000),
        as_int(params.top_k, name="top_k", lo=1, hi=10_000),
        1 if flag(params.long_only) else 0,
        1 if flag(params.require_positive) else 0,
        1 if flag(params.long_short) else 0,
    )


def _take(theta: Any, names: tuple[str, ...], key: str) -> Any:
    return theta[names.index(key)]


def _returns(prices: Any) -> Any:
    _, jnp = _libs()
    out = jnp.zeros_like(prices)
    out = out.at[1:].set(prices[1:] / jnp.maximum(prices[:-1], _EPS) - 1.0)
    return jnp.where(jnp.isfinite(out), out, 0.0)


def _hold(raw: Any, start: int, step: int) -> Any:
    """Rebalance clock. Matches ``numpy_core._scheduled``."""
    jax, jnp = _libs()
    n_names = raw.shape[1]

    def body(held: Any, t: Any) -> tuple[Any, Any]:
        update = (t >= start) & (jnp.mod(t - start, step) == 0)
        held = jnp.where(update, raw[t], held)
        out = jnp.where(t >= start, held, jnp.zeros_like(held))
        return held, out

    _, weights = jax.lax.scan(body, jnp.zeros(n_names), jnp.arange(raw.shape[0]))
    return weights


def _band(desired: Any, band: Any, beta: float, relax: bool) -> Any:
    jax, jnp = _libs()
    if not relax:
        return desired

    def body(held: Any, target: Any) -> tuple[Any, Any]:
        trade = soft_threshold(target - held, band, beta)
        held = held + trade
        return held, held

    _, weights = jax.lax.scan(body, jnp.zeros(desired.shape[1]), desired)
    return weights


def _gross_cap(weights: Any, max_gross: Any, beta: float, relax: bool) -> Any:
    _, jnp = _libs()
    if relax:
        gross = jnp.sum(soft_abs(weights, beta), axis=-1, keepdims=True)
        ratio = max_gross / jnp.maximum(gross, _EPS)
        scale = soft_min(jnp.ones_like(ratio), ratio, beta)
        return weights * scale
    gross = jnp.sum(jnp.abs(weights), axis=-1, keepdims=True)
    scale = jnp.where(gross > max_gross, max_gross / jnp.maximum(gross, _EPS), 1.0)
    return weights * scale


def _long_only_mix(weights: Any, mix: Any, beta: float, relax: bool) -> Any:
    _, jnp = _libs()
    if relax:
        clipped = jnp.logaddexp(0.0, beta * weights) / beta
        return mix * clipped + (1.0 - mix) * weights
    return jnp.where(mix >= 0.5, jnp.maximum(weights, 0.0), weights)


def _masked_sum(
    series: Any,
    length: Any,
    skip: Any,
    delay: int,
    beta: float,
) -> tuple[Any, Any]:
    """Soft rectangular window. ``length`` and ``skip`` may be traced."""
    _, jnp = _libs()
    t_len = series.shape[0]
    dtype = series.dtype
    t = jnp.arange(t_len, dtype=dtype)[:, None]
    j = jnp.arange(t_len, dtype=dtype)[None, :]
    end = t - delay + 1.0
    lag = end - j
    causal = ((j >= 0.0) & (j < end)).astype(dtype)
    gate = jax_sigmoid(beta * (length - lag + 0.5)) * jax_sigmoid(beta * (lag - skip - 0.5))
    mask = gate * causal
    return mask @ series, jnp.sum(mask, axis=1)


def jax_sigmoid(x: Any) -> Any:
    jax, _ = _libs()
    return jax.nn.sigmoid(x)


def _tsmom_raw(
    returns: Any,
    theta: Any,
    names: tuple[str, ...],
    *,
    mode: str,
    beta: float,
    delay: int,
    periods: float,
    lookback_i: int,
    skip_i: int,
    vol_i: int,
    long_only_i: int,
) -> Any:
    jax, jnp = _libs()
    relax = mode != "hard"
    t_len, n_names = returns.shape
    target_vol = _take(theta, names, "target_vol")
    max_gross = _take(theta, names, "max_gross")
    long_mix = _take(theta, names, "long_only") if relax else jnp.asarray(float(long_only_i))
    if relax:
        lookback = _take(theta, names, "lookback")
        skip = _take(theta, names, "skip")
        vol_lb = _take(theta, names, "vol_lookback")
        summed, _n = _masked_sum(returns, lookback, skip, delay, beta)
        s1, n_vol = _masked_sum(returns, vol_lb, jnp.asarray(0.0), delay, beta)
        s2, _ = _masked_sum(returns**2, vol_lb, jnp.asarray(0.0), delay, beta)
        n_vol = n_vol[:, None]
        n_safe = jnp.maximum(n_vol, 1.0)
        var = (s2 - s1**2 / n_safe) / jnp.maximum(n_vol - 1.0, 1.0)
        var = jnp.where(n_vol > 1.5, jnp.maximum(var, 0.0), jnp.inf)
        sigma = jnp.sqrt(var)
        signal = jnp.tanh(beta * summed)
        size = target_vol / (jnp.maximum(sigma, _EPS) * jnp.sqrt(periods))
        size = jnp.where(jnp.isfinite(size) & (sigma > _EPS), size, 0.0)
        raw = signal * size
        raw = _long_only_mix(raw, long_mix, beta, True)
        raw = _gross_cap(raw, max_gross, beta, True)
        return raw

    def row(t: Any) -> Any:
        end = t - delay + 1
        valid = (t >= lookback_i + delay) & (end >= lookback_i + 1)
        win_start = jnp.clip(end - lookback_i, 0, t_len - lookback_i)
        full = jax.lax.dynamic_slice(returns, (win_start, 0), (lookback_i, n_names))
        used = full[: lookback_i - skip_i] if skip_i > 0 else full
        signal = jnp.sign(jnp.sum(used, axis=0))
        vol_start = jnp.clip(end - vol_i, 0, t_len - vol_i)
        vol = jax.lax.dynamic_slice(returns, (vol_start, 0), (vol_i, n_names))
        sigma = jnp.std(vol, axis=0, ddof=1)
        sigma = jnp.where(sigma > _EPS, sigma, jnp.nan)
        size = target_vol / (sigma * jnp.sqrt(periods))
        w = jnp.where(jnp.isfinite(signal * size), signal * size, 0.0)
        w = jnp.where(long_only_i == 1, jnp.maximum(w, 0.0), w)
        gross = jnp.sum(jnp.abs(w))
        w = jnp.where(gross > max_gross, w * (max_gross / jnp.maximum(gross, _EPS)), w)
        return jnp.where(valid, w, jnp.zeros_like(w))

    return jax.vmap(row)(jnp.arange(t_len))


def _schedule_from_lookback(
    raw: Any,
    lookback: Any,
    delay: int,
    step: int,
    beta: float,
    relax: bool,
    lookback_i: int,
) -> Any:
    jax, jnp = _libs()
    if not relax:
        return _hold(raw, lookback_i + delay, step)
    anchor = jax.lax.stop_gradient(jnp.round(lookback))
    t_idx = jnp.arange(raw.shape[0])
    active = jax.nn.sigmoid(beta * (t_idx.astype(raw.dtype) + 0.5 - lookback - delay))
    is_reb = jnp.mod(t_idx - anchor - delay, step) == 0

    def body(held: Any, xs: tuple[Any, Any, Any]) -> tuple[Any, Any]:
        target, gate, reb = xs
        updated = jnp.where(reb, target, held)
        held = gate * updated + (1.0 - gate) * held
        return held, held

    _, weights = jax.lax.scan(body, jnp.zeros(raw.shape[1]), (raw, active, is_reb))
    return weights


def _topk_weights(
    prices: Any,
    theta: Any,
    names: tuple[str, ...],
    *,
    mode: str,
    beta: float,
    delay: int,
    every: int,
    lookback_i: int,
    skip_i: int,
    topk_i: int,
    require_i: int,
) -> Any:
    jax, jnp = _libs()
    relax = mode != "hard"
    t_len, n_names = prices.shape
    k_static = min(topk_i, n_names)
    if relax:
        lookback = _take(theta, names, "lookback")
        skip = _take(theta, names, "skip")
        k = jnp.clip(_take(theta, names, "top_k"), 1.0, float(n_names))
        require = _take(theta, names, "require_positive")

    def hard_fresh(t: Any) -> Any:
        end = t - delay
        past = end - lookback_i
        skip_at = end - skip_i
        valid = (t >= lookback_i + delay) & (past >= 0) & (skip_at > past)
        past_c = jnp.clip(past, 0, t_len - 1)
        skip_c = jnp.clip(skip_at, 0, t_len - 1)
        mom = prices[skip_c] / jnp.maximum(prices[past_c], _EPS) - 1.0
        mom = jnp.where(jnp.isfinite(mom), mom, -jnp.inf)
        mom = jnp.where((require_i == 1) & (mom <= 0.0), -jnp.inf, mom)
        order = jnp.argsort(mom)
        top = order[-k_static:]
        finite = jnp.isfinite(mom[top])
        n_ok = jnp.sum(finite)
        values = jnp.where(finite, 1.0 / jnp.maximum(n_ok, 1.0), 0.0)
        fresh = jnp.zeros(n_names).at[top].set(values)
        fresh = jnp.where(n_ok > 0, fresh, jnp.zeros(n_names))
        return jnp.where(valid, fresh, jnp.zeros(n_names)), valid

    if not relax:

        def body(held: Any, t: Any) -> tuple[Any, Any]:
            fresh, valid = hard_fresh(t)
            is_reb = (t >= lookback_i + delay) & (jnp.mod(t - (lookback_i + delay), every) == 0)
            chosen = jnp.where(is_reb & valid, fresh, held)
            out = jnp.where(t >= lookback_i + delay, chosen, jnp.zeros_like(held))
            held = jnp.where(t >= lookback_i + delay, chosen, held)
            return held, out

        _, weights = jax.lax.scan(body, jnp.zeros(n_names), jnp.arange(t_len))
        return weights

    def soft_row(t: Any) -> Any:
        end = t - delay
        past_f = end - lookback
        skip_f = end - skip
        mom = _blend_ratio(prices, skip_f, past_f)
        mom = mom - 10.0 * require * jax.nn.sigmoid(-beta * mom)
        raw = _soft_equal_top(mom, k, beta)
        return raw

    raw = jax.vmap(soft_row)(jnp.arange(t_len))
    return _schedule_from_lookback(raw, lookback, delay, every, beta, True, lookback_i)


def _blend_ratio(prices: Any, num_index: Any, den_index: Any) -> Any:
    """Price ratio at fractional indices. Linear in log price, causal clip."""
    _, jnp = _libs()
    t_len = prices.shape[0]
    log_p = jnp.log(jnp.maximum(prices, _EPS))

    def at(index: Any) -> Any:
        index = jnp.clip(index, 0.0, t_len - 1.001)
        i0 = jnp.floor(index).astype(jnp.int32)
        i1 = jnp.minimum(i0 + 1, t_len - 1)
        w = index - i0.astype(index.dtype)
        return (1.0 - w) * log_p[i0] + w * log_p[i1]

    return jnp.exp(at(num_index) - at(den_index)) - 1.0


def _rank_from_best(scores: Any, beta: float) -> Any:
    """How many names outrank ``scores``. See ``_RANK_SCALE``."""
    _, jnp = _libs()
    n_names = scores.shape[0]
    diff = (scores[None, :] - scores[:, None]) * _RANK_SCALE
    better = jax_sigmoid(beta * diff) * (1.0 - jnp.eye(n_names, dtype=scores.dtype))
    return jnp.sum(better, axis=1)


def _soft_equal_top(scores: Any, k: Any, beta: float) -> Any:
    """Soft top-k equal weight. Sharpens to uniform weight on the hard top-k."""
    _, jnp = _libs()
    rank = _rank_from_best(scores, beta)
    p = jax_sigmoid(beta * (k - 0.5 - rank))
    mass = jnp.sum(p)
    return jnp.where(mass > _EPS, p / mass, jnp.zeros_like(p))


def _antonacci_weights(
    prices: Any,
    theta: Any,
    names: tuple[str, ...],
    *,
    mode: str,
    beta: float,
    delay: int,
    every: int,
    lookback_i: int,
    skip_i: int,
) -> Any:
    jax, jnp = _libs()
    relax = mode != "hard"
    t_len = prices.shape[0]
    lookback = _take(theta, names, "lookback") if relax else jnp.asarray(float(lookback_i))
    skip = _take(theta, names, "skip") if relax else jnp.asarray(float(skip_i))

    def hard_body(last: Any, t: Any) -> tuple[Any, Any]:
        end = t - delay
        hold = (t > lookback_i + delay) & (jnp.mod(t - (lookback_i + delay), every) != 0)
        past = end - lookback_i
        skip_at = end - skip_i
        computable = (t >= lookback_i + delay) & (~hold) & (past >= 0) & (skip_at > past)
        past_c = jnp.clip(past, 0, t_len - 1)
        skip_c = jnp.clip(skip_at, 0, t_len - 1)
        m_r = prices[skip_c, 0] / prices[past_c, 0] - 1.0
        m_d = prices[skip_c, 1] / prices[past_c, 1] - 1.0
        finite = jnp.isfinite(m_r) & jnp.isfinite(m_d)
        risky = (m_r > m_d) & (m_r > 0.0) & finite
        defensive = (~risky) & (m_d > 0.0) & finite
        fresh = jnp.stack([risky.astype(prices.dtype), defensive.astype(prices.dtype)])
        chosen = jnp.where(hold, last, jnp.where(computable, fresh, last))
        out = jnp.where(t >= lookback_i + delay, chosen, jnp.zeros(2))
        last = jnp.where(t >= lookback_i + delay, chosen, last)
        return last, out

    if not relax:
        _, weights = jax.lax.scan(hard_body, jnp.zeros(2), jnp.arange(t_len))
        return weights

    def soft_row(t: Any) -> Any:
        end = t - delay
        m_r = _blend_ratio(prices[:, 0:1], end - skip, end - lookback)[0]
        m_d = _blend_ratio(prices[:, 1:2], end - skip, end - lookback)[0]
        logits = beta * jnp.stack([m_r, m_d, jnp.zeros_like(m_r)])
        prob = jax.nn.softmax(logits)
        return prob[:2]

    raw = jax.vmap(soft_row)(jnp.arange(t_len))
    return _schedule_from_lookback(raw, lookback, delay, every, beta, True, lookback_i)


def _risk_weights(
    returns: Any,
    theta: Any,
    names: tuple[str, ...],
    *,
    mode: str,
    beta: float,
    delay: int,
    vol_i: int,
) -> Any:
    jax, jnp = _libs()
    relax = mode != "hard"
    t_len, n_names = returns.shape
    max_gross = _take(theta, names, "max_gross")
    if relax:
        vol_lb = jnp.maximum(_take(theta, names, "vol_lookback"), 8.0)
        s1, n_vol = _masked_sum(returns, vol_lb, jnp.asarray(0.0), delay, beta)
        s2, _ = _masked_sum(returns**2, vol_lb, jnp.asarray(0.0), delay, beta)
        n_vol = n_vol[:, None]
        n_safe = jnp.maximum(n_vol, 1.0)
        var = (s2 - s1**2 / n_safe) / jnp.maximum(n_vol - 1.0, 1.0)
        var = jnp.where(n_vol > 1.5, jnp.maximum(var, 0.0), jnp.inf)
        sigma = jnp.sqrt(var)
        inv = jnp.where(sigma > _EPS, 1.0 / sigma, 0.0)
        total = jnp.sum(inv, axis=1, keepdims=True)
        raw = jnp.where(total > _EPS, inv / total, 0.0)
        # n_vol can hit the window length one bar before the hard book, which
        # waits for t >= vol_lookback + delay. Gate on that clock.
        t_idx = jnp.arange(t_len, dtype=returns.dtype)[:, None]
        ready = jax.nn.sigmoid(beta * (t_idx + 0.5 - vol_lb - delay))
        raw = raw * ready
        scale = soft_min(jnp.asarray(1.0), max_gross, beta)
        return raw * scale

    lb = max(vol_i, 8)

    def row(t: Any) -> Any:
        end = t - delay + 1
        valid = t >= lb + delay
        start = jnp.clip(end - lb, 0, t_len - lb)
        window = jax.lax.dynamic_slice(returns, (start, 0), (lb, n_names))
        sig = jnp.std(window, axis=0, ddof=1)
        inv = jnp.where(jnp.isfinite(sig) & (sig > _EPS), 1.0 / sig, 0.0)
        total = jnp.sum(inv)
        w = jnp.where(total > _EPS, inv / total, jnp.zeros(n_names))
        return jnp.where(valid, w, jnp.zeros(n_names))

    raw = jax.vmap(row)(jnp.arange(t_len))
    # Published blend is gross 1. Scale down only when max_gross < 1.
    return jnp.where(max_gross < 1.0, raw * max_gross, raw)


def _rank_weights(
    prices: Any,
    theta: Any,
    names: tuple[str, ...],
    *,
    reverse: bool,
    mode: str,
    beta: float,
    delay: int,
    every: int,
    lookback_i: int,
    long_short_i: int,
) -> Any:
    jax, jnp = _libs()
    relax = mode != "hard"
    t_len, n_names = prices.shape
    lookback = _take(theta, names, "lookback") if relax else jnp.asarray(float(lookback_i))
    fraction = _take(theta, names, "fraction")
    gross_limit = _take(theta, names, "gross_limit")
    max_name = _take(theta, names, "max_name_weight")
    buffer = _take(theta, names, "target_buffer")
    long_short = _take(theta, names, "long_short") if relax else jnp.asarray(float(long_short_i))
    budget = gross_limit * (1.0 - buffer)
    name_limit = max_name * (1.0 - buffer)
    # Hard book truncates (``int(N * fraction)``). Smooth mode keeps the
    # continuous count so ``fraction`` has a gradient; they agree when
    # ``N * fraction`` is an integer.
    raw_count = n_names * fraction
    if relax:
        count = jnp.clip(raw_count, 1.0, float(n_names))
    else:
        count = jnp.clip(jnp.floor(raw_count), 1.0, float(n_names))

    def row(t: Any) -> Any:
        end = t - delay
        if relax:
            signal = _blend_ratio(prices, jnp.asarray(end, dtype=prices.dtype), end - lookback)
        else:
            past = jnp.clip(end - lookback_i, 0, t_len - 1)
            end_c = jnp.clip(end, 0, t_len - 1)
            signal = prices[end_c] / jnp.maximum(prices[past], _EPS) - 1.0
            valid = (t >= lookback_i + delay) & ((end - lookback_i) >= 0)
            signal = jnp.where(valid, signal, -jnp.inf)
        if reverse:
            signal = -signal
        if relax:
            leg_long = budget
            leg_ls = budget / 2.0
            size_long = soft_min(leg_long / count, name_limit, beta)
            size_ls = soft_min(leg_ls / count, name_limit, beta)
            p_long = _soft_membership(signal, count, beta, top=True)
            p_short = _soft_membership(signal, count, beta, top=False)
            w_long = p_long * size_long
            w_ls = p_long * size_ls + p_short * (-size_ls)
            return (1.0 - long_short) * w_long + long_short * w_ls
        leg = jnp.where(long_short_i == 1, budget / 2.0, budget)
        size = jnp.minimum(leg / count, name_limit)
        order = jnp.argsort(jnp.where(jnp.isfinite(signal), signal, -jnp.inf))
        rank_from_worst = jnp.zeros(n_names).at[order].set(jnp.arange(n_names, dtype=prices.dtype))
        long_mask = rank_from_worst >= (n_names - count)
        short_mask = rank_from_worst < count
        w = jnp.where(long_mask, size, 0.0)
        w = jnp.where((long_short_i == 1) & short_mask, -size, w)
        valid = (t >= lookback_i + delay) & ((t - delay - lookback_i) >= 0)
        return jnp.where(valid, w, jnp.zeros(n_names))

    raw = jax.vmap(row)(jnp.arange(t_len))
    if relax:
        return _schedule_from_lookback(raw, lookback, delay, every, beta, True, lookback_i)
    return _hold(raw, lookback_i + delay, every)


def _soft_membership(scores: Any, k: Any, beta: float, *, top: bool) -> Any:
    _, jnp = _libs()
    n_names = scores.shape[0]
    rank_from_best = _rank_from_best(scores, beta)
    rank = rank_from_best if top else (n_names - 1.0) - rank_from_best
    return jax_sigmoid(beta * (k - 0.5 - rank))


def _equal_weights(
    prices: Any,
    theta: Any,
    names: tuple[str, ...],
    *,
    mode: str,
    beta: float,
    delay: int,
) -> Any:
    _, jnp = _libs()
    relax = mode != "hard"
    n_names = prices.shape[1]
    budget = _take(theta, names, "gross_limit") * (1.0 - _take(theta, names, "target_buffer"))
    name_limit = _take(theta, names, "max_name_weight") * (
        1.0 - _take(theta, names, "target_buffer")
    )
    per = budget / n_names
    size = soft_min(per, name_limit, beta) if relax else jnp.minimum(per, name_limit)
    gate = (jnp.arange(prices.shape[0]) >= delay).astype(prices.dtype)
    # ``size`` is a scalar. Broadcast across names explicitly; ``(T, 1) * scalar``
    # would drop the name axis and understate turnover.
    return gate[:, None] * size * jnp.ones((1, n_names), dtype=prices.dtype)


def _impact_sigma(returns: Any, vol_i: int, delay: int) -> Any:
    """Causal std for the impact term. Hard window; multiplied by impact_y."""
    jax, jnp = _libs()
    t_len, n_names = returns.shape
    lb = max(int(vol_i), 2)

    def row(t: Any) -> Any:
        end = t - delay + 1
        valid = end >= lb
        start = jnp.clip(end - lb, 0, max(t_len - lb, 0))
        window = jax.lax.dynamic_slice(returns, (start, 0), (lb, n_names))
        sig = jnp.std(window, axis=0, ddof=1)
        sig = jnp.where(jnp.isfinite(sig) & (sig > 0.0) & valid, sig, 0.0)
        return sig

    return jax.vmap(row)(jnp.arange(t_len))


def _costs(
    weights: Any,
    returns: Any,
    theta: Any,
    names: tuple[str, ...],
    sigma: Any,
    beta: float,
    relax: bool,
) -> tuple[Any, Any, Any]:
    _, jnp = _libs()
    delta = weights[1:] - weights[:-1]
    mag = soft_abs(delta, beta) if relax else jnp.abs(delta)
    l1 = jnp.concatenate([jnp.zeros((1,), dtype=weights.dtype), jnp.sum(mag, axis=1)])
    rate = (
        _take(theta, names, "one_way_cost")
        + (_take(theta, names, "commission_bps") + _take(theta, names, "half_spread_bps")) / 1e4
    )
    linear = rate * l1
    impact_y = _take(theta, names, "impact_y")
    per_name = mag if relax else jnp.abs(delta)
    units_body = jnp.sum(sigma[1:] * jnp.power(jnp.maximum(per_name, 0.0), 1.5), axis=1)
    units = jnp.concatenate([jnp.zeros((1,), dtype=weights.dtype), units_body])
    impact = impact_y * units
    gross = jnp.sum(weights * returns, axis=1)
    net = gross - linear - impact
    return net, l1, linear + impact


def _build_weights(
    prices: Any,
    returns: Any,
    theta: Any,
    *,
    strategy: str,
    mode: str,
    beta: float,
    delay: int,
    every: int,
    periods: float,
    names: tuple[str, ...],
    lookback_i: int,
    skip_i: int,
    vol_i: int,
    topk_i: int,
    long_only_i: int,
    require_i: int,
    long_short_i: int,
) -> Any:
    if strategy == "tsmom":
        raw = _tsmom_raw(
            returns,
            theta,
            names,
            mode=mode,
            beta=beta,
            delay=delay,
            periods=periods,
            lookback_i=lookback_i,
            skip_i=skip_i,
            vol_i=vol_i,
            long_only_i=long_only_i,
        )
        lookback = _take(theta, names, "lookback")
        weights = _schedule_from_lookback(
            raw, lookback, delay, every, beta, mode != "hard", lookback_i
        )
    elif strategy == "topk":
        weights = _topk_weights(
            prices,
            theta,
            names,
            mode=mode,
            beta=beta,
            delay=delay,
            every=every,
            lookback_i=lookback_i,
            skip_i=skip_i,
            topk_i=topk_i,
            require_i=require_i,
        )
    elif strategy == "antonacci":
        weights = _antonacci_weights(
            prices,
            theta,
            names,
            mode=mode,
            beta=beta,
            delay=delay,
            every=every,
            lookback_i=lookback_i,
            skip_i=skip_i,
        )
    elif strategy == "risk_parity":
        weights = _risk_weights(
            returns, theta, names, mode=mode, beta=beta, delay=delay, vol_i=vol_i
        )
    elif strategy in {"momentum", "reversal"}:
        weights = _rank_weights(
            prices,
            theta,
            names,
            reverse=strategy == "reversal",
            mode=mode,
            beta=beta,
            delay=delay,
            every=every,
            lookback_i=lookback_i,
            long_short_i=long_short_i,
        )
    elif strategy == "equal_weight":
        weights = _equal_weights(prices, theta, names, mode=mode, beta=beta, delay=delay)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    band = _take(theta, names, "rebalance_band")
    if mode == "hard":
        # band <= 0 is the identity, so a zero band stays bit-close to NumPy.
        return _hard_band(weights, band)
    return _band(weights, band, beta, True)


def _hard_band(desired: Any, band: Any) -> Any:
    jax, jnp = _libs()

    def apply(_: Any) -> Any:
        def body(held: Any, target: Any) -> tuple[Any, Any]:
            delta = target - held
            trade = jnp.sign(delta) * jnp.maximum(jnp.abs(delta) - band, 0.0)
            nxt = held + trade
            return nxt, nxt

        _, weights = jax.lax.scan(body, jnp.zeros(desired.shape[1], dtype=desired.dtype), desired)
        return weights

    return jax.lax.cond(band <= 0.0, lambda _: desired, apply, operand=None)


def _forward(
    prices: Any,
    theta: Any,
    *,
    strategy: str,
    mode: str,
    beta: float,
    delay: int,
    every: int,
    periods: float,
    initial_nav: float,
    names: tuple[str, ...],
    lookback_i: int,
    skip_i: int,
    vol_i: int,
    topk_i: int,
    long_only_i: int,
    require_i: int,
    long_short_i: int,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    jax, jnp = _libs()
    returns = _returns(prices)
    if mode == "ste":
        hard = _build_weights(
            prices,
            returns,
            theta,
            strategy=strategy,
            mode="hard",
            beta=beta,
            delay=delay,
            every=every,
            periods=periods,
            names=names,
            lookback_i=lookback_i,
            skip_i=skip_i,
            vol_i=vol_i,
            topk_i=topk_i,
            long_only_i=long_only_i,
            require_i=require_i,
            long_short_i=long_short_i,
        )
        soft = _build_weights(
            prices,
            returns,
            theta,
            strategy=strategy,
            mode="smooth",
            beta=beta,
            delay=delay,
            every=every,
            periods=periods,
            names=names,
            lookback_i=lookback_i,
            skip_i=skip_i,
            vol_i=vol_i,
            topk_i=topk_i,
            long_only_i=long_only_i,
            require_i=require_i,
            long_short_i=long_short_i,
        )
        weights = soft + jax.lax.stop_gradient(hard - soft)
        relax_costs = True
    else:
        weights = _build_weights(
            prices,
            returns,
            theta,
            strategy=strategy,
            mode=mode,
            beta=beta,
            delay=delay,
            every=every,
            periods=periods,
            names=names,
            lookback_i=lookback_i,
            skip_i=skip_i,
            vol_i=vol_i,
            topk_i=topk_i,
            long_only_i=long_only_i,
            require_i=require_i,
            long_short_i=long_short_i,
        )
        relax_costs = mode != "hard"
    sigma = _impact_sigma(returns, vol_i, delay)
    net, turnover, costs = _costs(weights, returns, theta, names, sigma, beta, relax_costs)
    nav = jnp.asarray(initial_nav, dtype=prices.dtype) * jnp.cumprod(1.0 + net)
    return weights, returns, turnover, costs, net, nav


def _objective_from_net(
    net: Any,
    name: str,
    *,
    initial_nav: float,
    periods: float,
    score_start: int,
    beta: float,
    smooth_drawdown: bool,
) -> Any:
    _, jnp = _libs()
    scored = net[score_start:]
    if name == "pnl":
        wealth = jnp.cumprod(1.0 + scored)
        return jnp.asarray(initial_nav) * wealth[-1] - initial_nav
    if name == "pnl_sum":
        return jnp.sum(scored)
    if name == "sharpe":
        mu = jnp.mean(scored)
        n = scored.shape[0]
        var = jnp.sum((scored - mu) ** 2) / (n - 1.0)
        vol = jnp.sqrt(jnp.maximum(var, 1e-18))
        return mu / vol * jnp.sqrt(periods)
    if name == "drawdown":
        return _drawdown(scored, beta, smooth=smooth_drawdown)
    raise ValueError(f"unknown objective {name!r}")


def _drawdown(net: Any, beta: float, *, smooth: bool) -> Any:
    jax, jnp = _libs()
    wealth = jnp.cumprod(1.0 + net)

    def body(peak: Any, wt: Any) -> tuple[Any, Any]:
        peak = soft_max(peak, wt, beta) if smooth else jnp.maximum(peak, wt)
        return peak, peak

    _, peaks = jax.lax.scan(body, jnp.asarray(1.0, dtype=net.dtype), wealth)
    dd = wealth / peaks - 1.0
    if smooth:
        return -jax.scipy.special.logsumexp(-beta * dd) / beta
    return jnp.min(dd)


@lru_cache(maxsize=64)
def _compiled(
    strategy: str,
    mode: str,
    beta: float,
    delay: int,
    every: int,
    periods: float,
    initial_nav: float,
    names: tuple[str, ...],
    lookback_i: int,
    skip_i: int,
    vol_i: int,
    topk_i: int,
    long_only_i: int,
    require_i: int,
    long_short_i: int,
) -> Any:
    jax, _ = _libs()

    def run(prices: Any, theta: Any) -> tuple[Any, ...]:
        return _forward(
            prices,
            theta,
            strategy=strategy,
            mode=mode,
            beta=beta,
            delay=delay,
            every=every,
            periods=periods,
            initial_nav=initial_nav,
            names=names,
            lookback_i=lookback_i,
            skip_i=skip_i,
            vol_i=vol_i,
            topk_i=topk_i,
            long_only_i=long_only_i,
            require_i=require_i,
            long_short_i=long_short_i,
        )

    return jax.jit(run)


def _call(
    prices: np.ndarray,
    params: StrategyParams,
    strategy: str,
    *,
    mode: str,
    beta: float,
    initial_nav: float,
) -> tuple[Any, ...]:
    if mode not in {"hard", "smooth", "ste"}:
        raise ValueError("mode must be 'hard', 'smooth', or 'ste'")
    validate_params(strategy, params)
    px = validate_prices(np.asarray(prices, dtype=np.float64))
    if strategy == "antonacci" and px.shape[1] != 2:
        raise ValueError("antonacci requires N=2 columns (risky, defensive)")
    names = active_parameters(strategy)
    theta = pack(params, names)
    lookback_i, skip_i, vol_i, topk_i, long_only_i, require_i, long_short_i = _static_ints(params)
    fn = _compiled(
        strategy,
        mode,
        float(beta),
        int(params.delay),
        int(params.rebalance_every),
        float(params.periods_per_year),
        float(initial_nav),
        names,
        lookback_i,
        skip_i,
        vol_i,
        topk_i,
        long_only_i,
        require_i,
        long_short_i,
    )
    _, jnp = _libs()
    return cast(tuple[Any, ...], fn(jnp.asarray(px), jnp.asarray(theta)))


def simulate_jax(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams | None = None,
    *,
    mode: str = "hard",
    beta: float = GRAD_BETA,
    initial_nav: float = 1.0,
) -> JaxSimulation:
    """Differentiable backtest. ``mode='hard'`` is the NumPy-parity book."""
    book = params or StrategyParams()
    # Hard parity uses the snapped integer windows the NumPy core uses.
    if mode == "hard":
        book = harden(book)
    weights, returns, turnover, costs, net, nav = _call(
        prices, book, strategy, mode=mode, beta=beta, initial_nav=initial_nav
    )
    weights_np = np.asarray(weights, dtype=np.float64)
    if weights_np.shape != np.asarray(prices).shape:
        raise RuntimeError(f"weight shape {weights_np.shape} != prices {np.asarray(prices).shape}")
    return JaxSimulation(
        strategy=strategy,
        mode=mode,
        beta=float(beta),
        weights=weights_np,
        asset_returns=np.asarray(returns, dtype=np.float64),
        turnover=np.asarray(turnover, dtype=np.float64),
        costs=np.asarray(costs, dtype=np.float64),
        net=np.asarray(net, dtype=np.float64),
        nav=np.asarray(nav, dtype=np.float64),
        initial_nav=float(initial_nav),
    )


def objective_value_jax(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams,
    objective: str,
    *,
    mode: str = "smooth",
    beta: float = GRAD_BETA,
    initial_nav: float = 1.0,
    score_start: int = 0,
) -> float:
    """Scalar objective. Smooth drawdown is used unless ``mode='hard'``."""
    book = harden(params) if mode == "hard" else params
    _w, _r, _t, _c, net, _nav = _call(
        prices, book, strategy, mode=mode, beta=beta, initial_nav=initial_nav
    )
    value = _objective_from_net(
        net,
        objective,
        initial_nav=initial_nav,
        periods=float(book.periods_per_year),
        score_start=int(score_start),
        beta=float(beta),
        smooth_drawdown=mode != "hard",
    )
    return float(value)


def objective_gradients(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams | None = None,
    objective: str = "pnl",
    *,
    mode: str = "smooth",
    beta: float = GRAD_BETA,
    initial_nav: float = 1.0,
    score_start: int = 0,
) -> GradientResult:
    """Reverse-mode gradient of ``objective`` w.r.t. active parameters and prices.

    ``mode='smooth'`` differentiates the relaxed book (finite-difference
    checkable). ``mode='ste'`` is the straight-through surrogate. ``mode='hard'``
    is the subgradient of the kinked book (zero across flat discrete regions).
    """
    jax, jnp = _libs()
    book = params or StrategyParams()
    if objective not in {"pnl", "pnl_sum", "sharpe", "drawdown"}:
        raise ValueError(f"unknown objective {objective!r}")
    if mode == "hard":
        book = harden(book)
    validate_params(strategy, book)
    px = validate_prices(np.asarray(prices, dtype=np.float64))
    names = active_parameters(strategy)
    theta0 = pack(book, names)
    lookback_i, skip_i, vol_i, topk_i, long_only_i, require_i, long_short_i = _static_ints(book)
    periods = float(book.periods_per_year)
    delay = int(book.delay)
    every = int(book.rebalance_every)

    def scalar(theta: Any, path: Any) -> Any:
        _w, _r, _t, _c, net, _nav = _forward(
            path,
            theta,
            strategy=strategy,
            mode=mode,
            beta=float(beta),
            delay=delay,
            every=every,
            periods=periods,
            initial_nav=float(initial_nav),
            names=names,
            lookback_i=lookback_i,
            skip_i=skip_i,
            vol_i=vol_i,
            topk_i=topk_i,
            long_only_i=long_only_i,
            require_i=require_i,
            long_short_i=long_short_i,
        )
        return _objective_from_net(
            net,
            objective,
            initial_nav=float(initial_nav),
            periods=periods,
            score_start=int(score_start),
            beta=float(beta),
            smooth_drawdown=mode != "hard",
        )

    grad_fn = jax.jit(jax.value_and_grad(scalar, argnums=(0, 1)))
    value, (g_theta, g_price) = grad_fn(jnp.asarray(theta0), jnp.asarray(px))
    return GradientResult(
        objective=objective,
        value=float(value),
        parameter_names=names,
        parameter_gradient=np.asarray(g_theta, dtype=np.float64),
        price_gradient=np.asarray(g_price, dtype=np.float64),
        mode=mode,
        beta=float(beta),
    )
