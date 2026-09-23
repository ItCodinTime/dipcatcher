"""Directional Moskowitz TSMOM / Antonacci books. Research only.

The CS ranker on ``future_idio_return_1`` is market-neutral ranking.
These books trade *total-return* paths: long-only 12–1, dual momentum,
ETF-basket TSMOM. Delay 1. Costs on turnover. Not a CS champion.
``blend_weight`` stays 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.quant_models.tsmom import tsmom_weights

Array = NDArray[np.float64]
_EPS = 1e-12

ETF_BASKET: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "EEM",
    "EFA",
    "TLT",
    "IEF",
    "GLD",
    "SLV",
    "XLE",
    "XLF",
    "XLK",
    "XLI",
    "XLP",
    "XLU",
    "XLV",
    "HYG",
    "LQD",
)


def simple_returns(px: Array) -> Array:
    """Close-to-close simple returns. Row 0 is 0."""
    p = np.asarray(px, dtype=float)
    if p.ndim == 1:
        r = np.zeros_like(p)
        r[1:] = p[1:] / np.maximum(p[:-1], _EPS) - 1.0
        r[~np.isfinite(r)] = 0.0
        return r
    r = np.zeros_like(p)
    r[1:] = p[1:] / np.maximum(p[:-1], _EPS) - 1.0
    r[~np.isfinite(r)] = 0.0
    return r


def tsmom_book_returns(
    px: Array,
    *,
    lookback: int = 252,
    skip: int = 21,
    vol_lookback: int = 60,
    asset_vol: float = 0.40,
    long_only: bool = True,
    delay: int = 1,
    rebalance_every: int = 21,
    one_way_cost: float = 0.001,
    max_gross: float = 1.0,
) -> Array:
    """MOP 12–1 weights on a close panel ``(T, N)``.

    ``w[t]`` uses returns through ``t-delay``. PnL at ``t`` is ``w[t]·r[t]``.
    Long-only clips shorts to cash, then scales to ``max_gross``. Monthly
    rebalance is the paper's unit (``rebalance_every=21``).
    """
    p = np.asarray(px, dtype=float)
    if p.ndim != 2:
        raise ValueError("px must be (T, N)")
    r = simple_returns(p)
    t_len, n_names = r.shape
    w = np.zeros((t_len, n_names), dtype=float)
    step = max(int(rebalance_every), 1)
    delay = max(int(delay), 0)
    start = lookback + delay
    prev: Array | None = None
    for t in range(start, t_len):
        if prev is not None and (t - start) % step != 0:
            w[t] = prev
            continue
        end = t - delay + 1
        if end < lookback + 1:
            continue
        raw = tsmom_weights(
            r[:end],
            lookback=lookback,
            skip=skip,
            vol_lookback=vol_lookback,
            target_vol=asset_vol,
        )
        if long_only:
            raw = np.maximum(raw, 0.0)
        gross = float(np.nansum(np.abs(raw)))
        if gross > float(max_gross) > 0.0:
            raw = raw * (float(max_gross) / gross)
        raw = np.where(np.isfinite(raw), raw, 0.0)
        w[t] = raw
        prev = raw
    pnl = np.sum(w * r, axis=1)
    turn = np.zeros(t_len, dtype=float)
    turn[1:] = np.sum(np.abs(w[1:] - w[:-1]), axis=1)
    return pnl - float(one_way_cost) * turn


def _passes_trend_crash(
    p: Array,
    i: int,
    end: int,
    *,
    sma: int,
    crash_lookback: int,
    crash_return: float,
) -> bool:
    """Price filters known at ``end`` (inclusive). ``sma=0`` / ``crash_lookback=0`` off."""
    px = float(p[end, i])
    if not np.isfinite(px) or px <= _EPS:
        return False
    if sma > 0:
        if end < sma - 1:
            return False
        window = p[end - sma + 1 : end + 1, i]
        if (not np.isfinite(window).all()) or px <= float(np.mean(window)):
            return False
    if crash_lookback > 0:
        if end < crash_lookback:
            return False
        base = float(p[end - crash_lookback, i])
        if not np.isfinite(base) or base <= _EPS:
            return False
        if px / base - 1.0 <= float(crash_return):
            return False
    return True


def topk_long_returns(
    px: Array,
    *,
    lookback: int = 252,
    skip: int = 21,
    top_k: int = 5,
    delay: int = 1,
    rebalance_every: int = 21,
    one_way_cost: float = 0.001,
    require_positive: bool = True,
    sma: int = 0,
    crash_lookback: int = 0,
    crash_return: float = -0.2,
) -> Array:
    """Equal-weight the top-k 12–1 names (optional: only if 12–1 > 0).

    Ranks refresh every ``rebalance_every`` bars. A positive ``sma`` or
    ``crash_lookback`` is checked every day on the current names and
    flattens failures to cash. Both use prices through ``t-delay``.
    """
    p = np.asarray(px, dtype=float)
    if p.ndim != 2:
        raise ValueError("px must be (T, N)")
    if sma < 0 or crash_lookback < 0:
        raise ValueError("sma and crash_lookback must be >= 0")
    r = simple_returns(p)
    t_len, n_names = r.shape
    k = max(int(top_k), 1)
    w = np.zeros((t_len, n_names), dtype=float)
    step = max(int(rebalance_every), 1)
    delay = max(int(delay), 0)
    warm = max(int(lookback), int(sma), int(crash_lookback))
    start = warm + delay
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
                mom = p[skip_i] / np.maximum(p[past], _EPS) - 1.0
                mom = np.where(np.isfinite(mom), mom, -np.inf)
                if require_positive:
                    mom = np.where(mom > 0.0, mom, -np.inf)
                order = np.argsort(mom)[::-1]
                picked = [int(i) for i in order[:k] if np.isfinite(mom[int(i)])]
            held = picked
        alive = [
            i
            for i in picked
            if _passes_trend_crash(
                p,
                i,
                end,
                sma=int(sma),
                crash_lookback=int(crash_lookback),
                crash_return=float(crash_return),
            )
        ]
        raw = np.zeros(n_names, dtype=float)
        if alive:
            raw[alive] = 1.0 / float(len(alive))
        w[t] = raw
    pnl = np.sum(w * r, axis=1)
    turn = np.zeros(t_len, dtype=float)
    turn[1:] = np.sum(np.abs(w[1:] - w[:-1]), axis=1)
    return pnl - float(one_way_cost) * turn


def antonacci_returns(
    spy: Array,
    tlt: Array,
    *,
    lookback: int = 252,
    skip: int = 21,
    delay: int = 1,
    rebalance_every: int = 21,
    one_way_cost: float = 0.001,
) -> Array:
    """GEM dual momentum: SPY vs TLT vs cash. 12–1, delay 1, monthly."""
    spy = np.asarray(spy, dtype=float)
    tlt = np.asarray(tlt, dtype=float)
    if spy.shape != tlt.shape:
        raise ValueError("spy and tlt must align")
    r_s = simple_returns(spy)
    r_t = simple_returns(tlt)
    n = spy.size
    pos_s = np.zeros(n, dtype=float)
    pos_t = np.zeros(n, dtype=float)
    step = max(int(rebalance_every), 1)
    delay = max(int(delay), 0)
    start = lookback + delay
    last = (0.0, 0.0)
    for t in range(start, n):
        if t > start and (t - start) % step != 0:
            pos_s[t], pos_t[t] = last
            continue
        end = t - delay
        past = end - lookback
        skip_i = end - skip
        if past < 0 or skip_i <= past:
            continue
        if spy[past] <= _EPS or tlt[past] <= _EPS:
            last = (0.0, 0.0)
            pos_s[t], pos_t[t] = last
            continue
        m_s = spy[skip_i] / spy[past] - 1.0
        m_t = tlt[skip_i] / tlt[past] - 1.0
        if (not np.isfinite(m_s)) or (not np.isfinite(m_t)):
            last = (0.0, 0.0)
        elif m_s > m_t and m_s > 0.0:
            last = (1.0, 0.0)
        elif m_t > 0.0:
            last = (0.0, 1.0)
        else:
            last = (0.0, 0.0)
        pos_s[t], pos_t[t] = last
    pnl = pos_s * r_s + pos_t * r_t
    turn = np.zeros(n, dtype=float)
    turn[1:] = np.abs(pos_s[1:] - pos_s[:-1]) + np.abs(pos_t[1:] - pos_t[:-1])
    return pnl - float(one_way_cost) * turn


def risk_parity_blend(
    streams: dict[str, Array],
    *,
    lookback: int = 60,
    delay: int = 1,
    one_way_cost: float = 0.0,
) -> Array:
    """Delay-1 inverse-vol mix. Weights at t use data through t-delay."""
    names = list(streams)
    mat = np.column_stack([np.asarray(streams[k], dtype=float) for k in names])
    t_len, n_s = mat.shape
    w = np.zeros((t_len, n_s), dtype=float)
    lb = max(int(lookback), 8)
    delay = max(int(delay), 0)
    for t in range(lb + delay, t_len):
        end = t - delay + 1
        window = mat[end - lb : end]
        sig = np.std(window, axis=0, ddof=1)
        inv = np.zeros_like(sig)
        ok = np.isfinite(sig) & (sig > _EPS)
        inv[ok] = 1.0 / sig[ok]
        s = float(inv.sum())
        if s > _EPS:
            w[t] = inv / s
    pnl = np.sum(w * mat, axis=1)
    turn = np.zeros(t_len, dtype=float)
    turn[1:] = np.sum(np.abs(w[1:] - w[:-1]), axis=1)
    return pnl - float(one_way_cost) * turn


def close_matrix(
    closes: dict[str, Array], names: tuple[str, ...] | list[str]
) -> tuple[list[str], Array]:
    keep = [s for s in names if s in closes]
    if not keep:
        raise ValueError("none of the requested names are on the tape")
    n = min(len(closes[s]) for s in keep)
    mat = np.column_stack([np.asarray(closes[s][:n], dtype=float) for s in keep])
    return keep, mat


def spy_buy_hold(spy: Array) -> Array:
    return simple_returns(np.asarray(spy, dtype=float))


def stream_card(name: str, returns: Array) -> dict[str, Any]:
    from quant_fund.hedge_lab.scoreboard import book_economic_scoreboard

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    econ = book_economic_scoreboard(r, data_source="file")
    econ["name"] = name
    return econ
