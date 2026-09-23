"""Causal top-K relative-strength momentum. Port of stockbook/strategy.py."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from quant_fund.lightspeed.ema import (
    clip,
    delay_weights,
    quantize,
    realized_ann_vol_from_closes,
    rolling_mean,
)
from quant_fund.lightspeed.specs import MomentumSpec, nautica_momentum_v1

_EPS = 1e-12


def momentum_scores(
    closes: dict[str, NDArray[np.float64]],
    symbols: tuple[str, ...],
    mom_fast: int,
    mom_slow: int,
    mom_blend: float,
) -> dict[str, NDArray[np.float64]]:
    """Causal blended momentum score path per symbol (NaN until defined)."""
    out: dict[str, NDArray[np.float64]] = {}
    for symbol in symbols:
        prices = np.asarray(closes[symbol], dtype=float)
        n = len(prices)
        score = np.full(n, np.nan, dtype=float)
        fast = np.full(n, np.nan, dtype=float)
        slow = np.full(n, np.nan, dtype=float)
        if n > mom_fast:
            fast[mom_fast:] = prices[mom_fast:] / np.maximum(prices[:-mom_fast], _EPS) - 1.0
        if n > mom_slow:
            slow[mom_slow:] = prices[mom_slow:] / np.maximum(prices[:-mom_slow], _EPS) - 1.0
        both = np.isfinite(fast) & np.isfinite(slow)
        score[both] = mom_blend * fast[both] + (1.0 - mom_blend) * slow[both]
        only_fast = ~both & np.isfinite(fast) & (mom_blend >= 1.0 - 1e-12)
        score[only_fast] = fast[only_fast]
        only_slow = ~both & np.isfinite(slow) & (mom_blend <= 1e-12)
        score[only_slow] = slow[only_slow]
        out[symbol] = score
    return out


def momentum_target_weights(
    closes: dict[str, NDArray[np.float64]],
    spec: MomentumSpec | None = None,
    *,
    periods_per_year: float = 252.0,
    risk_multiplier: dict[str, NDArray[np.float64]] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Path of desired weights. Risk sleeves + defensive residual sum to 1.0."""
    book = spec or nautica_momentum_v1()
    p = book.params
    symbols = book.universe.risk
    position_cap = min(p.max_position_weight, book.max_single_name_weight)
    n = len(np.asarray(closes[symbols[0]], dtype=float))
    for symbol in book.universe.traded:
        if symbol not in closes:
            raise ValueError(f"missing close series for {symbol}")
        if len(np.asarray(closes[symbol])) != n:
            raise ValueError("all universe series must be aligned")
    if risk_multiplier is not None:
        for path in risk_multiplier.values():
            if len(path) != n:
                raise ValueError("risk_multiplier paths must align with prices")

    scores = momentum_scores(closes, symbols, p.mom_fast, p.mom_slow, p.mom_blend)
    smas = {s: rolling_mean(np.asarray(closes[s], dtype=float), p.trend_sma) for s in symbols}
    vols = {
        s: realized_ann_vol_from_closes(
            np.asarray(closes[s], dtype=float), p.vol_window, periods_per_year
        )
        for s in symbols
    }

    weights = {s: np.zeros(n) for s in symbols}
    warmup = p.warmup
    prev = {s: 0.0 for s in symbols}

    for i in range(n):
        if i < warmup:
            continue

        crashed: set[str] = set()
        if p.crash_lookback > 0 and i >= p.crash_lookback:
            for s in symbols:
                if prev.get(s, 0.0) > _EPS:
                    trail = closes[s][i] / max(closes[s][i - p.crash_lookback], _EPS) - 1.0
                    if trail <= p.crash_return:
                        crashed.add(s)
        if crashed:
            for s in crashed:
                prev[s] = 0.0
            for s in symbols:
                weights[s][i] = prev[s]
            continue

        is_rebalance = (i - warmup) % p.rebalance_every == 0
        if not is_rebalance:
            for s in symbols:
                weights[s][i] = prev[s]
            continue

        eligible: list[tuple[float, str]] = []
        for s in symbols:
            sc = scores[s][i]
            sma = smas[s][i]
            if not (np.isfinite(sc) and np.isfinite(sma)):
                continue
            if sc <= p.min_score:
                continue
            if closes[s][i] <= sma:
                continue
            if p.crash_lookback > 0 and i >= p.crash_lookback:
                trail = closes[s][i] / max(closes[s][i - p.crash_lookback], _EPS) - 1.0
                if trail <= p.crash_return:
                    continue
            eligible.append((float(sc), s))
        eligible.sort(reverse=True)
        picked = [s for _, s in eligible[: p.top_k]]

        raw = {s: 0.0 for s in symbols}
        for s in picked:
            v = vols[s][i]
            if not np.isfinite(v) or v <= _EPS:
                raw[s] = position_cap
            else:
                raw[s] = clip(p.vol_budget / v, 0.0, position_cap)
            if risk_multiplier is not None and s in risk_multiplier:
                m = clip(float(risk_multiplier[s][i]), 0.0, 1.0)
                if math.isfinite(m):
                    raw[s] *= m

        banded: dict[str, float] = {}
        for s in symbols:
            tgt = raw[s]
            if s in picked and prev[s] > _EPS and abs(tgt - prev[s]) < p.no_trade_band:
                tgt = prev[s]
            banded[s] = clip(quantize(tgt, p.weight_quantum), 0.0, position_cap)

        gross = sum(banded.values())
        if gross > p.max_gross_weight and gross > 0:
            scale = p.max_gross_weight / gross
            banded = {
                s: clip(quantize(banded[s] * scale, p.weight_quantum), 0.0, position_cap)
                for s in symbols
            }

        for s in symbols:
            weights[s][i] = banded[s]
        prev = dict(banded)

    for s in symbols:
        weights[s] = delay_weights(weights[s], p.signal_delay_sessions)

    defensive = np.ones(n)
    for s in symbols:
        defensive -= weights[s]
    out: dict[str, NDArray[np.float64]] = dict(weights)
    out[book.universe.defensive] = np.maximum(defensive, 0.0)
    return out
