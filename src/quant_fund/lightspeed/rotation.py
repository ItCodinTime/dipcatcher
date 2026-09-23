"""Causal TQQQ/SGOV rotation. Port of cosmic-hydra/lightspeed strategy.py.

Decisions use completed bars only. Weights at t are formed from closes[0..t]
and then delay-shifted. No Alpaca, no live fills.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from quant_fund.lightspeed.ema import (
    clip,
    delay_weights,
    quantize,
    realized_ann_vol_from_returns,
    sma_seeded_ema,
)
from quant_fund.lightspeed.specs import TqqqParams, TqqqSpec, tqqq_long_full_v1

_EPS = 1e-12


def _raw_sleeves(
    p: TqqqParams,
    gap: float,
    qqq_vol: float,
    tqqq_vol: float,
    qqq_close: NDArray[np.float64],
    i: int,
) -> tuple[float, float]:
    downtrend = gap < 0.0
    if p.flatten_when_fast_below_slow and downtrend:
        inverse = 0.0
        if p.sqqq_target > 0 and p.inverse_confirm > 0 and i >= p.inverse_confirm:
            mom = qqq_close[i] / qqq_close[i - p.inverse_confirm] - 1.0
            if mom < 0:
                inverse = p.sqqq_target
        return 0.0, inverse

    if tqqq_vol <= _EPS and qqq_vol <= _EPS:
        raw = p.max_tqqq_weight
    else:
        ref = qqq_vol if p.vol_ref == "signal" else tqqq_vol
        raw = p.max_tqqq_weight if ref <= _EPS else p.vol_budget / ref
    risk = clip(raw, p.min_tqqq_weight, p.max_tqqq_weight)

    if qqq_vol >= p.crash_vol and 0.0 <= gap <= p.crash_gap_max:
        risk = 0.0
    elif 0.0 <= gap <= p.weak_trend_gap:
        risk *= p.weak_trend_multiplier

    if p.crash_lookback > 0 and i >= p.crash_lookback:
        trail = qqq_close[i] / qqq_close[i - p.crash_lookback] - 1.0
        if trail <= p.crash_return:
            risk = 0.0

    inverse = 0.0
    risk_gross = risk + inverse
    if risk_gross > p.max_risk_weight and risk_gross > 0:
        scale = p.max_risk_weight / risk_gross
        risk *= scale
        inverse *= scale
    return risk, inverse


def _apply_bands(
    p: TqqqParams,
    risk: float,
    inverse: float,
    prev_risk: float,
    prev_inv: float,
) -> tuple[float, float]:
    risk = quantize(risk, p.weight_quantum)
    risk = clip(risk, 0.0, p.max_tqqq_weight)
    if abs(risk - prev_risk) < p.no_trade_band:
        risk = prev_risk
    if abs(inverse - prev_inv) < p.no_trade_band:
        inverse = prev_inv
    if risk + inverse > p.max_risk_weight and (risk + inverse) > 0:
        scale = p.max_risk_weight / (risk + inverse)
        risk *= scale
        inverse *= scale
        risk = quantize(risk, p.weight_quantum)
    return risk, inverse


def tqqq_target_weights(
    qqq_close: NDArray[np.float64],
    tqqq_close: NDArray[np.float64],
    spec: TqqqSpec | None = None,
    *,
    periods_per_year: float = 252.0,
    risk_multiplier: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Path of desired sleeve weights. Each row sums to 1.0 after warmup."""
    book = spec or tqqq_long_full_v1()
    p = book.params
    qqq = np.asarray(qqq_close, dtype=float)
    tqqq = np.asarray(tqqq_close, dtype=float)
    n = len(qqq)
    if len(tqqq) != n:
        raise ValueError("QQQ and TQQQ series must be aligned")
    multiplier = None
    if risk_multiplier is not None:
        multiplier = np.asarray(risk_multiplier, dtype=float)
        if len(multiplier) != n:
            raise ValueError("risk_multiplier must align with the price series")

    qqq_ret = np.zeros(n)
    tqqq_ret = np.zeros(n)
    qqq_ret[1:] = qqq[1:] / np.maximum(qqq[:-1], _EPS) - 1.0
    tqqq_ret[1:] = tqqq[1:] / np.maximum(tqqq[:-1], _EPS) - 1.0

    fast = sma_seeded_ema(qqq, p.fast_ema)
    slow = sma_seeded_ema(qqq, p.slow_ema)
    qqq_vol = realized_ann_vol_from_returns(qqq_ret, p.vol_window, periods_per_year)
    tqqq_vol = realized_ann_vol_from_returns(tqqq_ret, p.vol_window, periods_per_year)

    w_risk = np.zeros(n)
    w_inv = np.zeros(n)
    warmup = book.warmup
    prev_risk = 0.0
    prev_inv = 0.0

    for i in range(n):
        if i < warmup or not np.isfinite(fast[i]) or not np.isfinite(slow[i]):
            continue
        is_rebalance = (i - warmup) % p.rebalance_every == 0
        if not is_rebalance:
            w_risk[i] = prev_risk
            w_inv[i] = prev_inv
            continue

        gap = float((fast[i] - slow[i]) / max(qqq[i], _EPS))
        qv = float(qqq_vol[i]) if np.isfinite(qqq_vol[i]) else 0.0
        tv = float(tqqq_vol[i]) if np.isfinite(tqqq_vol[i]) else 0.0
        risk, inverse = _raw_sleeves(p, gap, qv, tv, qqq, i)
        if multiplier is not None:
            m = clip(float(multiplier[i]), 0.0, 1.0)
            if not math.isfinite(m):
                m = 1.0
            risk *= m
            inverse *= m
        risk, inverse = _apply_bands(p, risk, inverse, prev_risk, prev_inv)
        w_risk[i] = risk
        w_inv[i] = inverse
        prev_risk = risk
        prev_inv = inverse

    w_risk = delay_weights(w_risk, p.signal_delay_sessions)
    w_inv = delay_weights(w_inv, p.signal_delay_sessions)
    w_def = np.maximum(1.0 - w_risk - w_inv, 0.0)
    out = {
        book.instruments.risk_on: w_risk,
        book.instruments.defensive: w_def,
    }
    if book.instruments.inverse:
        out[book.instruments.inverse] = w_inv
    return out
