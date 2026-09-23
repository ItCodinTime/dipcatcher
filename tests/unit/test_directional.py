"""Directional TSMOM / Antonacci / top-k long books."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.hedge_lab.directional import (
    antonacci_returns,
    risk_parity_blend,
    simple_returns,
    topk_long_returns,
    tsmom_book_returns,
)


def _trend_panel() -> np.ndarray:
    rng = np.random.default_rng(7)
    t, n = 400, 4
    px = np.full((t, n), 100.0)
    drift = np.array([0.002, 0.001, -0.001, 0.0002])
    for i in range(1, t):
        px[i] = px[i - 1] * (1.0 + drift + 0.004 * rng.normal(size=n))
    return px


def test_tsmom_long_only_does_not_short() -> None:
    px = _trend_panel()
    r = tsmom_book_returns(px, long_only=True, delay=1, rebalance_every=21, one_way_cost=0.0)
    assert r.shape[0] == px.shape[0]
    assert float(np.max(r[280:])) > 0.0


def test_tsmom_delay_one_ignores_today_close() -> None:
    px = _trend_panel()
    a = tsmom_book_returns(px, long_only=True, delay=1, rebalance_every=1, one_way_cost=0.0)
    px2 = px.copy()
    px2[350] *= 1.2
    b = tsmom_book_returns(px2, long_only=True, delay=1, rebalance_every=1, one_way_cost=0.0)
    # Weights at 350 use returns through 349; yesterday's PnL is untouched.
    assert float(a[349]) == pytest.approx(float(b[349]))
    # Tomorrow's weight sees the jump in r[350].
    assert abs(float(a[351]) - float(b[351])) > 1e-12


def test_topk_holds_the_winner() -> None:
    t = 300
    px = np.column_stack(
        [
            100.0 * np.exp(0.003 * np.arange(t)),
            100.0 * np.exp(0.0002 * np.arange(t)),
            100.0 * np.exp(-0.001 * np.arange(t)),
        ]
    )
    r = topk_long_returns(px, lookback=252, skip=21, top_k=1, delay=1, rebalance_every=21, one_way_cost=0.0)
    spy = simple_returns(px[:, 0])
    assert float(np.corrcoef(r[260:], spy[260:])[0, 1]) > 0.8


def test_trend_filter_flattens_a_name_under_its_sma() -> None:
    rise = 100.0 * np.exp(0.004 * np.arange(180))
    fall = rise[-1] * np.exp(-0.01 * np.arange(100))
    peak = np.concatenate([rise, fall])
    flat = np.full(peak.size, 100.0)
    px = np.column_stack([peak, flat])
    kwargs = {
        "lookback": 60,
        "skip": 5,
        "top_k": 1,
        "delay": 1,
        "rebalance_every": 5,
        "one_way_cost": 0.0,
    }
    bare = topk_long_returns(px, **kwargs)
    gated = topk_long_returns(px, sma=40, **kwargs)
    # Just after the peak, 60-day momentum is still positive but price is
    # already under the 40-day average.
    assert float(np.mean(np.abs(bare[185:220]))) > 0.0
    assert float(np.mean(np.abs(gated[185:220]))) < float(np.mean(np.abs(bare[185:220])))


def test_antonacci_goes_to_cash_when_both_negative() -> None:
    t = 300
    spy = 200.0 * np.exp(-0.002 * np.arange(t))
    tlt = 100.0 * np.exp(-0.002 * np.arange(t))
    r = antonacci_returns(spy, tlt, delay=1, rebalance_every=21, one_way_cost=0.0)
    assert float(np.max(np.abs(r[260:]))) == pytest.approx(0.0)


def test_risk_parity_delay_and_mix() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(0.0004, 0.01, size=200)
    b = rng.normal(0.0004, 0.03, size=200)
    mix = risk_parity_blend({"a": a, "b": b}, lookback=40, delay=1)
    assert mix.shape == (200,)
    assert float(np.std(mix[50:])) < float(np.std(b[50:]))
