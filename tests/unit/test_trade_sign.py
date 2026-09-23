"""Tests for models/trade_sign.py — Lee-Ready, tick rule, BVC."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.trade_sign import (
    bulk_volume_classify,
    lee_ready,
    signed_volume,
    signing_accuracy,
    tick_rule,
)


def test_tick_rule_basic() -> None:
    p = np.array([10.0, 10.1, 10.1, 9.9, 9.9, 10.2])
    s = tick_rule(p)
    assert s[0] == 0.0
    assert list(s[1:]) == [1.0, 1.0, -1.0, -1.0, 1.0]


def test_lee_ready_quote_rule() -> None:
    bid = np.array([99.9, 99.9, 99.9])
    ask = np.array([100.1, 100.1, 100.1])
    price = np.array([100.1, 99.9, 100.0])
    s = lee_ready(price, bid, ask)
    assert s[0] == 1.0 and s[1] == -1.0
    assert s[2] in (-1.0, 0.0, 1.0)  # mid -> tick rule on mid (flat -> 0)


def test_lee_ready_mid_uses_tick() -> None:
    bid = np.array([99.9, 99.9, 99.8, 99.8])
    ask = np.array([100.1, 100.1, 100.0, 100.0])
    mid = 0.5 * (bid + ask)
    price = mid.copy()
    s = lee_ready(price, bid, ask)
    # mid falls at index 2->3 ... tick rule on mid: dp[1]=0,dp[2]=-0.05
    assert s[3] == -1.0


def test_bvc_upward_drift_is_buy() -> None:
    rng = np.random.default_rng(4)
    n = 300
    dp = 0.5 + 0.5 * rng.standard_normal(n)  # persistent up-drift
    close = 100.0 + np.concatenate([[0.0], np.cumsum(dp)])
    vol = np.full(n + 1, 1000.0)
    out = bulk_volume_classify(close, vol)
    assert float(out["buy_share"].mean()) > 0.7
    assert np.isfinite(out["buy_volume"]).all()


def test_signed_volume_imbalance() -> None:
    signs = np.array([1.0, 1.0, -1.0, 1.0])
    vol = np.array([100.0, 200.0, 50.0, 150.0])
    out = signed_volume(signs, vol)
    assert out["buy_volume"] == 450.0
    assert out["sell_volume"] == 50.0
    assert out["imbalance"] == 400.0
    assert out["buy_share"] == pytest.approx(0.9)


def test_signing_accuracy() -> None:
    assert signing_accuracy(np.array([1, -1, 1]), np.array([1, -1, -1])) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        signing_accuracy(np.zeros(3), np.zeros(3))


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        tick_rule(np.array([np.nan, 1.0]))
    with pytest.raises(ValueError):
        lee_ready(np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        lee_ready(np.array([1.0]), np.array([1.5]), np.array([1.0]))  # crossed
    with pytest.raises(ValueError):
        bulk_volume_classify(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        signed_volume(np.array([1.0]), np.array([-5.0]))  # negative volume
