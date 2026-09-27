"""Tests for metrics/liquidity_extra.py — Hui-Heubel and Martin ratios."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.liquidity_extra import (
    hui_heubel_ratio,
    martin_ratio,
    ulcer_index,
)


def test_hui_heubel_decreases_with_volume() -> None:
    rng = np.random.default_rng(0)
    prices = 100.0 + np.cumsum(0.5 * rng.standard_normal(20))
    low_vol = np.full(20, 1e4)
    high_vol = np.full(20, 1e6)
    r_low = hui_heubel_ratio(prices, low_vol, shares_outstanding=1e7)
    r_high = hui_heubel_ratio(prices, high_vol, shares_outstanding=1e7)
    assert r_low > r_high > 0


def test_martin_ratio_rewards_smoothness() -> None:
    rng = np.random.default_rng(1)
    smooth = 0.001 + 0.003 * rng.standard_normal(500)
    choppy = 0.001 + 0.02 * rng.standard_normal(500)
    assert ulcer_index(choppy) > ulcer_index(smooth)
    assert martin_ratio(smooth) > martin_ratio(choppy)


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        hui_heubel_ratio(np.array([100.0]), np.array([1.0]), 1e6)  # too short
    with pytest.raises(ValueError):
        martin_ratio(0.01 + np.zeros(50))  # no drawdown -> undefined


def test_ulcer_includes_starting_wealth_and_martin_uses_percent_units() -> None:
    returns = np.array([-0.10, 1.0 / 9.0])
    assert ulcer_index(returns) == pytest.approx(10.0 / np.sqrt(2.0))
    annual_excess_pct = 100.0 * returns.mean() * 252.0
    assert martin_ratio(returns) == pytest.approx(
        annual_excess_pct / ulcer_index(returns)
    )


def test_liquidity_and_drawdown_invalid_inputs_rejected() -> None:
    prices = np.array([100.0, 101.0])
    volume = np.array([1000.0, 1000.0])
    for shares in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            hui_heubel_ratio(prices, volume, shares)
    for returns in (np.array([-1.0, 0.1]), np.array([-1.1, 0.1])):
        with pytest.raises(ValueError):
            ulcer_index(returns)
    with pytest.raises(ValueError):
        martin_ratio(np.array([-0.01, 0.02]), periods_per_year=0.0)
