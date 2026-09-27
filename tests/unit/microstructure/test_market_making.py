"""Tests for models/market_making.py — Avellaneda-Stoikov."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.market_making import (
    as_inventory_bounds,
    as_optimal_quotes,
    as_optimal_spread,
    as_reservation_price,
    estimate_arrival_intensity,
)

G, S, T, K = 0.1, 0.02, 0.5, 1.5


def test_reservation_shifts_with_inventory() -> None:
    r_long = float(as_reservation_price(100.0, 10.0, G, S, T))
    r_flat = float(as_reservation_price(100.0, 0.0, G, S, T))
    r_short = float(as_reservation_price(100.0, -10.0, G, S, T))
    assert r_long < r_flat < r_short
    assert r_flat == pytest.approx(100.0)
    assert r_long == pytest.approx(100.0 - 10 * G * S * S * T)


def test_quotes_straddle_reservation() -> None:
    q = as_optimal_quotes(100.0, 5.0, G, S, T, K)
    assert q["bid"] < q["reservation_price"] < q["ask"]
    assert q["ask"] - q["bid"] == pytest.approx(as_optimal_spread(G, S, T, K))
    # long inventory -> skew downward (incentive to sell)
    assert q["skew"] < 0.0


def test_spread_comparative_statics() -> None:
    base = as_optimal_spread(G, S, T, K)
    assert as_optimal_spread(G, 2 * S, T, K) > base  # more vol -> wider
    assert as_optimal_spread(G, S, T, 10 * K) < base  # faster decay -> tighter
    assert as_optimal_spread(2 * G, S, 0.0, K) != base


def test_terminal_spread() -> None:
    # tau -> 0 leaves only the liquidity term
    s = as_optimal_spread(G, S, 0.0, K)
    assert s == pytest.approx((2.0 / G) * np.log(1.0 + G / K))


def test_estimate_kappa_recovery() -> None:
    depths = np.linspace(0.1, 2.0, 20)
    true_a, true_k = 100.0, 3.0
    lam = true_a * np.exp(-true_k * depths)
    out = estimate_arrival_intensity(depths, lam)
    assert out["kappa"] == pytest.approx(true_k, rel=1e-8)
    assert out["A"] == pytest.approx(true_a, rel=1e-6)


def test_inventory_bounds() -> None:
    out = as_inventory_bounds(20.0, G, S, T, K)
    assert out["reservation_range"] == pytest.approx(2 * 20 * G * S * S * T)
    assert out["skew_per_unit"] == pytest.approx(G * S * S * T)


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        as_reservation_price(100.0, 0.0, -0.1, S, T)
    with pytest.raises(ValueError):
        as_optimal_spread(G, 0.0, T, K)
    with pytest.raises(ValueError):
        as_optimal_quotes(100.0, 0.0, G, S, T, 0.0)
    with pytest.raises(ValueError):
        estimate_arrival_intensity(np.array([0.1, 0.2]), np.array([1.0, 0.9]))
    with pytest.raises(ValueError):
        estimate_arrival_intensity(
            np.array([0.1, 0.2, 0.3, 0.4]), np.array([1.0, 1.2, 1.1, 1.3])
        )  # rising -> slope >= 0
