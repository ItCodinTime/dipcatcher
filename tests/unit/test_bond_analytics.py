"""Tests for models/bond_analytics.py — price, YTM, duration, convexity."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.bond_analytics import (
    bond_price,
    convexity,
    dv01,
    macaulay_duration,
    yield_to_maturity,
)


def _coupon_bond(mat: int = 5, coupon: float = 0.05, face: float = 100.0):
    times = np.arange(1, mat + 1, dtype=float)
    cfs = np.full(mat, coupon * face)
    cfs[-1] += face
    return times, cfs


def test_zero_coupon_duration_equals_maturity() -> None:
    times = np.array([7.0])
    cfs = np.array([100.0])
    y = 0.03
    assert abs(macaulay_duration(times, cfs, y) - 7.0) < 1e-12


def test_ytm_inverts_price() -> None:
    times, cfs = _coupon_bond()
    price = bond_price(times, cfs, 0.04)
    y = yield_to_maturity(price, times, cfs)
    assert abs(y - 0.04) < 1e-8


def test_price_decreasing_in_yield() -> None:
    times, cfs = _coupon_bond()
    assert bond_price(times, cfs, 0.02) > bond_price(times, cfs, 0.06)


def test_duration_and_convexity_positive() -> None:
    times, cfs = _coupon_bond()
    y = 0.04
    d = macaulay_duration(times, cfs, y)
    c = convexity(times, cfs, y)
    assert 0.0 < d < 5.0
    assert c > 0.0
    assert dv01(times, cfs, y) > 0.0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        bond_price(np.array([1.0, 2.0]), np.array([1.0]), 0.03)
    with pytest.raises(ValueError):
        yield_to_maturity(-5.0, np.array([1.0]), np.array([100.0]))
