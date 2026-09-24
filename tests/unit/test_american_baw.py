"""Tests for models/american_baw.py — Barone-Adesi-Whaley approximation."""

from __future__ import annotations

import pytest

from quant_fund.models.american_baw import baw_american
from quant_fund.quant_models.binomial import crr_american, crr_european


def test_call_no_dividend_equals_european() -> None:
    # With q = 0 (b = r) an American call should not be exercised early.
    baw = baw_american(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, "call")
    euro = crr_european(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 800, "call")
    assert abs(baw - euro) < 0.05


def test_put_matches_binomial() -> None:
    s, k, t, r, q, sigma = 100.0, 100.0, 0.5, 0.05, 0.05, 0.25
    baw = baw_american(s, k, t, r, q, sigma, "put")
    ref = crr_american(s, k, t, r, q, sigma, 1500, "put")
    assert abs(baw - ref) / ref < 0.03


def test_call_with_dividends_matches_binomial() -> None:
    s, k, t, r, q, sigma = 90.0, 100.0, 1.0, 0.08, 0.12, 0.3
    baw = baw_american(s, k, t, r, q, sigma, "call")
    ref = crr_american(s, k, t, r, q, sigma, 1500, "call")
    assert abs(baw - ref) / ref < 0.04


def test_american_at_least_intrinsic_and_european() -> None:
    s, k, t, r, q, sigma = 80.0, 100.0, 0.75, 0.05, 0.04, 0.2
    baw = baw_american(s, k, t, r, q, sigma, "put")
    assert baw >= (k - s) - 1e-9
    assert baw >= crr_european(s, k, t, r, q, sigma, 800, "put") - 1e-6


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        baw_american(-1.0, 100.0, 1.0, 0.05, 0.0, 0.2)
    with pytest.raises(ValueError):
        baw_american(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, option="swap")
