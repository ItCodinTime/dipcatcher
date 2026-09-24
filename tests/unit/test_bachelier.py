"""Tests for models/bachelier.py — Bachelier normal option model."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.bachelier import (
    bachelier_greeks,
    bachelier_implied_vol,
    bachelier_price,
)


def test_put_call_parity() -> None:
    f, k, t, s = 100.0, 105.0, 0.5, 8.0
    call = bachelier_price(f, k, t, s, "call")
    put = bachelier_price(f, k, t, s, "put")
    assert abs((call - put) - (f - k)) < 1e-8


def test_atm_price_formula() -> None:
    f, t, s = 100.0, 1.0, 10.0
    atm = bachelier_price(f, f, t, s, "call")
    assert abs(atm - s * np.sqrt(t / (2.0 * np.pi))) < 1e-8


def test_price_increases_with_vol() -> None:
    lo = bachelier_price(100.0, 100.0, 1.0, 5.0)
    hi = bachelier_price(100.0, 100.0, 1.0, 15.0)
    assert hi > lo


def test_implied_vol_roundtrip() -> None:
    f, k, t, s = 100.0, 98.0, 0.75, 12.0
    px = bachelier_price(f, k, t, s, "call")
    iv = bachelier_implied_vol(px, f, k, t, "call")
    assert abs(iv - s) < 1e-6


def test_greeks_bounds() -> None:
    g = bachelier_greeks(100.0, 100.0, 1.0, 10.0)
    assert abs(g["delta_call"] - 0.5) < 1e-6
    assert g["vega"] > 0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        bachelier_price(100.0, 100.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        bachelier_price(100.0, 100.0, 1.0, 5.0, option="swap")
    with pytest.raises(ValueError):
        bachelier_implied_vol(-1.0, 100.0, 100.0, 1.0)
