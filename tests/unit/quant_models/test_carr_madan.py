"""Tests for models/carr_madan.py — FFT option pricing vs Black-Scholes."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.carr_madan import bs_char_fn, carr_madan_call


def _bs_call(s0: float, k: float, r: float, t: float, sigma: float) -> float:
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return float(s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))


def test_matches_black_scholes_across_strikes() -> None:
    s0, r, t, sigma = 100.0, 0.03, 1.0, 0.2
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    prices = carr_madan_call(bs_char_fn(s0, r, t, sigma), r, t, strikes)
    ref = np.array([_bs_call(s0, k, r, t, sigma) for k in strikes])
    assert np.max(np.abs(prices - ref)) < 0.05


def test_prices_decrease_in_strike() -> None:
    s0, r, t, sigma = 100.0, 0.0, 0.5, 0.3
    strikes = np.linspace(70, 130, 13)
    prices = carr_madan_call(bs_char_fn(s0, r, t, sigma), r, t, strikes)
    assert np.all(np.diff(prices) < 0)


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        carr_madan_call(bs_char_fn(100.0, 0.0, 1.0, 0.2), 0.0, 0.0, np.array([100.0]))
    with pytest.raises(ValueError):
        carr_madan_call(bs_char_fn(100.0, 0.0, 1.0, 0.2), 0.0, 1.0, np.array([-100.0]))
