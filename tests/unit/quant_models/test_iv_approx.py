"""Tests for models/iv_approx.py — closed-form implied-vol approximations."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.iv_approx import brenner_subrahmanyam_iv, corrado_miller_iv


def _bs_call(s: float, k: float, r: float, t: float, sigma: float) -> float:
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return float(s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))


def test_brenner_subrahmanyam_atm() -> None:
    s, t, sigma = 100.0, 1.0, 0.25
    price = _bs_call(s, s, 0.0, t, sigma)  # ATM, r=0
    approx = brenner_subrahmanyam_iv(price, s, t)
    assert abs(approx - sigma) < 0.02


def test_corrado_miller_near_money() -> None:
    s, r, t, sigma = 100.0, 0.02, 0.5, 0.3
    for k in (95.0, 100.0, 105.0):
        price = _bs_call(s, k, r, t, sigma)
        approx = corrado_miller_iv(price, s, k, t, r)
        assert abs(approx - sigma) < 0.03


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        brenner_subrahmanyam_iv(-1.0, 100.0, 1.0)
    with pytest.raises(ValueError):
        corrado_miller_iv(5.0, 100.0, 100.0, 0.0)
