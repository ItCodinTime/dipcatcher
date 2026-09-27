"""Tests for models/leisen_reimer.py — Leisen-Reimer binomial tree."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.leisen_reimer import leisen_reimer


def _bs(s: float, k: float, r: float, q: float, t: float, sigma: float, call: bool) -> float:
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if call:
        return float(s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    return float(k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1))


def test_european_matches_black_scholes() -> None:
    s, k, r, q, t, sigma = 100.0, 105.0, 0.04, 0.01, 1.0, 0.22
    for opt, call in (("call", True), ("put", False)):
        lr = leisen_reimer(s, k, t, r, q, sigma, n=101, option=opt, american=False)
        bs = _bs(s, k, r, q, t, sigma, call)
        assert abs(lr - bs) < 0.01


def test_fast_convergence() -> None:
    s, k, r, q, t, sigma = 100.0, 100.0, 0.05, 0.0, 0.5, 0.2
    bs = _bs(s, k, r, q, t, sigma, True)
    err = abs(leisen_reimer(s, k, t, r, q, sigma, n=51, option="call") - bs)
    assert err < 0.01  # accurate even with few steps


def test_american_put_premium() -> None:
    s, k, r, q, t, sigma = 100.0, 110.0, 0.06, 0.0, 1.0, 0.3
    amer = leisen_reimer(s, k, t, r, q, sigma, n=201, option="put", american=True)
    euro = leisen_reimer(s, k, t, r, q, sigma, n=201, option="put", american=False)
    assert amer >= euro - 1e-9
    assert amer >= (k - s) - 1e-9


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        leisen_reimer(100.0, 100.0, 1.0, 0.05, 0.0, 0.0)
    with pytest.raises(ValueError):
        leisen_reimer(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, option="swap")
