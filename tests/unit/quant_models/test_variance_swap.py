"""Tests for models/variance_swap.py — model-free variance replication."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.variance_swap import variance_swap_fair_strike


def _bs_call_put(forward: float, strike: float, sigma: float, r: float, t: float):
    vol = sigma * np.sqrt(t)
    d1 = (np.log(forward / strike) + 0.5 * vol**2) / vol
    d2 = d1 - vol
    disc = np.exp(-r * t)
    call = disc * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    put = disc * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
    return call, put


def test_replication_recovers_flat_bs_variance() -> None:
    f, r, t, sigma = 100.0, 0.0, 0.5, 0.2
    strikes = np.linspace(20.0, 300.0, 281)
    calls = np.array([_bs_call_put(f, k, sigma, r, t)[0] for k in strikes])
    puts = np.array([_bs_call_put(f, k, sigma, r, t)[1] for k in strikes])
    out = variance_swap_fair_strike(strikes, calls, puts, forward=f, r=r, maturity=t)
    # Under flat BS vol the fair variance equals sigma^2.
    assert abs(out["fair_variance"] - sigma**2) < 0.02 * sigma**2
    assert abs(out["fair_vol"] - sigma) < 0.01


def test_higher_vol_gives_higher_fair_variance() -> None:
    f, r, t = 100.0, 0.0, 0.5
    strikes = np.linspace(20.0, 300.0, 281)

    def fair(sig: float) -> float:
        calls = np.array([_bs_call_put(f, k, sig, r, t)[0] for k in strikes])
        puts = np.array([_bs_call_put(f, k, sig, r, t)[1] for k in strikes])
        return variance_swap_fair_strike(strikes, calls, puts, f, r, t)["fair_variance"]

    assert fair(0.3) > fair(0.15)


def test_fail_closed() -> None:
    k = np.linspace(80, 120, 5)
    with pytest.raises(ValueError):
        variance_swap_fair_strike(k, k, k[:-1], 100.0, 0.0, 0.5)  # length mismatch
    with pytest.raises(ValueError):
        variance_swap_fair_strike(k, k, k, 100.0, 0.0, 0.0)  # bad maturity
