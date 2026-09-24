"""Tests for BSM pricing, Greeks, implied vol."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.options import (
    bs_greeks,
    bs_price,
    implied_vol,
    put_call_parity_gap,
    risk_neutral_density,
)


class TestBSM:
    def test_atm_call_put_parity(self):
        S, K, T, sig, r = 100.0, 100.0, 0.5, 0.2, 0.01
        c = bs_price(S, K, T, sig, r, call=True)
        p = bs_price(S, K, T, sig, r, call=False)
        assert abs(c - p - (S - K * np.exp(-r * T))) < 1e-9
        assert c > 0 and p > 0

    def test_deep_itm_put(self):
        p = bs_price(50.0, 100.0, 1.0, 0.3, 0.0, call=False)
        assert abs(p - 50.0) < 2.0  # approx intrinsic + time value

    def test_greeks_bounds(self):
        g = bs_greeks(100.0, 100.0, 0.25, 0.25, 0.02, call=True)
        assert 0 < g["delta"] < 1
        assert g["gamma"] > 0
        assert g["vega"] > 0
        assert g["theta"] < 0  # long option decays
        gp = bs_greeks(100.0, 100.0, 0.25, 0.25, 0.02, call=False)
        assert gp["delta"] < 0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            bs_price(-1.0, 100.0, 0.5, 0.2)
        with pytest.raises(ValueError):
            bs_price(100.0, 100.0, 0.0, 0.2)


class TestImpliedVol:
    def test_roundtrip(self):
        S, K, T, r = 100.0, 105.0, 0.75, 0.01
        for sig_true in (0.1, 0.25, 0.6):
            px = bs_price(S, K, T, sig_true, r, call=True)
            iv = implied_vol(px, S, K, T, r, call=True)
            assert abs(iv - sig_true) < 1e-5

    def test_put_roundtrip(self):
        px = bs_price(80.0, 90.0, 1.0, 0.35, 0.03, call=False)
        iv = implied_vol(px, 80.0, 90.0, 1.0, 0.03, call=False)
        assert abs(iv - 0.35) < 1e-5

    def test_failclosed_arbitrage(self):
        with pytest.raises(ValueError):
            implied_vol(120.0, 100.0, 100.0, 0.5)  # call > S
        with pytest.raises(ValueError):
            implied_vol(-1.0, 100.0, 100.0, 0.5)


class TestParity:
    def test_gap_zero_for_consistent(self):
        S, K, T, sig, r = 100.0, 95.0, 0.5, 0.2, 0.02
        c = bs_price(S, K, T, sig, r, True)
        p = bs_price(S, K, T, sig, r, False)
        assert abs(put_call_parity_gap(c, p, S, K, T, r)) < 1e-9

    def test_failclosed(self):
        with pytest.raises(ValueError):
            put_call_parity_gap(1.0, np.nan, 100, 100, 0.5)


class TestRNDensity:
    def test_lognormal_chain(self):
        S, T, sig, r = 100.0, 0.5, 0.25, 0.0
        K = np.linspace(70, 140, 30)
        C = np.array([bs_price(S, k, T, sig, r, call=True) for k in K])
        out = risk_neutral_density(K, C, T, r)
        assert np.all(out["density"] >= 0)
        assert out["density"].sum() * (K[1] - K[0]) > 0.5

    def test_failclosed(self):
        with pytest.raises(ValueError):
            risk_neutral_density(np.ones(4), np.ones(4), 0.5)
