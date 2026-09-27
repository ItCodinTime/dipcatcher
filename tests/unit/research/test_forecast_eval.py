"""Tests for forecast-evaluation tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.forecast_eval import (
    clark_west_test,
    encompassing_test,
    fluctuation_test,
    giacomini_white_test,
    hln_test,
)


class TestHLN:
    def test_detects_better_forecast(self):
        rng = np.random.default_rng(0)
        n = 200
        y = rng.normal(size=n)
        e_bad = y + 1.0  # biased forecast
        e_good = y - rng.normal(scale=0.2, size=n)
        out = hln_test(e_bad, e_good)
        assert out["pvalue"] < 0.05
        assert out["statistic"] > 0

    def test_no_power_when_equal(self):
        # Sign-flipped errors have identical squared loss — exactly equal
        # predictive ability must never reject.
        rng = np.random.default_rng(1)
        e1 = rng.normal(size=300)
        out = hln_test(e1, -e1)
        assert out["pvalue"] > 0.05

    def test_failclosed(self):
        with pytest.raises(ValueError):
            hln_test(np.ones(10), np.ones(10))


class TestClarkWest:
    def test_nested_signal_detected(self):
        rng = np.random.default_rng(2)
        n = 250
        y = rng.normal(size=n)
        f_null = np.zeros(n)  # benchmark: unconditional mean
        f_alt = y * 0.3  # correlated forecast
        e_null = y - f_null
        e_alt = y - f_alt
        out = clark_west_test(e_alt, e_null, f_null - f_alt)
        assert out["mspe_adj"] > 0  # adjusted series favors larger model
        assert out["pvalue"] < 0.05

    def test_failclosed(self):
        with pytest.raises(ValueError):
            clark_west_test(np.ones(10), np.ones(10), np.ones(9))


class TestGiacominiWhite:
    def test_detects_ability_difference(self):
        rng = np.random.default_rng(3)
        n = 200
        l1 = np.abs(rng.normal(scale=1.5, size=n))
        l2 = np.abs(rng.normal(scale=0.8, size=n))
        out = giacomini_white_test(l1, l2)
        assert out["pvalue"] < 0.05

    def test_equal_losses_not_rejected(self):
        rng = np.random.default_rng(4)
        base = np.abs(rng.normal(size=250))
        out = giacomini_white_test(
            base + rng.normal(scale=0.05, size=250), base + rng.normal(scale=0.05, size=250)
        )
        # Mild — should generally not reject at 1%.
        assert out["pvalue"] > 0.01

    def test_failclosed(self):
        with pytest.raises(ValueError):
            giacomini_white_test(np.ones(5), np.ones(5))


class TestEncompassing:
    def test_rival_adds_info(self):
        rng = np.random.default_rng(5)
        n = 250
        y = rng.normal(size=n)
        e_small = y  # forecast = 0
        e_large = y - y * 0.5  # forecast = 0.5y — much better
        out = encompassing_test(e_small, e_large)
        assert out["statistic"] > 0
        assert out["pvalue"] < 0.05

    def test_failclosed(self):
        with pytest.raises(ValueError):
            encompassing_test(np.ones(10), np.ones(11))


class TestFluctuation:
    def test_detects_skill_shift(self):
        rng = np.random.default_rng(6)
        n = 300
        y = rng.normal(size=n)
        e1 = np.abs(y + rng.normal(scale=0.3, size=n))
        e2 = np.abs(y + rng.normal(scale=0.3, size=n))
        # Mid-sample, e2 becomes much worse.
        e2[150:] += 2.0
        out = fluctuation_test(e1, e2, window=40)
        assert out["reject"] == 1.0
        # Break should be detected near the shift.
        assert out["argmax"] > 80

    def test_stable_skill_not_flagged(self):
        rng = np.random.default_rng(7)
        n = 200
        e1 = np.abs(rng.normal(size=n))
        e2 = np.abs(rng.normal(size=n))
        out = fluctuation_test(e1, e2, window=50)
        assert np.isfinite(out["sup"])

    def test_failclosed(self):
        with pytest.raises(ValueError):
            fluctuation_test(np.ones(50), np.ones(50), window=200)
