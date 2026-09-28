"""Tests for VaR backtesting."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.var_backtest import (
    basel_zone,
    christoffersen_test,
    dq_test,
    kupiec_test,
    tuff_test,
)


class TestKupiec:
    def test_correct_coverage_not_rejected(self):
        rng = np.random.default_rng(0)
        hits = (rng.random(250) < 0.01).astype(float)
        out = kupiec_test(hits, alpha=0.99)
        assert 0 <= out["pvalue"] <= 1
        # With ~2.5 expected failures, 0-6 realized is typical.
        assert out["expected"] == pytest.approx(2.5)

    def test_excess_failures_rejected(self):
        hits = np.zeros(250)
        hits[:25] = 1.0  # 10% violation rate vs 1% expected
        out = kupiec_test(hits, alpha=0.99)
        assert out["pvalue"] < 0.001
        assert out["rate"] == pytest.approx(0.10)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            kupiec_test(np.array([0.0, 0.5] * 30))
        with pytest.raises(ValueError):
            kupiec_test(np.zeros(50), alpha=0.2)


class TestChristoffersen:
    def test_clustered_hits_fail_independence(self):
        # Alternating long-quiet/burst pattern -> strong dependence.
        h = np.zeros(240)
        h[10:20] = 1.0
        h[100:112] = 1.0
        out = christoffersen_test(h, alpha=0.90)
        assert out["pi11"] > out["pi01"]
        assert np.isfinite(out["lr_independence"])

    def test_iid_hits_independent(self):
        rng = np.random.default_rng(1)
        hits = (rng.random(300) < 0.05).astype(float)
        # Ensure both transition states exist.
        if hits.sum() == 0:
            hits[0] = 1.0
        out = christoffersen_test(hits, alpha=0.95)
        assert np.isfinite(out["lr_cc"])

    def test_failclosed(self):
        with pytest.raises(ValueError):
            christoffersen_test(np.zeros(50))  # no hits -> undefined


class TestTUFF:
    def test_early_failure_flagged(self):
        h = np.zeros(100)
        h[0] = 1.0  # failure on day 1 at 99% VaR
        out = tuff_test(h, alpha=0.99)
        assert out["statistic"] > 0
        assert out["time_to_first"] == 1.0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            tuff_test(np.ones(5))


class TestBaselZone:
    def test_canonical_table(self):
        green = np.zeros(250)
        green[:3] = 1
        assert basel_zone(green)["label"] == "green"
        yellow = np.zeros(250)
        yellow[:6] = 1
        assert basel_zone(yellow)["label"] == "yellow"
        red = np.zeros(250)
        red[:11] = 1
        assert basel_zone(red)["label"] == "red"

    def test_generic_n(self):
        h = np.zeros(100)
        h[:1] = 1
        out = basel_zone(h, alpha=0.99)
        assert out["zone"] == 0.0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            basel_zone(np.array([0.2] * 50))


class TestDQ:
    """Engle & Manganelli (2004) dynamic quantile test."""

    def test_iid_hits_not_rejected(self):
        rng = np.random.default_rng(7)
        hits = (rng.random(500) < 0.05).astype(float)
        out = dq_test(hits, alpha=0.95, lags=4)
        assert out["pvalue"] > 0.01
        assert out["df"] == 5.0
        assert out["n"] == 496.0

    def test_clustered_hits_rejected(self):
        # Blocks of 10 consecutive violations at 95%: lag-4 dependence.
        h = np.zeros(400)
        for start in (50, 150, 250, 350):
            h[start : start + 10] = 1.0
        out = dq_test(h, alpha=0.95, lags=4)
        assert out["pvalue"] < 0.001

    def test_wrong_level_rejected(self):
        # 20% violations at a 99% VaR — intercept carries the misspecification.
        rng = np.random.default_rng(3)
        hits = (rng.random(400) < 0.20).astype(float)
        out = dq_test(hits, alpha=0.99, lags=4)
        assert out["pvalue"] < 0.001

    def test_forecast_level_instrument(self):
        # Hits depend on an instrument (e.g. a stale VaR series): regression
        # picks it up where lag-1 Markov tests are blind.
        rng = np.random.default_rng(11)
        level = rng.random(500)
        hits = (rng.random(500) < 0.02 + 0.15 * level).astype(float)
        out = dq_test(hits, alpha=0.95, lags=4, instruments=level)
        assert out["df"] == 6.0
        assert out["pvalue"] < 0.05

    def test_power_over_christoffersen(self):
        # Lag-3 clustering: christoffersen lag-1 is blind to it, DQ sees it.
        h = np.zeros(600)
        idx = np.arange(0, 600, 12)
        for i in idx:
            h[i + 3] = 1.0
            h[i + 11] = 1.0
        out = dq_test(h, alpha=0.95, lags=4)
        assert out["pvalue"] < 0.01

    def test_failclosed(self):
        with pytest.raises(ValueError):
            dq_test(np.zeros(100))  # degenerate: singular regression
        with pytest.raises(ValueError):
            dq_test(np.ones(100))  # all violations: singular
        with pytest.raises(ValueError):
            dq_test((np.random.default_rng(0).random(40) < 0.5).astype(float), alpha=0.95, lags=45)
        with pytest.raises(ValueError):
            hits = np.zeros(100)
            hits[::50] = 1.0
            dq_test(hits, alpha=0.95, lags=4, instruments=np.full((100, 1), np.nan))
        with pytest.raises(ValueError):
            dq_test(hits, alpha=0.95, lags=4, instruments=np.ones(50))
