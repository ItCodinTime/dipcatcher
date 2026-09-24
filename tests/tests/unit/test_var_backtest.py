"""Tests for VaR backtesting."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.var_backtest import (
    basel_zone,
    christoffersen_test,
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
