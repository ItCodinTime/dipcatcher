"""Tests for VaR backtesting."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.var_backtest import (
    basel_zone,
    christoffersen_test,
    kratz_test,
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


class TestKratz:
    ALPHAS = np.array([0.95, 0.975, 0.99, 0.999])

    def _var(self, n: int, scale: float = 1.0) -> np.ndarray:
        from scipy import stats

        return np.stack([np.full(n, scale * stats.norm.ppf(a)) for a in self.ALPHAS], axis=1)

    def test_correct_spec_not_rejected(self):
        rng = np.random.default_rng(0)
        r = rng.standard_normal(2000)
        out = kratz_test(r, self._var(2000), self.ALPHAS)
        assert out["df"] == 4.0
        assert out["pvalue"] > 0.01
        assert out["counts"].sum() == 2000.0

    def test_perfect_counts_statistic_zero(self):
        # Crafted exact multinomial counts: 950/40/10 vs expected 950/40/10.
        r = np.concatenate([np.full(950, 0.0), np.full(40, 0.7), np.full(10, 2.0)])
        v = np.stack([np.full(1000, 0.5), np.full(1000, 1.0)], axis=1)
        out = kratz_test(r, v, np.array([0.95, 0.99]))
        assert abs(out["statistic"]) < 1e-12
        assert out["pvalue"] == 1.0

    def test_underestimated_var_rejected(self):
        rng = np.random.default_rng(0)
        r = rng.standard_normal(2000)
        out = kratz_test(r, self._var(2000, scale=0.8), self.ALPHAS)
        assert out["pvalue"] < 0.001

    def test_tail_shape_miss_rejected(self):
        # Student-t(4) is matched at alpha=0.95 by the normal VaR but has
        # a much fatter deeper tail — the multinomial sees the band pileup
        # a single-level Kupiec misses.
        rng = np.random.default_rng(0)
        r = rng.standard_t(4, 2000) / np.sqrt(2.0)
        out = kratz_test(r, self._var(2000), self.ALPHAS)
        assert out["pvalue"] < 0.001

    def test_failclosed(self):
        n = 200
        v = self._var(n)
        r = np.zeros(n)
        with pytest.raises(ValueError):
            kratz_test(np.zeros(20), self._var(20), self.ALPHAS)  # too short
        with pytest.raises(ValueError):
            kratz_test(r, v[:, 0], self.ALPHAS)  # 1-D var
        with pytest.raises(ValueError):
            kratz_test(r, v, self.ALPHAS[:2])  # mismatched alphas
        with pytest.raises(ValueError):
            kratz_test(r, v[:, ::-1], self.ALPHAS)  # non-monotone VaR
        with pytest.raises(ValueError):
            kratz_test(r, v, np.array([0.9, 0.8]))  # alphas not increasing
        rr = r.copy()
        rr[5] = np.nan
        with pytest.raises(ValueError):
            kratz_test(rr, v, self.ALPHAS)
