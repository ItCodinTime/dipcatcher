"""Cointegration screen: planted-pair detection + honest multiple-testing."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fund.research.pairs.cointegration import (
    EG_CV_5PCT,
    benjamini_hochberg,
    bonferroni,
    correlation_prefilter,
    engle_granger_adf,
    screen_pairs,
)
from quant_fund.research.pairs.fixtures import (
    independent_walks_panel,
    planted_pair_panel,
)


class TestPrefilter:
    def test_keeps_only_correlated_pairs_sorted(self):
        panel = planted_pair_panel(n_assets=8, n_dates=400, seed=3)
        pairs, corrs = correlation_prefilter(panel["log_prices"], min_corr=0.5)
        assert pairs.ndim == 2 and pairs.shape[1] == 2
        # The planted pair shares a factor -> very high |corr of differences|.
        assert {int(pairs[0][0]), int(pairs[0][1])} == {0, 1}
        assert np.all(corrs >= 0.5)
        assert np.all(np.diff(corrs) <= 1e-12)  # sorted descending

    def test_threshold_excludes_independent_walks(self):
        panel = independent_walks_panel(n_assets=4, n_dates=300, seed=11)
        pairs, corrs = correlation_prefilter(panel["log_prices"], min_corr=0.9)
        # Random-walk differences are ~uncorrelated; a 0.9 bar rarely passes.
        assert pairs.shape[0] <= 2
        assert np.all(corrs >= 0.9)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            correlation_prefilter(np.ones((10, 3)), min_corr=0.5)
        with pytest.raises(ValueError):
            correlation_prefilter(np.random.default_rng(0).normal(size=(100, 1)))


class TestEngleGranger:
    def test_stationary_residual_detected(self):
        panel = planted_pair_panel(n_assets=4, n_dates=600, seed=5)
        lp = panel["log_prices"]
        out = engle_granger_adf(lp[:, 0], lp[:, 1])
        assert out.adf_tau < EG_CV_5PCT
        assert out.adf_pvalue < 0.01
        assert out.hedge_ratio == pytest.approx(1.0, abs=0.15)

    def test_random_walk_pair_not_detected(self):
        panel = independent_walks_panel(n_assets=2, n_dates=600, seed=7)
        lp = panel["log_prices"]
        out = engle_granger_adf(lp[:, 0], lp[:, 1])
        # Independent I(1) series should not reject at the strict EG CV.
        assert out.adf_tau > EG_CV_5PCT or out.adf_pvalue > 0.001


class TestMultipleTesting:
    def test_bh_known_answer(self):
        # m=4, sorted p = .001, .01, .02, .5 -> q = .004, .02, .0267, .5
        p = np.array([0.02, 0.5, 0.001, 0.01])
        q = benjamini_hochberg(p)
        np.testing.assert_allclose(
            q[np.argsort(p)],
            [0.004, 0.02, 4 * 0.02 / 3, 0.5],
            rtol=1e-12,
        )

    def test_bh_monotone_and_leq_raw_times_m(self):
        p = np.array([0.9, 0.001, 0.3, 0.05])
        q = benjamini_hochberg(p)
        assert np.all(q >= p)  # adjustment never shrinks a p-value
        assert np.all(q <= 1.0)
        assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)  # monotone in rank

    def test_bonferroni(self):
        np.testing.assert_allclose(bonferroni(np.array([0.01, 0.5])), [0.02, 1.0])

    def test_failclosed(self):
        with pytest.raises(ValueError):
            benjamini_hochberg(np.array([]))
        with pytest.raises(ValueError):
            benjamini_hochberg(np.array([0.5, np.nan]))


class TestScreen:
    def test_planted_pair_ranked_top_and_passes(self):
        panel = planted_pair_panel(n_assets=8, n_dates=600, seed=0)
        frame = screen_pairs(panel["prices"], min_corr=0.5, alpha=0.05)
        assert frame.height >= 1
        top = frame.row(0, named=True)
        assert {top["a"], top["b"]} == {0, 1}
        assert top["passes_bh"] is True
        assert top["passes_eg_cv"] is True
        assert top["adf_tau"] < EG_CV_5PCT
        assert 0.5 < top["half_life"] < 60.0

    def test_sorted_by_adjusted_p(self):
        panel = planted_pair_panel(n_assets=10, n_dates=600, seed=2)
        frame = screen_pairs(panel["prices"], min_corr=0.3, alpha=0.05)
        p_bh = frame["p_bh"].to_numpy()
        assert np.all(np.diff(p_bh) >= -1e-12)

    def test_null_panel_family_control(self):
        # No pair is truly cointegrated: BH should reject few or none.
        panel = independent_walks_panel(n_assets=5, n_dates=600, seed=13)
        frame = screen_pairs(panel["prices"], min_corr=0.3, alpha=0.05)
        if frame.height:
            assert int(frame["passes_bh"].sum()) <= 2

    def test_empty_result_schema(self):
        rng = np.random.default_rng(0)
        logs = np.cumsum(rng.normal(size=(200, 3)), axis=0)
        frame = screen_pairs(np.exp(logs), min_corr=0.999)
        assert frame.height == 0
        assert set(frame.columns) == {
            "a",
            "b",
            "corr_diff",
            "direction",
            "hedge_ratio",
            "intercept",
            "adf_tau",
            "adf_pvalue",
            "adf_lag",
            "half_life",
            "p_bonferroni",
            "p_bh",
            "passes_eg_cv",
            "passes_bh",
        }

    def test_failclosed(self):
        with pytest.raises(ValueError):
            screen_pairs(np.ones((100, 3)))
        with pytest.raises(ValueError):
            screen_pairs(-np.ones((100, 3)))
        with pytest.raises(ValueError):
            screen_pairs(np.abs(np.random.default_rng(0).normal(size=(100, 3))) + 1, alpha=1.5)


def test_frame_is_polars():
    panel = planted_pair_panel(n_assets=4, n_dates=300, seed=1)
    assert isinstance(screen_pairs(panel["prices"]), pl.DataFrame)
