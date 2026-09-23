"""Tests for calibration and scoring extensions."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.calibration2 import (
    dawid_sebastiani,
    hosmer_lemeshow,
    murphy_decomposition,
    pinball_score,
    reliability_diagram,
    spread_skill,
    variogram_score,
    winkler_interval_score,
)


class TestMurphy:
    def test_decomposition_exact(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 500)
        y = (rng.random(500) < p).astype(float)
        out = murphy_decomposition(p, y, n_bins=10)
        # Exact only for bin-constant forecasts; continuous probs leave a
        # small within-bin dispersion residual.
        assert abs(out["decomp_error"]) < 0.02
        assert out["reliability"] >= 0 and out["resolution"] >= 0

    def test_calibrated_better_than_flat(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0.1, 0.9, 400)
        y = (rng.random(400) < p).astype(float)
        good = murphy_decomposition(p, y)["brier"]
        flat = murphy_decomposition(np.full(400, y.mean()), y)["brier"]
        assert good < flat

    def test_failclosed(self):
        with pytest.raises(ValueError):
            murphy_decomposition(np.array([0.5, 1.5, 0.2] * 10), np.array([0, 1, 0] * 10))
        with pytest.raises(ValueError):
            murphy_decomposition(np.full(20, 0.5), np.arange(20) * 0.1)


class TestIntervalAndDS:
    def test_winkler_rewards_coverage(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=300)
        lo, hi = -1.64 * np.ones(300), 1.64 * np.ones(300)
        wide = winkler_interval_score(lo, hi, y, alpha=0.1)
        narrow = winkler_interval_score(-0.5 * np.ones(300), 0.5 * np.ones(300), y, alpha=0.1)
        # Wide interval covers but is penalized by width; narrow misses more.
        assert wide["coverage"][0] > narrow["coverage"][0]
        assert wide["mean_width"][0] > narrow["mean_width"][0]
        # Well-calibrated interval beats missing one.
        assert wide["mean_score"][0] < narrow["mean_score"][0]

    def test_winkler_failclosed(self):
        with pytest.raises(ValueError):
            winkler_interval_score(np.ones(10), np.zeros(10), np.ones(10))
        with pytest.raises(ValueError):
            winkler_interval_score(np.zeros(10), np.ones(10), np.ones(10), alpha=1.5)

    def test_dawid_sebastiani(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=200)
        good = dawid_sebastiani(np.zeros(200), np.ones(200), y)["mean_score"][0]
        bad = dawid_sebastiani(np.zeros(200), np.full(200, 9.0), y)["mean_score"][0]
        assert good < bad


class TestHosmer:
    def test_calibrated_passes(self):
        rng = np.random.default_rng(4)
        p = rng.uniform(0.05, 0.95, 1000)
        y = (rng.random(1000) < p).astype(float)
        assert hosmer_lemeshow(p, y, n_groups=10)["pvalue"] > 0.01

    def test_miscalibrated_fails(self):
        rng = np.random.default_rng(5)
        p = rng.uniform(0.2, 0.4, 1000)
        y = (rng.random(1000) < 0.8).astype(float)  # truth much higher
        assert hosmer_lemeshow(p, y, n_groups=10)["pvalue"] < 0.01

    def test_reliability_diagram(self):
        rng = np.random.default_rng(6)
        p = rng.uniform(0.05, 0.95, 500)
        y = (rng.random(500) < p).astype(float)
        out = reliability_diagram(p, y, n_bins=10)
        mask = np.isfinite(out["obs_freq"])
        # Calibrated: obs_freq ~ bin_center.
        err = np.abs(out["obs_freq"][mask] - out["bin_center"][mask])
        assert err.mean() < 0.15
        assert out["count"].sum() == 500


class TestMultivariate:
    def test_variogram_prefers_good_ensemble(self):
        rng = np.random.default_rng(7)
        d, m = 6, 60
        y = rng.normal(size=d)
        good_ens = y[None, :] + rng.normal(scale=0.3, size=(m, d))
        bad_ens = rng.normal(scale=5.0, size=(m, d))
        vs_good = variogram_score(good_ens, y)
        vs_bad = variogram_score(bad_ens, y)
        assert vs_good < vs_bad

    def test_pinball(self):
        rng = np.random.default_rng(8)
        y = rng.normal(size=400)
        q = np.column_stack([np.full(400, -1.0), np.zeros(400), np.full(400, 1.0)])
        out = pinball_score(q, y, np.array([0.1, 0.5, 0.9]))
        assert out["per_quantile"].shape == (3,)
        # Pinball magnitudes differ per tau — the meaningful check is that
        # the true quantile beats a misspecified one at the same tau.
        q_good = np.column_stack([np.full(400, -1.2816), np.zeros(400), np.full(400, 1.2816)])
        good = pinball_score(q_good, y, np.array([0.1, 0.5, 0.9]))["mean"][0]
        bad = pinball_score(np.zeros((400, 3)), y, np.array([0.1, 0.5, 0.9]))["mean"][0]
        assert good < bad

    def test_spread_skill(self):
        rng = np.random.default_rng(9)
        n = 800
        sd = np.abs(rng.normal(size=n)) + 0.1
        y = rng.normal(scale=sd, size=n)
        out = spread_skill(np.zeros(n), sd, y, n_bins=5)
        assert np.isfinite(out["slope"][0])
        # RMSE should roughly track spread -> slope ~ 1.
        assert 0.5 < out["slope"][0] < 1.6
