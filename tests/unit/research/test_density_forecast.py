"""Tests for density-forecast evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as st

from quant_fund.metrics.density_forecast import (
    berkowitz_test,
    pit_autocorrelation,
    pit_histogram,
)


class TestPITHistogram:
    def test_uniform_pits_accepted(self):
        rng = np.random.default_rng(0)
        pits = rng.uniform(0.001, 0.999, 500)
        out = pit_histogram(pits, bins=10)
        assert out["pvalue"] > 0.05
        assert out["shares"].shape == (10,)

    def test_nonuniform_detected(self):
        # Miscalibrated: center-humped PITs (overdispersed density) —
        # Beta(2,2) concentrates mass near 0.5.
        rng = np.random.default_rng(1)
        pits = np.clip(st.beta.rvs(2, 2, size=500, random_state=rng), 1e-6, 1 - 1e-6)
        out = pit_histogram(pits, bins=10)
        assert out["pvalue"] < 0.05

    def test_failclosed(self):
        with pytest.raises(ValueError):
            pit_histogram(np.zeros(100))  # boundaries invalid
        with pytest.raises(ValueError):
            pit_histogram(np.random.default_rng(0).random(30))


class TestBerkowitz:
    def test_correct_density_not_rejected(self):
        rng = np.random.default_rng(2)
        pits = rng.uniform(0.001, 0.999, 400)
        out = berkowitz_test(pits)
        assert out["pvalue"] > 0.01
        assert abs(out["mu_hat"]) < 0.3

    def test_biased_density_rejected(self):
        rng = np.random.default_rng(3)
        # PIT skewed right -> z has positive mean.
        pits = np.clip(st.beta.rvs(1.8, 1.0, size=400, random_state=rng), 1e-6, 1 - 1e-6)
        out = berkowitz_test(pits)
        assert out["pvalue"] < 0.05
        assert out["mu_hat"] > 0.1

    def test_censored_runs(self):
        rng = np.random.default_rng(4)
        pits = rng.uniform(0.001, 0.999, 300)
        out = berkowitz_test(pits, tail_censor=0.05)
        assert np.isfinite(out["statistic"])

    def test_failclosed(self):
        with pytest.raises(ValueError):
            berkowitz_test(np.full(100, 1.5))


class TestPITAutocorr:
    def test_iid_pits_pass(self):
        rng = np.random.default_rng(5)
        out = pit_autocorrelation(rng.uniform(0.001, 0.999, 400), lags=5)
        assert out["pvalue"] > 0.01

    def test_dependent_pits_detected(self):
        rng = np.random.default_rng(6)
        n = 500
        z = np.empty(n)
        z[0] = 0.0
        for t in range(1, n):
            z[t] = 0.7 * z[t - 1] + rng.normal()
        pits = np.clip(st.norm.cdf(z), 1e-6, 1 - 1e-6)
        out = pit_autocorrelation(pits, lags=5)
        assert out["pvalue"] < 0.01

    def test_failclosed(self):
        with pytest.raises(ValueError):
            pit_autocorrelation(np.ones(100))
