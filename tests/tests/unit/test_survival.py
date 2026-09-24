"""Tests for survival analysis."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.survival import (
    fit_cox_ph,
    kaplan_meier,
    log_rank_test,
    nelson_aalen,
)


class TestKaplanMeier:
    def test_exponential_decay(self):
        rng = np.random.default_rng(0)
        t = rng.exponential(scale=10.0, size=300)
        d = np.ones(300)
        km = kaplan_meier(t, d)
        assert np.all(np.diff(km["survival"]) <= 1e-9)  # non-increasing
        assert 0 <= km["survival"][-1] < km["survival"][0] <= 1
        assert np.all(km["se"] >= 0)
        # Median survival ~ 10 ln 2 ~ 6.9.
        med = km["times"][np.argmax(km["survival"] < 0.5)]
        assert 4.0 < med < 11.0

    def test_censoring(self):
        rng = np.random.default_rng(1)
        t = rng.exponential(scale=8.0, size=200)
        d = (rng.random(200) < 0.6).astype(float)
        km = kaplan_meier(t, d)
        assert np.all(np.isfinite(km["survival"]))

    def test_failclosed(self):
        with pytest.raises(ValueError):
            kaplan_meier(np.ones(20), np.zeros(20))  # no events
        with pytest.raises(ValueError):
            kaplan_meier(-np.ones(20), np.ones(20))


class TestNelsonAalen:
    def test_cumulative_hazard_increases(self):
        rng = np.random.default_rng(2)
        t = rng.exponential(scale=5.0, size=250)
        na = nelson_aalen(t, np.ones(250))
        assert np.all(np.diff(na["hazard"]) >= 0)
        # For exponential with rate 1/5, H(t) ~ t/5 — terminal H ~ E[t]/5 ~ 1.
        assert na["hazard"][-1] > 0.5

    def test_failclosed(self):
        with pytest.raises(ValueError):
            nelson_aalen(np.ones(3), np.ones(3))


class TestLogRank:
    def test_different_survival_detected(self):
        rng = np.random.default_rng(3)
        t1 = rng.exponential(scale=5.0, size=150)
        t2 = rng.exponential(scale=15.0, size=150)
        out = log_rank_test(t1, np.ones(150), t2, np.ones(150))
        assert out["pvalue"] < 0.01
        assert out["observed"] > out["expected"]  # group 1 dies faster

    def test_same_survival_not_rejected(self):
        rng = np.random.default_rng(4)
        t1 = rng.exponential(scale=10.0, size=150)
        t2 = rng.exponential(scale=10.0, size=150)
        out = log_rank_test(t1, np.ones(150), t2, np.ones(150))
        assert out["pvalue"] > 0.05

    def test_failclosed(self):
        with pytest.raises(ValueError):
            log_rank_test(np.ones(5), np.ones(5), np.ones(5), np.ones(5))


class TestCoxPH:
    def test_recovers_hazard_ratio(self):
        rng = np.random.default_rng(5)
        n = 400
        x = rng.normal(size=(n, 1))
        beta_true = 0.8
        lam = 0.05
        # Simulate exponential durations with hazard lam * exp(beta x).
        u = rng.random(n)
        t = -np.log(u) / (lam * np.exp(x[:, 0] * beta_true))
        d = np.ones(n)
        fit = fit_cox_ph(x, t, d)
        assert abs(fit["beta"][0] - beta_true) < 0.25
        assert fit["hazard_ratio"][0] > 1.5
        assert fit["concordance"][0] > 0.6

    def test_failclosed(self):
        with pytest.raises(ValueError):
            fit_cox_ph(np.ones((50, 1)), np.ones(50), np.ones(50))  # constant X
