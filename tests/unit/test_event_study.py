"""Tests for event-study methodology."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.event_study import (
    abnormal_returns,
    bmp_test,
    corrado_rank_test,
    cumulative_abnormal,
    event_study,
    market_model_fit,
    patell_test,
)


class TestMarketModel:
    def test_fit_recovers_beta(self):
        rng = np.random.default_rng(0)
        rm = rng.normal(size=300)
        ri = 0.0005 + 1.3 * rm + rng.normal(scale=0.02, size=300)
        fit = market_model_fit(ri, rm)
        assert abs(fit["beta"] - 1.3) < 0.05
        assert abs(fit["alpha"] - 0.0005) < 0.005

    def test_abnormal_and_car(self):
        rng = np.random.default_rng(1)
        rm = rng.normal(size=200)
        ri = 0.5 * rm + rng.normal(scale=0.02, size=200)
        fit = market_model_fit(ri, rm)
        # Event window with a true +5% abnormal shock on day 1.
        ev_m = rng.normal(size=5)
        ev_r = 0.5 * ev_m + rng.normal(scale=0.02, size=5)
        ev_r[1] += 0.05
        ars = abnormal_returns(ev_r, ev_m, fit)
        assert abs(ars[1] - 0.05) < 0.03
        assert abs(cumulative_abnormal(ars) - 0.05) < 0.08

    def test_failclosed(self):
        with pytest.raises(ValueError):
            market_model_fit(np.ones(10), np.ones(10))


class TestPatell:
    def test_positive_sars_detected(self):
        rng = np.random.default_rng(2)
        sar = rng.normal(size=(40, 5)) + 0.8  # positive abnormal
        out = patell_test(sar, np.full(40, 120.0))
        assert out["statistic"] > 0
        assert out["pvalue"] < 0.01

    def test_null(self):
        rng = np.random.default_rng(3)
        sar = rng.normal(size=(50, 3))
        out = patell_test(sar, np.full(50, 150.0))
        # Null may or may not reject; statistic should be finite.
        assert np.isfinite(out["statistic"])

    def test_failclosed(self):
        with pytest.raises(ValueError):
            patell_test(np.ones((1, 5)), np.ones(1))


class TestCorrado:
    def test_event_day_rank_shift(self):
        rng = np.random.default_rng(4)
        n_events, K = 30, 21
        R = np.tile(np.arange(1, K + 1), (n_events, 1)) + rng.integers(-2, 3, (n_events, K))
        # Event day (index 10) ranks pushed high.
        R[:, 10] = K + rng.integers(0, 5, n_events)
        out = corrado_rank_test(R.astype(float))
        assert np.isfinite(out["statistic"])

    def test_failclosed(self):
        with pytest.raises(ValueError):
            corrado_rank_test(np.ones((1, 10)))


class TestBMP:
    def test_positive_car_detected(self):
        rng = np.random.default_rng(5)
        sar = rng.normal(size=(30, 3)) + 1.0
        out = bmp_test(sar)
        assert out["mean_car"] > 0
        assert out["pvalue"] < 0.01

    def test_failclosed(self):
        with pytest.raises(ValueError):
            bmp_test(np.ones((2, 3)))


class TestEventStudyWrapper:
    def test_detects_abnormal_event(self):
        rng = np.random.default_rng(6)
        rm = rng.normal(size=250)
        ri = 0.8 * rm + rng.normal(scale=0.015, size=250)
        ev_m = np.array([0.01, -0.005, 0.002])
        ev_r = 0.8 * ev_m + np.array([0.06, 0.0, 0.0])  # +6% day-0 shock
        out = event_study(ev_r, ev_m, ri, rm)
        assert out["car"] > 0.04
        assert out["pvalue"] < 0.05
