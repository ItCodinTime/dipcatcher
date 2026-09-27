"""Tests for panel unit-root and dependence tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.panel import (
    fisher_adf,
    hadri_test,
    im_pesaran_shin,
    levin_lin_chu,
    pesaran_cd,
)


def _panel_ar1(n=8, t=120, rho=0.9, seed=0):
    rng = np.random.default_rng(seed)
    p = np.empty((t, n))
    p[0] = rng.normal(size=n)
    for s in range(1, t):
        p[s] = rho * p[s - 1] + rng.normal(size=n)
    return p


def _panel_rw(n=8, t=120, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(t, n)), axis=0)


class TestLLC:
    def test_stationary_panel_rejects(self):
        p = _panel_ar1(rho=0.6)
        out = levin_lin_chu(p, lags=1)
        assert out["statistic"] < 0  # negative rho -> reject unit root

    def test_failclosed(self):
        with pytest.raises(ValueError):
            levin_lin_chu(np.ones((10, 2)))


class TestIPS:
    def test_stationary_vs_rw(self):
        stat_p = im_pesaran_shin(_panel_ar1(rho=0.7))["statistic"]
        rw_p = im_pesaran_shin(_panel_rw())["statistic"]
        assert stat_p < rw_p  # stationary panel -> more negative tbar

    def test_failclosed(self):
        with pytest.raises(ValueError):
            im_pesaran_shin(np.ones((20, 2)))


class TestFisherADF:
    def test_stationary_rejects_more(self):
        s = fisher_adf(_panel_ar1(rho=0.5))["statistic"]
        r = fisher_adf(_panel_rw())["statistic"]
        assert s > r  # Fisher chi2 larger for stationary panels


class TestHadri:
    def test_stationary_not_rejected(self):
        rng = np.random.default_rng(3)
        p = rng.normal(size=(150, 6))  # iid stationary
        out = hadri_test(p)
        assert out["pvalue"] > 0.01

    def test_rw_rejected(self):
        p = _panel_rw(n=6, t=150)
        out = hadri_test(p)
        assert out["pvalue"] < 0.05

    def test_failclosed(self):
        with pytest.raises(ValueError):
            hadri_test(np.ones((10, 6)))


class TestPesaranCD:
    def test_independent_panel(self):
        rng = np.random.default_rng(4)
        p = rng.normal(size=(200, 10))
        out = pesaran_cd(p)
        assert out["pvalue"] > 0.05
        assert out["mean_abs_corr"] < 0.15

    def test_common_factor_detected(self):
        rng = np.random.default_rng(5)
        f = rng.normal(size=200)
        p = np.column_stack(
            [f * rng.uniform(0.5, 1.5) + rng.normal(scale=0.3, size=200) for _ in range(8)]
        )
        out = pesaran_cd(p)
        assert out["pvalue"] < 0.01

    def test_failclosed(self):
        with pytest.raises(ValueError):
            pesaran_cd(np.ones((200, 10)))  # constant -> degenerate
