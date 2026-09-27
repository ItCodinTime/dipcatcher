"""Tests for drawdown-based risk measures."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.drawdown import (
    burke_ratio,
    conditional_drawdown_at_risk,
    drawdown_at_risk,
    expected_max_drawdown,
    kappa_ratio,
    omega_ratio,
    pain_index,
    tail_ratio,
    ulcer_index,
)


def _crash_series() -> np.ndarray:
    r = np.full(300, 0.001)
    r[100] = -0.15
    r[150] = -0.08
    r[250] = -0.20
    return r


class TestDrawdownRisk:
    def test_dar_bounds(self):
        r = _crash_series()
        dar = drawdown_at_risk(r, 0.95)
        dd = np.abs(np.cumprod(1 + r) / np.maximum.accumulate(np.cumprod(1 + r)) - 1)
        assert 0.0 < dar <= dd.max() + 1e-9
        assert dar >= np.quantile(dd, 0.90)

    def test_cdar_exceeds_dar(self):
        r = _crash_series()
        out = conditional_drawdown_at_risk(r, 0.90)
        assert out["cdar"] >= out["dar"] > 0.0

    def test_omega(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.02, 1000)
        om = omega_ratio(r, 0.0)
        assert om > 0.5  # roughly symmetric-ish with slight positive drift
        rng2 = np.random.default_rng(1)
        r_bad = rng2.normal(-0.002, 0.02, 1000)
        assert omega_ratio(r_bad, 0.0) < om
        # All-positive returns -> undefined.
        with pytest.raises(ValueError):
            omega_ratio(np.full(100, 0.01))

    def test_kappa(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.002, 0.02, 1000)
        k3 = kappa_ratio(r, order=3)
        assert np.isfinite(k3)
        # Positive mean -> positive Kappa.
        assert k3 > 0.0

    def test_ulcer_and_pain(self):
        calm = np.full(300, 0.001)
        crash = _crash_series()
        assert ulcer_index(crash) > ulcer_index(calm)
        assert pain_index(crash) > pain_index(calm)
        assert ulcer_index(crash) >= pain_index(crash)

    def test_burke_and_tail(self):
        r = _crash_series()
        b = burke_ratio(r)
        assert np.isfinite(b)
        rng = np.random.default_rng(3)
        # Right-skewed: tail ratio > 1.
        r_skew = np.concatenate([rng.normal(0.005, 0.005, 900), rng.normal(-0.05, 0.01, 100)])
        tr = tail_ratio(r_skew)
        assert np.isfinite(tr)
        with pytest.raises(ValueError):
            tail_ratio(np.full(100, 0.01))  # no lower tail

    def test_expected_max_dd(self):
        out = expected_max_drawdown(mu=0.0005, sigma=0.02, horizon=252, n_paths=500)
        assert 0.0 < out["q50"] <= out["q95"]
        assert out["q95"] < 1.0
        # Higher vol -> deeper expected drawdown.
        out2 = expected_max_drawdown(mu=0.0005, sigma=0.05, horizon=252, n_paths=500)
        assert out2["mean"] > out["mean"]

    def test_failclosed(self):
        with pytest.raises(ValueError):
            drawdown_at_risk(np.arange(5.0) * 0.001)
        with pytest.raises(ValueError):
            conditional_drawdown_at_risk(np.random.default_rng(0).normal(size=100), alpha=1.2)
        with pytest.raises(ValueError):
            kappa_ratio(np.random.default_rng(0).normal(size=100), order=0)
        with pytest.raises(ValueError):
            ulcer_index(np.arange(3.0) * 0.01)
        with pytest.raises(ValueError):
            expected_max_drawdown(0.0, -0.1, 100)
        with pytest.raises(ValueError):
            expected_max_drawdown(0.0, 0.02, 100, n_paths=10)
