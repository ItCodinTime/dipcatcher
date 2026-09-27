"""Tests for rough-volatility estimators."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.rough_vol import (
    logvol_hurst,
    rough_signature,
    simulate_fou,
    variance_curve_fit,
)


class TestLogVolHurst:
    def test_rough_process_low_H(self):
        x = simulate_fou(n=1500, H=0.12, seed=0)
        out = logvol_hurst(x)
        assert out["H"] < 0.35  # rough path -> H well below 0.5

    def test_smooth_process_high_H(self):
        # OU-like smooth log-vol -> H ~ 0.5.
        rng = np.random.default_rng(1)
        n = 1200
        x = np.empty(n)
        x[0] = 0.0
        for t in range(1, n):
            x[t] = 0.98 * x[t - 1] + rng.normal(scale=0.05)
        out = logvol_hurst(x)
        assert out["H"] > 0.3

    def test_failclosed(self):
        with pytest.raises(ValueError):
            logvol_hurst(np.ones(50))


class TestVarianceCurve:
    def test_recovers_H(self):
        x = simulate_fou(n=2000, H=0.15, nu=0.4, seed=2)
        out = variance_curve_fit(x)
        assert abs(out["H"] - 0.15) < 0.2
        assert out["nu"] > 0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            variance_curve_fit(np.ones(30))


class TestSimulateFOU:
    def test_stationary_output(self):
        x = simulate_fou(n=500, H=0.2, nu=1.0, theta=-1.0, seed=3)
        assert x.shape == (500,)
        assert np.all(np.isfinite(x))
        assert abs(x.mean() + 1.0) < 0.5

    def test_failclosed(self):
        with pytest.raises(ValueError):
            simulate_fou(500, H=0.7)


class TestSignature:
    def test_rough_vs_smooth(self):
        rough = simulate_fou(n=1200, H=0.1, seed=4)
        h_rough = rough_signature(rough)["H"]
        rng = np.random.default_rng(5)
        smooth = np.cumsum(rng.normal(scale=0.02, size=1200))
        h_smooth = rough_signature(smooth)["H"]
        assert h_rough < h_smooth

    def test_failclosed(self):
        with pytest.raises(ValueError):
            rough_signature(np.ones(80))
