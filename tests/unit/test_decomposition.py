"""Tests for SSA/EMD signal decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.decomposition import (
    emd,
    hilbert_spectrum,
    robust_trend,
    ssa_decompose,
    ssa_forecast,
)


class TestSSA:
    def test_recovers_sine_and_trend(self):
        n = 300
        t = np.arange(n)
        rng = np.random.default_rng(0)
        sine = 2.0 * np.sin(2 * np.pi * t / 25.0)
        trend = 0.01 * t
        v = sine + trend + rng.normal(scale=0.1, size=n)
        dec = ssa_decompose(v, window=60)
        # First two eigentriples carry the sine (paired oscillation).
        comp = dec["components"]
        recon_osc = comp[:, 0] + comp[:, 1]
        assert np.corrcoef(recon_osc[20:-20], sine[20:-20])[0, 1] > 0.9
        # Eigenvalue shares are ordered and positive.
        assert np.all(np.diff(dec["eigvals"]) <= 1e-12)
        assert dec["share"].sum() == pytest.approx(1.0)

    def test_forecast_continues_oscillation(self):
        n = 200
        t = np.arange(n)
        v = 2.0 * np.sin(2 * np.pi * t / 20.0) + 0.005 * t
        out = ssa_forecast(v, window=50, groups=4, steps=10)
        assert out["forecast"].shape == (10,)
        # Forecasts continue the sine + trend pattern.
        future_t = np.arange(n, n + 10)
        expected = 2.0 * np.sin(2 * np.pi * future_t / 20.0) + 0.005 * future_t
        err = np.abs(out["forecast"] - expected)
        assert err.mean() < 1.0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            ssa_decompose(np.ones(20), window=15)
        with pytest.raises(ValueError):
            ssa_forecast(np.ones(10), window=8)


class TestEMD:
    def test_separates_frequencies(self):
        n = 400
        t = np.arange(n)
        fast = np.sin(2 * np.pi * t / 8.0)
        slow = 0.5 * np.sin(2 * np.pi * t / 60.0)
        v = fast + slow
        out = emd(v, max_imfs=4)
        assert out["imfs"].shape[1] >= 2
        imf1 = out["imfs"][:, 0]
        # IMF1 should track the fast component.
        core = slice(20, -20)
        assert np.corrcoef(imf1[core], fast[core])[0, 1] > 0.7
        # Residue + IMFs reconstruct input.
        assert np.allclose(out["imfs"].sum(axis=1) + out["residue"], v)

    def test_monotone_residue_stops(self):
        v = np.linspace(0, 10, 200)
        out = emd(v)
        assert out["n_imf"][0] == 0.0
        assert np.allclose(out["residue"], v)

    def test_hilbert_spectrum(self):
        n = 300
        t = np.arange(n)
        v = 2.0 * np.sin(2 * np.pi * t / 15.0)
        out = hilbert_spectrum(v)
        freq = out["inst_freq"][10:-10]
        assert np.nanmean(freq) == pytest.approx(1.0 / 15.0, rel=0.15)
        assert np.nanmean(out["amplitude"]) == pytest.approx(2.0, rel=0.15)

    def test_robust_trend(self):
        rng = np.random.default_rng(1)
        t = np.arange(200)
        v = 0.05 * t + rng.normal(scale=0.5, size=200)
        out = robust_trend(v, window=21)
        corr = np.corrcoef(out["trend"][10:-10], t[10:-10])[0, 1]
        assert corr > 0.95
        assert np.abs(out["detrended"]).mean() < np.abs(v - v.mean()).mean()

    def test_failclosed(self):
        with pytest.raises(ValueError):
            emd(np.ones(5))
        with pytest.raises(ValueError):
            robust_trend(np.ones(30), window=10)
        with pytest.raises(ValueError):
            hilbert_spectrum(np.linspace(0, 1, 50))
