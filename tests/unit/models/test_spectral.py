"""Tests for spectral estimation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.spectral import (
    band_power,
    coherence,
    dominant_frequency,
    periodogram_daniell,
    periodogram_multitaper,
    periodogram_welch,
)


def _sine_plus_noise(n=512, freq=0.08, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * freq * t) + rng.normal(scale=noise, size=n)


class TestDaniell:
    def test_peak_at_true_freq(self):
        v = _sine_plus_noise()
        out = periodogram_daniell(v, m=4)
        f = dominant_frequency(out["freqs"], out["density"], lo=0.01)
        assert abs(f - 0.08) < 0.02

    def test_failclosed(self):
        with pytest.raises(ValueError):
            periodogram_daniell(np.ones(64))
        with pytest.raises(ValueError):
            periodogram_daniell(np.ones(100), m=20)


class TestWelch:
    def test_peak(self):
        v = _sine_plus_noise(seed=1)
        out = periodogram_welch(v, seg_len=128)
        f = dominant_frequency(out["freqs"], out["density"], lo=0.01)
        assert abs(f - 0.08) < 0.02
        assert out["n_seg"] >= 3

    def test_white_flat(self):
        rng = np.random.default_rng(2)
        out = periodogram_welch(rng.normal(size=512), seg_len=128)
        d = out["density"]
        # White noise -> roughly flat spectrum.
        assert d.max() / max(d.min(), 1e-12) < 20

    def test_failclosed(self):
        with pytest.raises(ValueError):
            periodogram_welch(np.ones(40), seg_len=64)


class TestMultitaper:
    def test_peak(self):
        v = _sine_plus_noise(n=400, freq=0.12, seed=3)
        out = periodogram_multitaper(v, nw=3.0)
        f = dominant_frequency(out["freqs"], out["density"], lo=0.02)
        assert abs(f - 0.12) < 0.02
        assert out["n_tapers"] == 5.0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            periodogram_multitaper(np.ones(100), nw=0.2)


class TestCoherence:
    def test_shared_component_high(self):
        rng = np.random.default_rng(4)
        n = 512
        t = np.arange(n)
        common = np.sin(2 * np.pi * 0.1 * t)
        x = common + rng.normal(scale=0.2, size=n)
        y = common + rng.normal(scale=0.2, size=n)
        out = coherence(x, y, seg_len=128)
        idx = np.argmin(np.abs(out["freqs"] - 0.1))
        assert out["coherence"][idx] > 0.7

    def test_independent_low(self):
        rng = np.random.default_rng(5)
        out = coherence(rng.normal(size=512), rng.normal(size=512), seg_len=128)
        assert out["coherence"].mean() < 0.4

    def test_failclosed(self):
        with pytest.raises(ValueError):
            coherence(np.ones(50), np.ones(40))


class TestBandPower:
    def test_power_concentrates(self):
        v = _sine_plus_noise(freq=0.2, noise=0.1)
        out = periodogram_daniell(v, m=3)
        in_band = band_power(out["freqs"], out["density"], 0.15, 0.25)
        out_band = band_power(out["freqs"], out["density"], 0.3, 0.45)
        assert in_band > out_band

    def test_failclosed(self):
        with pytest.raises(ValueError):
            band_power(np.linspace(0, 0.5, 20), np.ones(20), 0.4, 0.6)
