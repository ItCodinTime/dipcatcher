"""Tests for cycle/spectral filters."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.features.cycles import (
    bandpass_filter,
    cycle_periodogram,
    dominant_cycle_fft,
    goertzel_power,
    hilbert_instantaneous_frequency,
    hilbert_transform_indicator,
    roofing_filter,
    supersmoother,
)


def _sine(period: float, n: int = 512, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / period) + noise * rng.normal(size=n)


class TestGoertzel:
    def test_peak_at_true_period(self):
        x = _sine(20.0)
        periods, power = cycle_periodogram(x, np.arange(6, 60))
        best = periods[np.argmax(power)]
        assert abs(best - 20.0) <= 1.5

    def test_single_bin(self):
        x = _sine(15.0)
        p15 = goertzel_power(x, 15)
        p40 = goertzel_power(x, 40)
        assert p15 > p40 * 5

    def test_failclosed(self):
        with pytest.raises(ValueError):
            goertzel_power(np.arange(10.0), 20)
        with pytest.raises(ValueError):
            goertzel_power(_sine(20), 3)
        with pytest.raises(ValueError):
            cycle_periodogram(_sine(20), np.array([2, 3]))


class TestFFT:
    def test_dominant_cycle(self):
        x = _sine(32.0)
        out = dominant_cycle_fft(x, min_period=8, max_period=100)
        assert abs(out["period"] - 32.0) < 4.0
        assert out["power_fraction"] > 0.3

    def test_failclosed(self):
        with pytest.raises(ValueError):
            dominant_cycle_fft(_sine(20), min_period=4, max_period=3)
        with pytest.raises(ValueError):
            dominant_cycle_fft(np.arange(10.0))


class TestHilbert:
    def test_if_near_true_period(self):
        x = _sine(24.0, noise=0.0)
        per = hilbert_instantaneous_frequency(x, smooth=1)
        tail = per[-80:]
        tail = tail[np.isfinite(tail)]
        assert tail.size > 20
        assert abs(np.median(tail) - 24.0) < 8.0

    def test_if_does_not_see_the_future(self):
        x = _sine(24.0, noise=0.0)
        a = hilbert_instantaneous_frequency(x, smooth=5)
        x2 = x.copy()
        x2[-1] += 5.0
        b = hilbert_instantaneous_frequency(x2, smooth=5)
        assert np.allclose(a[:-1], b[:-1], equal_nan=True)

    def test_analytic_signal(self):
        x = _sine(30.0, noise=0.0)
        out = hilbert_transform_indicator(x)
        tail = out["amplitude"][-80:]
        tail = tail[np.isfinite(tail)]
        assert tail.size > 20
        assert np.median(tail) > 0.2


class TestEhlersFilters:
    def test_supersmoother_smooths(self):
        x = _sine(40.0, noise=0.5)
        s = supersmoother(x, period=12.0)
        assert s.shape == x.shape
        assert np.var(s[50:]) < np.var(x[50:])

    def test_roofing_passes_cycles(self):
        # Pure sine in band -> filter preserves most amplitude.
        x = _sine(30.0, noise=0.0)
        r = roofing_filter(x, hp_period=100.0, lp_period=10.0)
        assert np.std(r[100:-100]) > 0.1

    def test_bandpass_selective(self):
        t = np.arange(512)
        x = np.sin(2 * np.pi * t / 20.0) + np.sin(2 * np.pi * t / 5.0)
        bp = bandpass_filter(x, period=20.0, bandwidth=0.3)
        # Correlate with pure 20-cycle sine — should dominate.
        ref20 = np.sin(2 * np.pi * t / 20.0)
        ref5 = np.sin(2 * np.pi * t / 5.0)
        tail = slice(150, -50)
        c20 = abs(np.corrcoef(bp[tail], ref20[tail])[0, 1])
        c5 = abs(np.corrcoef(bp[tail], ref5[tail])[0, 1])
        assert c20 > c5

    def test_failclosed(self):
        with pytest.raises(ValueError):
            supersmoother(np.arange(5.0))
        with pytest.raises(ValueError):
            supersmoother(_sine(20), period=1.0)
        with pytest.raises(ValueError):
            roofing_filter(_sine(20), hp_period=0.0)
        with pytest.raises(ValueError):
            bandpass_filter(_sine(20), bandwidth=1.5)
        with pytest.raises(ValueError):
            hilbert_instantaneous_frequency(np.arange(8.0))
