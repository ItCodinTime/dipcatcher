"""features/_kernels.py — call each njit kernel's .py_func for coverage parity."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quant_fund.features import _kernels


def _reset_bounds_py(weights: np.ndarray, threshold: float) -> np.ndarray:
    n = weights.shape[0]
    starts: list[int] = []
    acc = 0.0
    start = 0
    for t in range(n):
        acc += weights[t]
        if acc >= threshold:
            starts.append(start)
            start = t + 1
            acc = 0.0
    return np.asarray([s for s in starts if s < n - 1], dtype=np.int64)


class TestResetBounds:
    def test_empty_and_single(self) -> None:
        assert_array_equal(
            _kernels.reset_bounds.py_func(np.array([]), 1.0), np.array([], dtype=np.int64)
        )
        assert_array_equal(
            _kernels.reset_bounds.py_func(np.array([2.0]), 1.0), np.array([], dtype=np.int64)
        )

    def test_exact_excess_discarded(self) -> None:
        w = np.array([0.6, 0.6, 0.5, 1.1, 0.2])
        expected = _reset_bounds_py(w, 1.0)
        assert_array_equal(_kernels.reset_bounds.py_func(w, 1.0), expected)
        assert_array_equal(_kernels.reset_bounds(w, 1.0), expected)
        # acc hits threshold at t=1 (start 0) and t=3 (start 2); t=4 is final -> dropped.
        assert_array_equal(expected, np.array([0, 2]))

    def test_trailing_partial_not_emitted(self) -> None:
        w = np.array([3.0, 0.1, 0.1])
        # t=0 emits start 0; remainder 0.2 < 1.0 never reaches the threshold.
        assert_array_equal(_kernels.reset_bounds.py_func(w, 1.0), np.array([0]))


def _imbalance_py(values: np.ndarray, alpha: float, e_ticks: float, e_init: float) -> np.ndarray:
    n = values.shape[0]
    starts: list[int] = []
    theta = 0.0
    e_theta = e_init
    start = 0
    for t in range(n):
        theta += values[t]
        e_theta = (1.0 - alpha) * e_theta + alpha * abs(values[t])
        floor = e_theta if e_theta > 1e-12 else 1e-12
        if abs(theta) >= e_ticks * floor:
            starts.append(start)
            start = t + 1
            theta = 0.0
    return np.asarray([s for s in starts if s < n - 1], dtype=np.int64)


class TestImbalanceBounds:
    @pytest.mark.parametrize("seed", [0, 3])
    def test_matches_reference(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        v = rng.normal(0.0, 2.0, 200)
        expected = _imbalance_py(v, 0.1, 2.0, 1.0)
        assert_array_equal(_kernels.imbalance_bounds.py_func(v, 0.1, 2.0, 1.0), expected)
        assert len(expected) > 0

    def test_tiny_flow_floor(self) -> None:
        # Zero-magnitude ticks keep e_theta pinned at the 1e-12 floor.
        v = np.zeros(10)
        assert_array_equal(
            _kernels.imbalance_bounds.py_func(v, 0.5, 2.0, 0.0), np.array([], dtype=np.int64)
        )


def _run_py(signs: np.ndarray, alpha: float, e_ticks: float) -> np.ndarray:
    n = signs.shape[0]
    starts: list[int] = []
    start = 0
    e_share = 0.5
    run_buy = run_sell = 0.0
    for t in range(n):
        e_share = (1.0 - alpha) * e_share + alpha * (1.0 if signs[t] > 0.0 else 0.0)
        if signs[t] > 0.0:
            run_buy += 1.0
        elif signs[t] < 0.0:
            run_sell += 1.0
        run = max(run_buy, run_sell)
        share = max(e_share, 1.0 - e_share)
        thresh = max(e_ticks * share, 2.0)
        if run >= thresh:
            starts.append(start)
            start = t + 1
            run_buy = run_sell = 0.0
    return np.asarray([s for s in starts if s < n - 1], dtype=np.int64)


class TestRunBounds:
    def test_matches_reference(self) -> None:
        rng = np.random.default_rng(1)
        signs = rng.choice([-1.0, 0.0, 1.0], size=300, p=[0.4, 0.1, 0.5])
        expected = _run_py(signs, 0.05, 3.0)
        assert_array_equal(_kernels.run_bounds.py_func(signs, 0.05, 3.0), expected)
        assert len(expected) > 0

    def test_all_one_side(self) -> None:
        signs = np.ones(8)
        out = _kernels.run_bounds.py_func(signs, 0.5, 2.0)
        # e_share converges to 1 -> thresh 2 -> emits every 2 ticks.
        assert len(out) == 4

    def test_zeros_neither_side(self) -> None:
        signs = np.zeros(12)
        out = _kernels.run_bounds.py_func(signs, 0.5, 2.0)
        assert out.size == 0


def _goertzel_py(demeaned: np.ndarray, periods: np.ndarray) -> np.ndarray:
    n = demeaned.shape[0]
    out = np.empty(periods.shape[0])
    n2 = float(n * n)
    for pi, period in enumerate(periods):
        coeff = 2.0 * math.cos(2.0 * math.pi / period)
        s_prev = s_prev2 = 0.0
        for x in demeaned:
            s = x + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s
        out[pi] = (s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2) / n2
    return out


class TestGoertzelPowers:
    def test_peaks_at_true_period(self) -> None:
        n = 240
        t = np.arange(n, dtype=float)
        signal = np.sin(2.0 * np.pi * t / 30.0)
        demeaned = signal - signal.mean()
        periods = np.array([10.0, 20.0, 30.0, 40.0, 60.0])
        out = _kernels.goertzel_powers.py_func(demeaned, periods)
        assert_allclose(out, _goertzel_py(demeaned, periods), rtol=1e-12, atol=1e-15)
        assert np.argmax(out) == 2  # the 30-bar component

    def test_empty_signal_zero_power(self) -> None:
        out = _kernels.goertzel_powers.py_func(np.zeros(16), np.array([8.0]))
        assert out[0] == 0.0
