"""Tests for metrics/forecast_accuracy.py — Theil U and MASE."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.forecast_accuracy import mase, theil_u1, theil_u2


def test_perfect_forecast_zero() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert theil_u1(a, a) == pytest.approx(0.0)
    assert theil_u2(a, a) == pytest.approx(0.0)
    assert mase(a, a) == pytest.approx(0.0)


def test_theil_u2_naive_is_one() -> None:
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.standard_normal(200))
    naive = np.concatenate([[a[0]], a[:-1]])  # last-value forecast
    assert abs(theil_u2(a, naive) - 1.0) < 1e-9


def test_good_forecast_beats_naive() -> None:
    rng = np.random.default_rng(1)
    a = np.cumsum(rng.standard_normal(300))
    good = a + 0.1 * rng.standard_normal(300)  # small error
    assert theil_u2(a, good) < 1.0
    assert mase(a, good) < 1.0
    assert 0.0 <= theil_u1(a, good) <= 1.0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        theil_u1(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        mase(np.arange(10.0), np.arange(10.0), seasonality=0)
