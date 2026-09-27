"""Tests for models/seasonal.py — decomposition + strength."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.seasonal import (
    seasonal_decompose,
    seasonal_strength,
    trend_strength,
)


def _seasonal_series(n: int = 200, period: int = 12, amp: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.02 * t
    seasonal = amp * np.sin(2 * np.pi * t / period)
    return 10.0 + trend + seasonal + 0.2 * rng.standard_normal(n)


def test_additive_decompose_recovers_seasonal() -> None:
    y = _seasonal_series()
    out = seasonal_decompose(y, period=12)
    idx = np.asarray(out["indices"])
    # seasonal indices should approximate 2*sin(2*pi*k/12)
    true_idx = 2.0 * np.sin(2 * np.pi * np.arange(12) / 12)
    assert np.abs(idx - true_idx).max() < 0.4
    resid = np.asarray(out["resid"])
    valid = np.isfinite(resid)
    assert np.abs(resid[valid]).mean() < 0.4


def test_seasonal_strength_ordering() -> None:
    strong = _seasonal_series(amp=3.0, seed=1)
    weak = _seasonal_series(amp=0.2, seed=2)
    none = np.random.default_rng(3).standard_normal(200)
    s_strong = seasonal_strength(strong, 12)
    s_weak = seasonal_strength(weak, 12)
    s_none = seasonal_strength(none, 12)
    assert s_strong > s_weak > s_none
    assert 0.0 <= s_none <= 1.0 and s_strong <= 1.0


def test_trend_strength() -> None:
    rng = np.random.default_rng(4)
    t = np.arange(240)
    trendy = 0.15 * t + 0.3 * np.sin(2 * np.pi * t / 12) + 0.2 * rng.standard_normal(240)
    flat = 0.3 * np.sin(2 * np.pi * t / 12) + 0.2 * rng.standard_normal(240)
    assert trend_strength(trendy, 12) > trend_strength(flat, 12)


def test_multiplicative() -> None:
    rng = np.random.default_rng(5)
    t = np.arange(240)
    y = (
        np.exp(0.003 * t)
        * (1.0 + 0.3 * np.sin(2 * np.pi * t / 12))
        * np.exp(0.02 * rng.standard_normal(240))
    )
    out = seasonal_decompose(y, 12, model="multiplicative")
    resid = np.asarray(out["resid"])
    valid = np.isfinite(resid)
    assert np.abs(resid[valid] - 1.0).mean() < 0.15


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        seasonal_decompose(np.arange(10.0), period=12)  # < 2 periods
    with pytest.raises(ValueError):
        seasonal_decompose(np.random.default_rng(0).standard_normal(100), period=1)
    with pytest.raises(ValueError):
        seasonal_decompose(
            -np.abs(np.random.default_rng(0).standard_normal(100)) - 1, 12, model="multiplicative"
        )
    with pytest.raises(ValueError):
        seasonal_decompose(np.full(50, np.nan), 12)
