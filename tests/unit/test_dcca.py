"""Tests for metrics/dcca.py — detrended cross-correlation analysis."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.dcca import dcca


def _random_walk_pair(
    rho_sign: float, n: int = 2000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n)
    ex = rng.standard_normal(n)
    ey = rng.standard_normal(n)
    x = common + 0.3 * ex
    y = rho_sign * common + 0.3 * ey
    return x, y


def test_positive_cross_correlation() -> None:
    x, y = _random_walk_pair(1.0)
    out = dcca(x, y)
    assert out["rho_mean"] > 0.5
    assert np.isfinite(out["exponent"])


def test_independent_series_near_zero() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)
    out = dcca(x, y)
    assert abs(out["rho_mean"]) < 0.3


def test_negative_cross_correlation() -> None:
    x, y = _random_walk_pair(-1.0, seed=3)
    out = dcca(x, y)
    assert out["rho_mean"] < -0.4


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        dcca(np.arange(10.0), np.arange(10.0))  # too short
    with pytest.raises(ValueError):
        dcca(np.arange(100.0), np.arange(50.0))  # misaligned
