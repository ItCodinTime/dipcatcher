"""Tests for models/smoothers.py — NW, local linear, spline."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.smoothers import (
    gcv_score,
    local_linear,
    nadaraya_watson,
    smoothing_spline,
)


def _curve(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 4 * np.pi, n))
    m = np.sin(x)
    y = m + 0.1 * rng.standard_normal(n)
    return x, y, m


def test_nw_recovers_smooth() -> None:
    x, y, m = _curve()
    fit = nadaraya_watson(x, y, h=0.4)
    assert np.abs(fit - m).mean() < 0.15


def test_local_linear_less_boundary_bias() -> None:
    # linear trend: local-linear should be ~unbiased at boundaries,
    # NW biased. Test interior recovery of slope.
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 10, 400))
    y = 2.0 * x + 0.2 * rng.standard_normal(400)
    out = local_linear(x, y, h=1.0)
    slope = np.asarray(out["slope"])
    fit = np.asarray(out["fit"])
    assert np.abs(slope[20:-20] - 2.0).mean() < 0.15
    assert np.abs(fit - 2.0 * x).mean() < 0.3


def test_spline_interpolation_limit() -> None:
    x, y, _ = _curve(n=60)
    out = smoothing_spline(x, y, lam=0.0)
    f = np.asarray(out["fit_sorted"])
    ys = y[np.argsort(x)]
    assert np.abs(f - ys).max() < 1e-6  # lam=0 interpolates


def test_spline_smooths_noise() -> None:
    x, y, m = _curve()
    out = smoothing_spline(x, y, lam=1.0)
    f = np.asarray(out["fit_sorted"])
    ms = m  # already sorted
    assert np.abs(f - ms).mean() < 0.15
    assert 1.0 < out["edf"] < x.size


def test_gcv_prefers_interior_lambda() -> None:
    x, y, _ = _curve()
    grid = np.logspace(-3, 3, 13)
    scores = np.array([gcv_score(x, y, lam) for lam in grid])
    best = grid[np.argmin(scores)]
    assert grid[1] < best < grid[-2]  # interior minimum


def test_fail_closed() -> None:
    x, y, _ = _curve(n=20)
    with pytest.raises(ValueError):
        nadaraya_watson(x, y, h=-1.0)
    with pytest.raises(ValueError):
        nadaraya_watson(x, np.full_like(y, np.nan), h=1.0)
    with pytest.raises(ValueError):
        local_linear(np.ones(10), np.ones(10), h=1.0)  # degenerate design
    with pytest.raises(ValueError):
        smoothing_spline(x, y, lam=-0.5)
    with pytest.raises(ValueError):
        nadaraya_watson(x[:4], y[:4], h=1.0)  # fewer than 5 obs
