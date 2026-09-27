"""Tests for models/lowess.py — Cleveland robust LOWESS."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.lowess import lowess


def test_lowess_tracks_smooth_signal() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 2 * np.pi, 300)
    truth = np.sin(x)
    y = truth + 0.3 * rng.standard_normal(x.size)
    out = lowess(x, y, frac=0.3, iters=3)
    # smoothed curve is much closer to the truth than the noisy data
    assert np.mean((out["fitted"] - truth) ** 2) < 0.5 * np.mean((y - truth) ** 2)
    assert np.corrcoef(out["fitted"], truth)[0, 1] > 0.95


def test_lowess_resists_outliers() -> None:
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 200)
    truth = 2.0 + 0.5 * x
    y = truth + 0.1 * rng.standard_normal(x.size)
    y[::20] += 30.0  # gross outliers
    out = lowess(x, y, frac=0.4, iters=4)
    mid = slice(50, 150)
    assert np.max(np.abs(out["fitted"][mid] - truth[mid])) < 3.0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        lowess(np.arange(3.0), np.arange(3.0))
    with pytest.raises(ValueError):
        lowess(np.arange(50.0), np.arange(50.0), frac=1.5)
