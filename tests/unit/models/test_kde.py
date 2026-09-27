"""Tests for metrics/kde.py — Gaussian KDE and bandwidth selectors."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.metrics.kde import (
    gaussian_kde,
    lscv_bandwidth,
    scott_bandwidth,
    silverman_bandwidth,
)


def test_kde_integrates_to_one() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    h = silverman_bandwidth(x)
    grid = np.linspace(-8, 8, 4001)
    area = float(np.trapezoid(gaussian_kde(x, grid, h), grid))
    assert abs(area - 1.0) < 1e-3


def test_kde_recovers_normal_peak() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4000)
    h = silverman_bandwidth(x)
    dens0 = float(gaussian_kde(x, np.array([0.0]), h)[0])
    assert abs(dens0 - norm.pdf(0.0)) < 0.05


def test_bandwidths_positive_and_scale() -> None:
    rng = np.random.default_rng(2)
    x = 3.0 * rng.standard_normal(1000)
    hs = silverman_bandwidth(x)
    hsc = scott_bandwidth(x)
    hl = lscv_bandwidth(x)
    assert hs > 0 and hsc > 0 and hl > 0
    # all comparable to the normal-reference scale
    assert 0.2 * hs < hl < 5.0 * hs


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        silverman_bandwidth(np.arange(3.0))
    with pytest.raises(ValueError):
        gaussian_kde(np.random.default_rng(0).standard_normal(50), np.array([0.0]), 0.0)
