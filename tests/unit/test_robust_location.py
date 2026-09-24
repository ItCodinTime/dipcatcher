"""Tests for metrics/robust_location.py — Hodges-Lehmann & Siegel repeated median."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.robust_location import (
    hodges_lehmann,
    hodges_lehmann_shift,
    siegel_repeated_median,
)


def test_hodges_lehmann_centers_symmetric_data() -> None:
    rng = np.random.default_rng(0)
    x = 4.0 + rng.standard_normal(200)
    hl = hodges_lehmann(x)["estimate"]
    assert abs(hl - 4.0) < 0.25


def test_hodges_lehmann_resists_outliers() -> None:
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.standard_normal(100), np.full(8, 1000.0)])
    hl = hodges_lehmann(x)["estimate"]
    # mean is wrecked by the outliers; HL stays near the true center
    assert abs(hl) < 1.0
    assert abs(float(x.mean())) > 30.0


def test_hodges_lehmann_shift() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(150) + 3.0
    y = rng.standard_normal(150)
    shift = hodges_lehmann_shift(x, y)["estimate"]
    assert abs(shift - 3.0) < 0.3


def test_siegel_recovers_line_with_outliers() -> None:
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 60)
    y = 1.5 + 2.0 * x + 0.2 * rng.standard_normal(x.size)
    # corrupt 25% of responses
    idx = rng.choice(x.size, size=15, replace=False)
    y[idx] += 50.0
    out = siegel_repeated_median(x, y)
    assert abs(out["slope"] - 2.0) < 0.3
    assert abs(out["intercept"] - 1.5) < 1.0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        hodges_lehmann(np.array([np.nan, 1.0]))
    with pytest.raises(ValueError):
        siegel_repeated_median(np.arange(5.0), np.arange(4.0))
    with pytest.raises(ValueError):
        siegel_repeated_median(np.ones(5), np.arange(5.0))  # no distinct x
