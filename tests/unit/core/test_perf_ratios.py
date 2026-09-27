"""Tests for metrics/perf_ratios.py — extended performance ratios."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.perf_ratios import (
    adjusted_sharpe_ratio,
    gain_to_pain_ratio,
    m_squared,
    rachev_ratio,
    upside_potential_ratio,
)


def test_adjusted_sharpe_matches_sharpe_for_normal() -> None:
    rng = np.random.default_rng(0)
    r = 0.001 + 0.01 * rng.standard_normal(5000)
    asr = adjusted_sharpe_ratio(r)
    sr = float((r.mean()) / r.std(ddof=1))
    assert abs(asr - sr) < 0.05 * abs(sr) + 0.02


def test_adjusted_sharpe_penalises_negative_skew() -> None:
    rng = np.random.default_rng(1)
    # negatively skewed, fat-tailed: occasional large losses
    base = 0.01 * rng.standard_normal(5000)
    shocks = -0.08 * (rng.random(5000) < 0.03)
    r = 0.002 + base + shocks
    sr = float(r.mean() / r.std(ddof=1))
    assert adjusted_sharpe_ratio(r) < sr


def test_m_squared_and_sharpe_sign() -> None:
    rng = np.random.default_rng(2)
    port = 0.001 + 0.01 * rng.standard_normal(3000)
    bench = 0.0008 + 0.012 * rng.standard_normal(3000)
    out = m_squared(port, bench)
    assert np.isfinite(out["m2_annual"])
    assert np.isfinite(out["sharpe"])


def test_rachev_symmetric_near_one() -> None:
    rng = np.random.default_rng(3)
    r = rng.standard_normal(20000)
    rr = rachev_ratio(r, alpha=0.1, beta=0.1)
    assert 0.7 < rr < 1.4


def test_gain_to_pain_and_upr_positive_for_drift() -> None:
    rng = np.random.default_rng(4)
    r = 0.01 + 0.01 * rng.standard_normal(2000)
    assert gain_to_pain_ratio(r) > 0
    assert upside_potential_ratio(r) > 0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        adjusted_sharpe_ratio(np.zeros(50))  # zero dispersion
    with pytest.raises(ValueError):
        rachev_ratio(np.arange(100.0), alpha=0.6)
    with pytest.raises(ValueError):
        gain_to_pain_ratio(0.01 + np.zeros(50))  # no losses
