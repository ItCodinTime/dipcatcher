"""Tests for metrics/utility.py — CRRA utility and certainty equivalent."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.utility import certainty_equivalent, crra_utility, expected_utility


def test_risk_neutral_ce_equals_mean() -> None:
    rng = np.random.default_rng(0)
    r = 0.01 + 0.05 * rng.standard_normal(10000)
    ce = certainty_equivalent(r, gamma=0.0)
    assert abs(ce - float(r.mean())) < 1e-6


def test_risk_averse_ce_below_mean() -> None:
    rng = np.random.default_rng(1)
    r = 0.01 + 0.05 * rng.standard_normal(10000)
    ce = certainty_equivalent(r, gamma=3.0)
    assert ce < float(r.mean())


def test_ce_decreasing_in_risk_aversion() -> None:
    rng = np.random.default_rng(2)
    r = 0.01 + 0.06 * rng.standard_normal(10000)
    ces = [certainty_equivalent(r, g) for g in (0.5, 2.0, 5.0, 10.0)]
    assert all(b < a for a, b in zip(ces, ces[1:], strict=False))


def test_log_utility_branch() -> None:
    w = np.array([1.0, np.e])
    u = crra_utility(w, gamma=1.0)
    assert np.allclose(u, [0.0, 1.0])
    assert np.isfinite(expected_utility(np.array([0.01, -0.02, 0.03]), gamma=1.0))


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        crra_utility(np.array([-1.0, 2.0]), gamma=2.0)  # non-positive wealth
    with pytest.raises(ValueError):
        certainty_equivalent(np.array([-1.5, 0.1]), gamma=2.0)  # 1 + r <= 0
