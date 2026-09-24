"""Tests for metrics/entropic_risk.py — entropic risk and Ahmadi-Javid EVaR."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.entropic_risk import entropic_risk_measure, entropic_value_at_risk


def test_entropic_small_theta_approaches_mean() -> None:
    rng = np.random.default_rng(0)
    losses = rng.standard_normal(5000)
    assert abs(entropic_risk_measure(losses, theta=1e-3) - float(losses.mean())) < 0.02


def test_entropic_increases_with_theta() -> None:
    rng = np.random.default_rng(1)
    losses = rng.standard_normal(5000)
    assert entropic_risk_measure(losses, 2.0) > entropic_risk_measure(losses, 0.5)


def test_evar_dominates_cvar() -> None:
    rng = np.random.default_rng(2)
    losses = rng.standard_normal(20000)
    alpha = 0.95
    var = float(np.quantile(losses, alpha))
    cvar = float(losses[losses >= var].mean())
    evar = entropic_value_at_risk(losses, alpha)["evar"]
    assert evar >= cvar - 1e-6
    assert evar >= var - 1e-6


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        entropic_risk_measure(np.zeros(10), theta=0.0)
    with pytest.raises(ValueError):
        entropic_value_at_risk(np.zeros(10), alpha=1.5)
    with pytest.raises(ValueError):
        entropic_risk_measure(np.array([1.0, np.nan, 2.0, 3.0, 4.0]))
