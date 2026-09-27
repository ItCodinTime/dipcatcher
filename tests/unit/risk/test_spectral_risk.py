"""Tests for metrics/spectral_risk.py — Acerbi spectral risk measures."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.spectral_risk import (
    expected_shortfall_srm,
    exponential_spectral_risk,
    power_spectral_risk,
    spectral_risk_measure,
)


def test_power_gamma_one_is_mean() -> None:
    rng = np.random.default_rng(0)
    losses = rng.standard_normal(500)
    assert abs(power_spectral_risk(losses, gamma=1.0) - float(losses.mean())) < 1e-9


def test_exponential_increases_with_risk_aversion() -> None:
    rng = np.random.default_rng(1)
    losses = rng.standard_normal(2000)
    low = exponential_spectral_risk(losses, k=1.0)
    high = exponential_spectral_risk(losses, k=20.0)
    assert high > low
    assert high > float(losses.mean())


def test_es_srm_matches_tail_mean() -> None:
    losses = np.arange(100, dtype=float)  # sorted 0..99
    es = expected_shortfall_srm(losses, alpha=0.9)
    # worst 10% = the 10 largest values (90..99)
    assert abs(es - float(np.mean(losses[90:]))) < 1e-9


def test_srm_at_least_mean_and_leq_max() -> None:
    rng = np.random.default_rng(2)
    losses = rng.standard_normal(1000)
    srm = exponential_spectral_risk(losses, k=8.0)
    assert float(losses.mean()) <= srm <= float(losses.max()) + 1e-9


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        exponential_spectral_risk(np.array([1.0, np.nan, 2.0, 3.0, 4.0]))
    with pytest.raises(ValueError):
        power_spectral_risk(np.arange(10.0), gamma=0.5)
    with pytest.raises(ValueError):
        expected_shortfall_srm(np.arange(10.0), alpha=1.5)
    with pytest.raises(ValueError):
        spectral_risk_measure(np.arange(10.0), np.ones(9))  # misaligned weights
