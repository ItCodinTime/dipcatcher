"""Pathwise drawdown, ruin, recovery, ES intervals, and POT diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sstats

from quant_fund.mc_engine.tails import (
    batch_means_es_interval,
    gpd_var_es,
    path_risk_stats,
    pot_gpd,
    spectral_es,
    weighted_expected_shortfall,
    wilson_interval,
)
from quant_fund.metrics.spectral_risk import expected_shortfall_srm


def test_drawdown_ruin_and_recovery_on_a_hand_path() -> None:
    returns = np.array([[-0.5, 0.1, 1.0]])
    stats = path_risk_stats(returns, ruin_level=0.6)
    # wealth: 1, 0.5, 0.55, 1.1. Max drawdown 0.5 at the first step after start.
    assert float(stats["max_drawdown"][0]) == pytest.approx(0.5)
    assert int(stats["ruined"][0]) == 1
    assert int(stats["recovered"][0]) == 1
    assert int(stats["recovery_steps"][0]) == 2
    assert float(stats["loss"][0]) == pytest.approx(1.0 - 1.1)
    flat = path_risk_stats(np.zeros((1, 4)), ruin_level=0.5)
    assert int(flat["no_drawdown"][0]) == 1
    assert int(flat["recovered"][0]) == 0
    assert int(flat["recovery_steps"][0]) == -1


def test_censored_recovery_is_not_treated_as_a_short_recovery() -> None:
    returns = np.array([[-0.2, -0.1, -0.1]])
    stats = path_risk_stats(returns, ruin_level=0.01)
    assert int(stats["recovered"][0]) == 0
    assert int(stats["recovery_steps"][0]) == -1
    assert int(stats["no_drawdown"][0]) == 0


def test_spectral_es_matches_the_canonical_definition_and_weights() -> None:
    losses = np.linspace(-1.0, 3.0, 40)
    assert spectral_es(losses, 0.9) == pytest.approx(expected_shortfall_srm(losses, 0.9))
    weights = np.ones_like(losses)
    assert weighted_expected_shortfall(losses, weights, 0.9) == pytest.approx(
        spectral_es(losses, 0.9)
    )
    # 97.5% ES is at least the 99%? No: higher alpha is the more extreme tail.
    assert spectral_es(losses, 0.99) >= spectral_es(losses, 0.975) - 1e-12


def test_batch_means_interval_contains_the_pooled_estimate_on_a_normal_sample() -> None:
    rng = np.random.default_rng(8)
    losses = rng.normal(size=4_000)
    interval = batch_means_es_interval(losses, 0.975, 0.95)
    assert interval["ci_low"] < interval["estimate"] < interval["ci_high"]
    # Theoretical ES of a standard normal at 97.5% is phi(q)/(1-alpha).
    q = float(sstats.norm.ppf(0.975))
    theoretical = float(sstats.norm.pdf(q) / 0.025)
    assert abs(float(interval["estimate"]) - theoretical) < 0.15
    small = batch_means_es_interval(losses[:30], 0.975, 0.95)
    assert small["ci_low"] is None
    assert small["reason"] == "fewer than 8 batches"


def test_wilson_interval_covers_zero_and_one_counts() -> None:
    low, high = wilson_interval(0.0, 100, 1.959963984540054)
    assert 0.0 <= low <= high <= 1.0
    assert low == 0.0
    assert high > 0.0
    low_one, high_one = wilson_interval(100.0, 100, 1.959963984540054)
    assert high_one == 1.0
    assert low_one < 1.0


def test_gpd_fit_recovers_a_known_shape_and_fails_closed() -> None:
    rng = np.random.default_rng(9)
    excess = sstats.genpareto.rvs(c=0.25, loc=0.0, scale=1.0, size=8_000, random_state=rng)
    body = np.linspace(-1.0, 1.9, 2_000)
    sample = np.concatenate([body, excess + 2.0])
    report = pot_gpd(sample, threshold=2.0)
    assert report["available"] is True
    assert abs(float(report["xi"]) - 0.25) < 0.08
    assert report["diagnostics_ok"] is True
    assert report["es_finite"] is True
    assert report["tail"]["es_0.99"] is not None
    too_small = pot_gpd(np.arange(10, dtype=float))
    assert too_small["available"] is False
    assert too_small["diagnostics_ok"] is False


def test_gpd_var_es_formula_matches_the_mean_excess_identity() -> None:
    var, es = gpd_var_es(0.2, 1.0, threshold=1.5, phi_u=0.05, alpha=0.99)
    assert es > var > 1.5
    with pytest.raises(ValueError):
        gpd_var_es(0.2, 1.0, 1.5, 0.05, alpha=0.9)
