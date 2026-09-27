"""Variance-reduction factors, including the cases that must not be claimed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.mc_engine.variance import (
    antithetic_mean_vrf,
    control_variate_from_stats,
    cross_stats,
    effective_sample_size,
    importance_mean_vrf,
    importance_shift_vector,
    importance_weights,
    rqmc_mean_vrf,
)


def test_antithetic_exact_cancellation_is_infinite_not_a_finite_factor() -> None:
    values = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 4.0, -4.0])
    report = antithetic_mean_vrf(values)
    assert report["variance_reduction_factor"] is None
    assert report["variance_reduction_factor_infinite"] is True
    assert report["variance_reduced_estimator"] == 0.0
    assert float(report["variance_crude_estimator"]) > 0.0


def test_antithetic_factor_can_be_below_one() -> None:
    # Identical pairs: antithetic pairing does not reduce variance.
    values = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    report = antithetic_mean_vrf(values)
    assert report["variance_reduction_factor_infinite"] is False
    factor = float(report["variance_reduction_factor"])
    assert 0.0 < factor < 1.0


def test_control_variate_uses_the_pilot_only() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=400)
    y = 3.0 * x + 0.05 * rng.normal(size=400)
    pilot = cross_stats(y[:40], x[:40])
    evaluation = cross_stats(y[40:], x[40:])
    report = control_variate_from_stats(pilot, evaluation, control_mean=0.0)
    assert float(report["coefficient"]) == pytest.approx(3.0, abs=0.15)
    factor = float(report["variance_reduction_factor"])
    assert factor > 5.0
    # In-sample coefficient on the evaluation slice would be a different number;
    # the report's coefficient must match the pilot formula exactly.
    x_p, y_p = x[:40], y[:40]
    numer = float(np.sum((x_p - 0.0) * (y_p - y_p.mean())))
    den = float(np.sum((x_p - 0.0) ** 2))
    assert float(report["coefficient"]) == pytest.approx(numer / den)


def test_control_variate_fail_closed_on_a_tiny_pilot() -> None:
    report = control_variate_from_stats((1, 0.0, 0.0, 0.0, 0.0, 0.0), (10, 1, 1, 1, 1, 1), 0.0)
    assert report["variance_reduction_factor"] is None
    assert "pilot" in str(report["reason"])


def test_importance_weights_are_unbiased_for_the_mean_and_report_ess() -> None:
    rng = np.random.default_rng(5)
    n = 20_000
    dim = 8
    shift = importance_shift_vector(n_steps=dim, n_factors=1, shift=-0.25)
    y = rng.normal(size=(n, dim))
    z = y + shift
    weights = importance_weights(z, shift)
    assert abs(float(weights.mean()) - 1.0) < 0.05
    # E[w Z_0] = E[Y_0] = 0
    assert abs(float(np.mean(weights * z[:, 0]))) < 0.05
    ess = effective_sample_size(weights)
    assert 0.2 * n < ess <= n


def test_importance_vrf_on_a_rare_event_beats_crude_monte_carlo() -> None:
    rng = np.random.default_rng(6)
    n = 8_000
    threshold = 2.5
    crude = (rng.normal(size=n) > threshold).astype(np.float64)
    proposal = rng.normal(loc=threshold, size=n)
    # 1-d likelihood ratio for N(threshold, 1) vs N(0, 1)
    log_w = -threshold * proposal + 0.5 * threshold**2
    weighted = np.exp(log_w) * (proposal > threshold)
    report = importance_mean_vrf(crude, weighted)
    factor = float(report["variance_reduction_factor"])
    assert factor > 2.0


def test_rqmc_factor_is_the_declared_ratio() -> None:
    scramble_means = np.array([0.01, -0.02, 0.0, 0.015])
    crude = np.array([1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
    report = rqmc_mean_vrf(scramble_means, crude)
    var_scramble = float(np.var(scramble_means, ddof=1))
    var_crude_mean = float(np.var(crude, ddof=1) / crude.size)
    assert float(report["variance_reduction_factor"]) == pytest.approx(
        var_crude_mean / var_scramble
    )
    assert report["variance_reduction_factor_infinite"] is False


def test_rqmc_without_enough_scrambles_does_not_invent_a_factor() -> None:
    report = rqmc_mean_vrf(np.array([0.2]), np.arange(5, dtype=float))
    assert report["variance_reduction_factor"] is None
