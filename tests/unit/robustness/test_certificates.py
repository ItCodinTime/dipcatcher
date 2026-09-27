"""Closed forms: smoothing radii, Wasserstein bounds, toy strategy radii."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from quant_fund.robustness.analytic import (
    clopper_pearson_lower,
    cost_shock_radius,
    gelbrich_worst_case_ratio,
    linear_l2_radius,
    linear_positive_probability,
    lipschitz_worst_case,
    smoothing_radius,
    wasserstein_worst_case_mean,
)
from quant_fund.robustness.strategies import LinearMargin, MeanSign, Threshold


def test_clopper_pearson_matches_the_beta_inversion() -> None:
    # 81 successes in 100 trials. The one-sided bound at alpha=0.025 is the
    # lower end of the two-sided 95% Clopper-Pearson interval.
    bound = clopper_pearson_lower(81, 100, 0.025)
    assert bound == pytest.approx(float(stats.beta.ppf(0.025, 81, 20)))
    assert 0.70 < bound < 0.81
    assert clopper_pearson_lower(0, 20, 0.05) == 0.0


def test_cohen_binary_radius_matches_the_simplified_formula() -> None:
    sigma = 0.5
    p_lower = 0.9
    general = smoothing_radius(p_lower, 1.0 - p_lower, sigma)
    binary = sigma * float(stats.norm.ppf(p_lower))
    assert general == pytest.approx(binary)
    assert smoothing_radius(0.4, 0.4, sigma) == 0.0


def test_gaussian_tail_round_trip_stays_near_the_geometric_radius() -> None:
    # Hypothesis found weights [0, 1], sample [0, 1], sigma 0.15625.
    # Standardized margin is 6.4. scipy.stats.norm.ppf(norm.cdf(6.4))
    # undershoots by about 9e-8, so the radius gap is about 1.4e-8.
    # That is float64 inversion error, not a gap in Cohen's identity.
    weights = np.array([0.0, 1.0])
    sample = np.array([0.0, 1.0])
    sigma = 0.15625
    distance = linear_l2_radius(weights, 0.0, sample)
    assert distance / sigma > 5.5
    positive = linear_positive_probability(weights, 0.0, sample, sigma)
    certified = smoothing_radius(positive, 1.0 - positive, sigma)
    assert abs(certified - distance) < 1e-7


def test_linear_population_radius_equals_distance_to_the_boundary() -> None:
    weights = np.array([0.5, 0.0, -0.5, 1.0])
    sample = np.array([0.2, 1.0, -0.4, 0.3])
    sigma = 0.25
    distance = linear_l2_radius(weights, 0.1, sample)
    positive = linear_positive_probability(weights, 0.1, sample, sigma)
    assert positive > 0.5
    certified = smoothing_radius(positive, 1.0 - positive, sigma)
    assert certified == pytest.approx(distance)


def test_threshold_and_mean_sign_match_their_linear_radii() -> None:
    level = 0.3
    sample = np.array([0.8])
    threshold = Threshold(level)
    assert threshold.analytic_l2_radius(sample) == pytest.approx(0.5)
    window = np.array([0.2, -0.1, 0.4, 0.1])
    mean_sign = MeanSign(window.size)
    # |sum| / sqrt(n) = |0.6| / 2
    assert mean_sign.analytic_l2_radius(window) == pytest.approx(0.3)
    strategy = LinearMargin(np.ones(window.size))
    assert strategy.decision(window) == 1


def test_worst_case_mean_is_the_translation() -> None:
    assert wasserstein_worst_case_mean(0.4, 0.1) == pytest.approx(0.3)
    assert wasserstein_worst_case_mean(-0.2, 0.0) == pytest.approx(-0.2)


def test_lipschitz_shift_is_symmetric() -> None:
    lower = lipschitz_worst_case(1.5, 2.0, 0.25, lower=True)
    upper = lipschitz_worst_case(1.5, 2.0, 0.25, lower=False)
    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(2.0)


def test_gelbrich_ratio_matches_a_boundary_grid() -> None:
    cases = [
        (1.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, 0.5, None),
        (2.0, 1.0, 0.3, None),
        (0.0, 1.0, 0.4, None),
        (-1.0, 2.0, 0.2, None),
        (0.5, 1.0, 0.5, 0.0),
        (2.0, 1.0, 1.0, 0.75),
        (1.0, 1.0, 1.0, 0.0),
    ]
    for mean, scale, radius, expected in cases:
        value, _kind = gelbrich_worst_case_ratio(mean, scale, radius)
        grid = _grid_minimum(mean, scale, radius)
        assert value is not None
        assert value == pytest.approx(grid, rel=1e-4, abs=1e-4)
        if expected is not None:
            assert value == pytest.approx(expected)


def test_gelbrich_ratio_is_unbounded_when_the_disk_allows_it() -> None:
    value, kind = gelbrich_worst_case_ratio(-2.0, 1.0, 1.0)
    assert value is None
    assert kind == "unbounded_below"
    value, kind = gelbrich_worst_case_ratio(2.0, 1.0, 3.0)
    assert value is None
    assert kind == "unbounded_below"


def test_cost_shock_radius_is_algebraic() -> None:
    assert cost_shock_radius(0.3, 1.0, 0.0) == pytest.approx(0.3)
    assert cost_shock_radius(0.3, 1.0, 0.1) == pytest.approx(0.2)
    assert cost_shock_radius(0.0, 1.0, 0.0) == 0.0
    assert cost_shock_radius(0.2, 0.0, 0.0) is None


def _grid_minimum(mean: float, scale: float, radius: float) -> float:
    if radius == 0.0:
        return mean / scale
    thetas = np.linspace(0.0, 2.0 * math.pi, 20000, endpoint=False)
    delta = radius * np.cos(thetas)
    eta = radius * np.sin(thetas)
    scale_prime = scale + eta
    mean_prime = mean + delta
    ok = scale_prime > 1e-8
    return float(np.min(mean_prime[ok] / scale_prime[ok]))
