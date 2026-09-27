"""Reverse-stress geometry and the VaR/ES backtest wiring."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from quant_fund.metrics.risk import gaussian_es, gaussian_var, historical_es, historical_var
from quant_fund.stress.reverse import (
    default_radius,
    mahalanobis_distance,
    plausibility_score,
    reverse_stress,
    sample_mean_cov,
    worst_linear_scenario,
)
from quant_fund.stress.risk import (
    backtest_var_es,
    bootstrap_var_es_interval,
    causal_historical_forecasts,
    garch_filtered_var_es,
    point_estimates,
)


def test_linear_reverse_stress_matches_the_analytic_boundary() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(6, 3))
    cov = raw.T @ raw / 6.0 + np.eye(3) * 0.05
    mu = np.array([0.01, -0.02, 0.0])
    weights = np.array([0.4, 0.5, -0.1])
    radius = default_radius(3, contour=0.05)
    analytic = worst_linear_scenario(weights, mu, cov, radius)
    assert analytic.method == "analytic_linear"
    assert analytic.on_boundary is True
    assert analytic.mahalanobis_distance == pytest.approx(radius, rel=1e-8)
    assert analytic.plausibility_score == pytest.approx(0.05, abs=1e-8)
    assert analytic.gaussian_density_ratio == pytest.approx(float(np.exp(-0.5 * radius**2)))
    expected_loss = float(-weights @ mu + radius * np.sqrt(weights @ cov @ weights))
    assert analytic.loss == pytest.approx(expected_loss)

    searched = reverse_stress(
        lambda x: float(-weights @ x), mu, cov, radius, n_directions=48, seed=1
    )
    assert searched.loss == pytest.approx(analytic.loss, abs=1e-5)
    assert searched.scenario == pytest.approx(analytic.scenario, abs=1e-4)
    assert searched.mahalanobis_distance <= radius * (1.0 + 1e-6)
    assert plausibility_score(0.0, 3) == pytest.approx(1.0)


def test_interior_optimum_stays_at_the_center() -> None:
    mu = np.array([0.2, -0.1])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    radius = 1.5

    def loss(x: np.ndarray) -> float:
        d = np.asarray(x, dtype=float) - mu
        return float(-(d @ d))

    found = reverse_stress(loss, mu, cov, radius, n_directions=16, seed=2)
    assert found.loss == pytest.approx(0.0, abs=1e-8)
    assert found.scenario == pytest.approx(mu, abs=1e-6)
    assert found.on_boundary is False
    assert found.plausibility_score == pytest.approx(1.0)
    assert found.loss >= loss(mu + np.array([0.01, 0.0]))


def test_reverse_stress_fail_closed() -> None:
    mu = np.zeros(2)
    cov = np.eye(2)
    with pytest.raises(TypeError):
        reverse_stress(object(), mu, cov, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        worst_linear_scenario(np.ones(2), mu, np.array([[1.0, 2.0], [0.0, 1.0]]), 1.0)
    with pytest.raises(ValueError):
        worst_linear_scenario(np.ones(2), mu, np.array([[1.0, 0.0], [0.0, 0.0]]), 1.0)
    with pytest.raises(ValueError):
        reverse_stress(lambda x: float(x.sum()), mu, cov, 1.0, n_directions=True)  # type: ignore[arg-type]
    mean, fitted = sample_mean_cov(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 4.0]]), ridge=1e-4)
    assert mean.shape == (2,)
    assert float(np.linalg.eigvalsh(fitted)[0]) > 0.0
    assert mahalanobis_distance(mu, mu, cov) == pytest.approx(0.0)


def test_point_estimates_match_the_existing_battery() -> None:
    losses = np.random.default_rng(3).normal(size=400)
    level = 0.95
    estimates = point_estimates(losses, level)
    assert estimates["historical"]["var"] == pytest.approx(historical_var(losses, level))
    assert estimates["historical"]["es"] == pytest.approx(historical_es(losses, level))
    assert estimates["gaussian"]["var"] == pytest.approx(gaussian_var(losses, level))
    assert estimates["gaussian"]["es"] == pytest.approx(gaussian_es(losses, level))
    for name in ("student_t", "cornish_fisher", "garch_filtered"):
        assert estimates[name]["es"] >= estimates[name]["var"]
    filtered = garch_filtered_var_es(losses, level)
    assert filtered["sigma_last"] > 0.0
    with pytest.raises(ValueError):
        point_estimates(losses, level=0.2)


def test_bootstrap_interval_covers_the_gaussian_quantile() -> None:
    losses = np.random.default_rng(4).normal(size=8000)
    intervals = bootstrap_var_es_interval(losses, 0.95, n_boot=200, seed=4)
    true_var = float(stats.norm.ppf(0.95))
    assert intervals["var"].low <= true_var <= intervals["var"].high
    assert intervals["es"].low < intervals["es"].high
    assert intervals["var"].method == "stationary_bootstrap_percentile"
    with pytest.raises(ValueError):
        bootstrap_var_es_interval(losses, n_boot=10)


def test_backtests_reject_a_bad_var_and_keep_a_correct_one() -> None:
    rng = np.random.default_rng(8)
    losses = rng.normal(size=800)
    level = 0.95
    z = float(stats.norm.ppf(level))
    true_es = float(stats.norm.pdf(z) / (1.0 - level))
    correct = backtest_var_es(
        losses, np.full(losses.size, z), np.full(losses.size, true_es), level, n_boot=200, seed=0
    )
    assert correct["kupiec"]["status"] == "ok"
    assert correct["kupiec"]["pvalue"] > 0.05
    assert correct["christoffersen"]["status"] == "ok"
    assert np.isfinite(correct["christoffersen"]["lr_cc"])
    assert correct["acerbi_szekely_z1"]["status"] == "ok"
    assert np.isfinite(correct["acerbi_szekely_z1"]["z1"])
    assert correct["acerbi_szekely_bootstrap"]["status"] == "ok"

    understated = np.full(losses.size, float(stats.norm.ppf(0.80)))
    bad = backtest_var_es(
        losses, understated, np.full(losses.size, true_es), level, n_boot=80, seed=1
    )
    assert bad["kupiec"]["pvalue"] < 1e-6

    quiet = np.zeros(40)
    undefined = backtest_var_es(quiet, np.ones(40), np.full(40, 2.0), level, n_boot=50, seed=0)
    assert undefined["christoffersen"]["status"] == "undefined"
    assert undefined["acerbi_szekely_z1"]["status"] == "undefined"
    assert undefined["acerbi_szekely_bootstrap"]["status"] == "undefined"

    realized, var, es = causal_historical_forecasts(
        np.random.default_rng(9).normal(size=120), level, min_history=40
    )
    assert realized.shape == var.shape == es.shape
    assert realized.size == 80
    with pytest.raises(ValueError):
        backtest_var_es(np.ones(10), np.ones(10), np.ones(10), level)
