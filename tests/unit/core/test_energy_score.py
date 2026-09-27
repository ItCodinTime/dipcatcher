"""Tests for the multivariate energy score (Gneiting & Raftery 2007)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.energy_score import (
    EnergyScoreCurve,
    energy_score,
    energy_score_curve,
    threshold_energy_score,
)


def test_hand_computable_3x2_matches_analytic() -> None:
    ensemble = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    obs = np.zeros(2)
    # d_obs = [0, 1, 1] -> term1 = 2/3.
    # sum_{i!=j} ||x_i - x_j|| = 2*(1 + 1 + sqrt(2)) -> term2 = (2 + sqrt(2))/6.
    expected = (2.0 - np.sqrt(2.0)) / 6.0
    assert energy_score(ensemble, obs) == pytest.approx(expected, abs=1e-12)


def test_hand_computable_threshold_weighted_3x2() -> None:
    ensemble = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    obs = np.zeros(2)
    # weight = 2, threshold = 0.5: w([0,1,1]) = [1, 2, 2] (tails amplified).
    # term1 = (0*1 + 1*2 + 1*2)/3 = 4/3.
    # weighted pair sum = 8 + 8*sqrt(2) -> term2 = (2 + 2*sqrt(2))/3.
    expected = (2.0 - 2.0 * np.sqrt(2.0)) / 3.0
    got = threshold_energy_score(ensemble, obs, threshold=0.5, weight=2.0)
    assert got == pytest.approx(expected, abs=1e-12)
    # weight = 1.0 reduces exactly to the plain energy score.
    plain = energy_score(ensemble, obs)
    assert threshold_energy_score(ensemble, obs, threshold=0.5, weight=1.0) == pytest.approx(
        plain, abs=1e-12
    )


def test_sharper_ensemble_scores_lower_than_over_dispersed() -> None:
    # Propriety is a statement about expectations, so average each ensemble's
    # score over 64 observations (all ~ N(0, I2)) per seed; the strict
    # inequality then holds on every seed.
    n_seeds = 20
    for s in range(n_seeds):
        rng = np.random.default_rng(1000 + s)
        obs = rng.normal(0.0, 1.0, size=(64, 2))
        sharp = rng.normal(0.0, 1.0, size=(400, 2))
        wide = rng.normal(0.0, 3.0, size=(400, 2))
        sharp_scores = np.mean([energy_score(sharp, y) for y in obs])
        wide_scores = np.mean([energy_score(wide, y) for y in obs])
        assert sharp_scores < wide_scores, f"seed {s}"


def test_correct_correlation_beats_wrong_sign_correlation() -> None:
    # Observation law N(0, Sigma_rho); correct ensemble matches rho, the
    # challenger has the wrong sign (-rho) with identical marginals. The
    # expectation gap (~0.06 at rho=0.7) is small, so average over 64
    # observations per seed to make the direction strict on every seed.
    rho = 0.7
    cov = np.array([[1.0, rho], [rho, 1.0]])
    cov_wrong = cov * np.array([[1.0, -1.0], [-1.0, 1.0]])
    for s in range(10):
        rng = np.random.default_rng(2000 + s)
        obs = rng.multivariate_normal(np.zeros(2), cov, size=64)
        correct = rng.multivariate_normal(np.zeros(2), cov, size=800)
        wrong = rng.multivariate_normal(np.zeros(2), cov_wrong, size=800)
        correct_mean = np.mean([energy_score(correct, y) for y in obs])
        wrong_mean = np.mean([energy_score(wrong, y) for y in obs])
        assert correct_mean < wrong_mean, f"seed {s}"


def test_threshold_weighting_penalizes_tail_more_than_central() -> None:
    rng = np.random.default_rng(3000)
    ensemble = rng.normal(0.0, 1.0, size=(2000, 2))
    central = np.array([0.3, 0.0])
    tail = np.array([3.0, 0.0])
    tw_central = threshold_energy_score(ensemble, central, threshold=1.5, weight=2.0)
    tw_tail = threshold_energy_score(ensemble, tail, threshold=1.5, weight=2.0)
    plain_central = energy_score(ensemble, central)
    plain_tail = energy_score(ensemble, tail)
    # Tail hit penalized more than central miss, weighted and plain.
    assert tw_tail > tw_central
    assert plain_tail > plain_central
    # Weighting widens the tail-vs-central gap: amplification is what the
    # threshold-weighted score adds over the plain score.
    assert tw_tail - tw_central > plain_tail - plain_central


def test_energy_score_curve_flat_at_weight_one() -> None:
    rng = np.random.default_rng(4000)
    ensemble = rng.normal(0.0, 1.0, size=(200, 2))
    obs = rng.normal(0.0, 1.0, size=2)
    thresholds = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    curve = energy_score_curve(ensemble, obs, thresholds, weight=1.0)
    assert isinstance(curve, EnergyScoreCurve)
    plain = energy_score(ensemble, obs)
    np.testing.assert_allclose(curve.scores, np.full_like(thresholds, plain), atol=1e-12)
    assert curve.weight == 1.0
    np.testing.assert_array_equal(curve.thresholds, thresholds)
    amplified = energy_score_curve(ensemble, obs, thresholds, weight=2.0)
    # threshold = 0 amplifies every member (all distances > 0): the weighted
    # score is 2*term1 - 4*term2 = 2*plain - 2*term2 < plain.
    assert amplified.scores[0] < plain
    assert amplified.weight == 2.0
    assert np.all(np.isfinite(amplified.scores))


def test_fail_closed_edges() -> None:
    rng = np.random.default_rng(5000)
    ens = rng.normal(0.0, 1.0, size=(10, 2))
    obs = np.zeros(2)
    bad_nan = ens.copy()
    bad_nan[0, 0] = np.nan
    bad_inf = obs.copy()
    bad_inf[1] = np.inf
    with pytest.raises(ValueError):
        energy_score(ens[:1], obs)  # n_samples < 2
    with pytest.raises(ValueError):
        energy_score(ens, np.zeros(3))  # dim mismatch
    with pytest.raises(ValueError):
        energy_score(ens, np.zeros((2, 1)))  # observation not 1-D
    with pytest.raises(ValueError):
        energy_score(ens.ravel(), obs)  # ensemble not 2-D
    with pytest.raises(ValueError):
        energy_score(bad_nan, obs)  # non-finite ensemble
    with pytest.raises(ValueError):
        energy_score(ens, bad_inf)  # non-finite observation
    with pytest.raises(ValueError):
        threshold_energy_score(ens, obs, threshold=-1.0)  # negative threshold
    with pytest.raises(ValueError):
        threshold_energy_score(ens, obs, threshold=np.inf)  # non-finite threshold
    with pytest.raises(ValueError):
        threshold_energy_score(ens, obs, threshold=1.0, weight=0.5)  # weight < 1
    with pytest.raises(ValueError):
        threshold_energy_score(ens, obs, threshold=1.0, weight=np.nan)  # non-finite weight
    with pytest.raises(ValueError):
        energy_score_curve(ens, obs, np.array([]))  # empty grid
    with pytest.raises(ValueError):
        energy_score_curve(ens, obs, np.array([0.5, np.nan]))  # non-finite grid
