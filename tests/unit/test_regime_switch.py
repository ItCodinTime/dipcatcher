"""Tests for Markov-switching models."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.regime_switch import (
    ergodic_probabilities,
    fit_markov_switching_mean,
    fit_markov_switching_regression,
    regime_classification_uncertainty,
    regime_expected_duration,
)


def _two_state_series(n: int = 400, seed: int = 0):
    """Low-vol state (mu=-0.5, sigma=0.5) and high-vol state (mu=0.5, sigma=2)."""
    rng = np.random.default_rng(seed)
    P_true = np.array([[0.98, 0.02], [0.03, 0.97]])
    state = np.zeros(n, dtype=int)
    y = np.zeros(n)
    mus = [-0.5, 0.5]
    sigs = [0.5, 2.0]
    for t in range(1, n):
        if rng.random() < P_true[state[t - 1], 1 - state[t - 1]]:
            state[t] = 1 - state[t - 1]
        else:
            state[t] = state[t - 1]
        y[t] = rng.normal(mus[state[t]], sigs[state[t]])
    return y, state


class TestMSMean:
    def test_recovers_regimes(self):
        y, state = _two_state_series(n=500)
        fit = fit_markov_switching_mean(y, n_states=2, seed=0)
        mus = np.sort(fit["means"])
        assert abs(mus[0] + 0.5) < 0.25
        assert abs(mus[1] - 0.5) < 0.25
        sigs = np.sort(fit["sigmas"])
        assert abs(sigs[0] - 0.5) < 0.25
        assert abs(sigs[1] - 2.0) < 0.5
        # Filtered probs recover states (allow label flip).
        pred = np.argmax(fit["smoothed"], axis=1)
        acc = max(np.mean(pred == state), np.mean(1 - pred == state))
        assert acc > 0.8

    def test_persistent_states(self):
        y, _ = _two_state_series()
        fit = fit_markov_switching_mean(y, n_states=2, seed=1)
        dur = regime_expected_duration(fit["P"])
        assert np.all(dur > 5.0)  # true durations ~ 33-50
        erg = ergodic_probabilities(fit["P"])
        assert np.isclose(erg.sum(), 1.0)
        assert np.all(erg > 0)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            fit_markov_switching_mean(np.ones(10))
        with pytest.raises(ValueError):
            fit_markov_switching_mean(np.ones(50), n_states=7)


class TestMSRegression:
    def test_recovers_betas(self):
        rng = np.random.default_rng(3)
        n = 400
        x = np.column_stack([np.ones(n), rng.normal(size=n)])
        state = np.zeros(n, dtype=int)
        y = np.zeros(n)
        P_true = np.array([[0.97, 0.03], [0.04, 0.96]])
        for t in range(1, n):
            if rng.random() < P_true[state[t - 1], 1 - state[t - 1]]:
                state[t] = 1 - state[t - 1]
            else:
                state[t] = state[t - 1]
            beta = np.array([0.0, 1.0]) if state[t] == 0 else np.array([1.0, -1.0])
            y[t] = x[t] @ beta + rng.normal(scale=0.3)
        fit = fit_markov_switching_regression(y, x, n_states=2, seed=0)
        # Recover the two beta pairs (up to label permutation).
        slopes = np.sort(fit["betas"][:, 1])
        assert abs(slopes[0] + 1.0) < 0.35
        assert abs(slopes[1] - 1.0) < 0.35

    def test_failclosed(self):
        with pytest.raises(ValueError):
            fit_markov_switching_regression(np.ones(30), np.ones((20, 1)))


class TestDiagnostics:
    def test_uncertainty_low_when_separated(self):
        s = np.tile(np.array([0.99, 0.01]), (100, 1))
        assert regime_classification_uncertainty(s) < 0.1
        s2 = np.tile(np.array([0.5, 0.5]), (100, 1))
        assert regime_classification_uncertainty(s2) > 0.6

    def test_ergodic(self):
        P = np.array([[0.9, 0.1], [0.2, 0.8]])
        pi = ergodic_probabilities(P)
        assert np.isclose(pi.sum(), 1.0)
        # Stationarity: pi @ P ~ pi.
        assert np.allclose(pi @ P, pi, atol=1e-6)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            regime_expected_duration(np.array([[0.5, 0.6], [0.4, 0.6]]))
        with pytest.raises(ValueError):
            regime_classification_uncertainty(np.full((10, 2), np.nan))
