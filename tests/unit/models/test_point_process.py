"""Hawkes point-process battery: MLE recovery, residuals, simulation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.point_process import (
    hawkes2_mle,
    hawkes_intensity,
    hawkes_loglik,
    hawkes_mle,
    hawkes_residuals,
    hawkes_simulate,
)


def _hawkes(n_target: int = 400, mu=0.4, alpha=0.6, beta=1.2, seed=0) -> np.ndarray:
    # Simulate generously then truncate to n events for a stable MLE fixture.
    ev = hawkes_simulate(mu=mu, alpha=alpha, beta=beta, horizon=2000.0, seed=seed)
    assert ev.size >= n_target
    return ev[:n_target]


def _poisson(n: int = 300, seed=1) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).exponential(1.0, size=n))


def test_intensity_and_loglik_positive() -> None:
    ev = _hawkes()
    lam = hawkes_intensity(ev, 0.4, 0.6, 1.2)
    assert np.all(lam > 0.0)
    ll = hawkes_loglik(ev, 0.4, 0.6, 1.2)
    assert np.isfinite(ll)


def test_mle_recovers_branching_ratio() -> None:
    ev = _hawkes(n_target=600, alpha=0.6, seed=3)
    fit = hawkes_mle(ev)
    assert fit["branching_ratio"] == pytest.approx(0.6, abs=0.25)
    assert fit["stationary"] == 1.0


def test_mle_poisson_low_reflexivity() -> None:
    ev = _poisson(400)
    fit = hawkes_mle(ev)
    assert fit["branching_ratio"] < 0.7  # Poisson -> weak excitation


def test_compensator_residuals_iid_for_true_model() -> None:
    ev = _hawkes(n_target=500, mu=0.5, alpha=0.5, beta=1.0, seed=5)
    out = hawkes_residuals(ev, 0.5, 0.5, 1.0)
    # Under the true params the transformed times are ~uniform.
    assert out["ks_pvalue"] > 0.01
    assert out["mean_tau"] == pytest.approx(1.0, abs=0.3)


def test_simulate_event_count_scales() -> None:
    short = hawkes_simulate(0.5, 0.5, 1.0, horizon=200.0, seed=7)
    long = hawkes_simulate(0.5, 0.5, 1.0, horizon=800.0, seed=7)
    assert long.size > short.size
    # Rough intensity check: rate ~ mu/(1-alpha) = 1.0 -> ~200 events in 200.
    assert 80 < short.size < 400


def test_simulate_fail_closed_explosive() -> None:
    with pytest.raises(ValueError):
        hawkes_simulate(0.5, 1.2, 1.0, horizon=10.0)


def test_hawkes2_mutual_excitation() -> None:
    # Cross-excitation 1->2: drive t2 from a hawkes sim, then fit.
    rng = np.random.default_rng(9)
    t1 = hawkes_simulate(0.4, 0.4, 1.0, horizon=800.0, seed=9)
    t2 = np.sort(t1 + rng.exponential(0.2, size=t1.size))  # clustered after t1
    t2 = np.unique(np.concatenate([t2, hawkes_simulate(0.3, 0.3, 1.0, 800.0, seed=10)]))
    out = hawkes2_mle(t1[:300], t2[:300])
    assert out["branching_ratio"] >= 0.0
    assert np.isfinite(out["loglik"])
    assert out["alpha21"] >= 0.0  # 1->2 excitation channel present in params


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        hawkes_mle(np.array([1.0, 1.0, 2.0, 2.0, 3.0]))  # not strictly increasing
    with pytest.raises(ValueError):
        hawkes_mle(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        hawkes_intensity(_hawkes(50), mu=0.0, alpha=0.5, beta=1.0)
    with pytest.raises(ValueError):
        hawkes_residuals(np.arange(10.0) + 1, mu=-0.1, alpha=0.5, beta=1.0)  # mu<=0
