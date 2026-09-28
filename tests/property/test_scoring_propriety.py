"""Strict-propriety invariants for the proper-score implementations (SYNTHETIC).

A scoring rule is strictly proper iff the true data-generating distribution
uniquely minimizes its expected score. Every KAT in the suite can pass while
the rule is subtly wrong (wrong tail weight, dropped term, axis swap) — only a
propriety check catches that class. Exact identities are used where the math
permits; Monte-Carlo cases use seeded generators so the suite is deterministic.

- Brier: E[Brier(q)] over a Bernoulli sample with exactly p·n ones is
  minimized iff q == p (closed form; tested on the empirical measure).
- Pinball: in-sample, the empirical tau-quantile uniquely minimizes
  mean pinball (exact), and under an Exponential draw the true quantile
  beats shifted quantiles in expectation.
- CRPS-from-quantiles: a Gaussian mixture's true quantile grid scores lower
  than any shifted/scaled perturbation of it.
- QLIKE: with rv ~ v_true * chi^2_1 (unbiased proxy), E[QLIKE(v)] is
  minimized at v == v_true (Patton 2011 propriety condition).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.metrics.probability import brier_score
from quant_fund.metrics.scoring import (
    crps_from_quantiles,
    gaussian_mixture_quantiles,
    mean_pinball,
    qlike,
)


@given(
    n_one=st.integers(1, 40),
    n_zero=st.integers(1, 40),
    q=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=60, deadline=None)
def test_brier_minimized_at_event_frequency(n_one: int, n_zero: int, q: float) -> None:
    """Exact propriety on the empirical Bernoulli measure: p̂ is the unique min."""
    y = np.concatenate([np.ones(n_one), np.zeros(n_zero)])
    p_hat = n_one / (n_one + n_zero)
    at_p = brier_score(np.full(y.shape, p_hat), y)
    at_q = brier_score(np.full(y.shape, q), y)
    if abs(q - p_hat) < 1e-12:
        assert at_q == pytest.approx(at_p)
    else:
        assert at_q > at_p + 1e-12


@given(
    tau=st.floats(0.02, 0.98, allow_nan=False, allow_infinity=False),
    delta=st.floats(0.01, 0.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_pinball_minimized_at_empirical_quantile(tau: float, delta: float) -> None:
    """In-sample propriety is exact: the empirical tau-quantile is the argmin.

    The minimizer is the order statistic X_(ceil(n*tau)); np.quantile's
    default linear interpolation can land off the minimizing interval at
    extreme tau, so the minimizer is taken directly.
    """
    rng = np.random.default_rng(7)
    y = rng.standard_t(df=4, size=2000)
    q_true = float(np.sort(y)[max(int(np.ceil(y.size * tau)) - 1, 0)])
    at_true = mean_pinball(y, np.full(y.shape, q_true), tau)
    for shifted in (q_true - delta, q_true + delta, q_true + 5 * delta):
        assert mean_pinball(y, np.full(y.shape, shifted), tau) >= at_true


@given(
    lam=st.floats(0.2, 5.0, allow_nan=False, allow_infinity=False),
    tau=st.floats(0.3, 0.95, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_pinball_exponential_true_quantile_beats_shifts(lam: float, tau: float) -> None:
    """MC propriety: true Exp(λ) τ-quantile beats ±20% scale shifts.

    τ is floored at 0.3: for small τ the pinball surface is shallow enough
    that the expected gap falls below MC noise at any feasible sample size.
    """
    rng = np.random.default_rng(11)
    y = rng.exponential(scale=1.0 / lam, size=20000)
    q_true = -np.log(1.0 - tau) / lam
    at_true = mean_pinball(y, np.full(y.shape, q_true), tau)
    for shifted in (q_true * 0.8, q_true * 1.2):
        assert mean_pinball(y, np.full(y.shape, shifted), tau) > at_true


_TAUS = np.linspace(0.05, 0.95, 19)


@given(
    w1=st.floats(0.2, 0.8, allow_nan=False, allow_infinity=False),
    mu2=st.floats(0.5, 3.0, allow_nan=False, allow_infinity=False),
    s2=st.floats(0.3, 2.0, allow_nan=False, allow_infinity=False),
    shift=st.floats(0.02, 0.4, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=None)
def test_crps_from_quantiles_minimized_by_true_mixture(
    w1: float, mu2: float, s2: float, shift: float
) -> None:
    """The mixture's true quantile grid beats shifted and scale-stretched grids."""
    rng = np.random.default_rng(13)
    weights = np.array([w1, 1.0 - w1])
    mu = np.array([0.0, mu2])
    sigma = np.array([0.5, s2])
    # Draws from the exact mixture CDF via component sampling.
    comp = rng.binomial(1, w1, size=8000)
    y = np.where(comp == 1, rng.normal(mu[0], sigma[0], 8000), rng.normal(mu[1], sigma[1], 8000))
    q_true = np.tile(gaussian_mixture_quantiles(weights, mu, sigma, _TAUS), (y.size, 1))
    at_true = crps_from_quantiles(y, q_true, _TAUS)
    q_shifted = q_true + shift
    q_stretched = q_true * (1.0 + shift)
    assert crps_from_quantiles(y, q_shifted, _TAUS) > at_true
    assert crps_from_quantiles(y, q_stretched, _TAUS) > at_true


@given(
    v_true=st.floats(0.01, 5.0, allow_nan=False, allow_infinity=False),
    bias=st.floats(0.05, 0.6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_qlike_minimized_at_true_variance(v_true: float, bias: float) -> None:
    """Patton propriety: with an unbiased rv proxy, E[QLIKE] is min at v_true."""
    rng = np.random.default_rng(17)
    rv = v_true * rng.chisquare(df=1, size=8000)
    at_true = qlike(rv, np.full(rv.shape, v_true))
    for biased in (v_true * (1.0 - bias), v_true * (1.0 + bias)):
        assert qlike(rv, np.full(rv.shape, biased)) > at_true
