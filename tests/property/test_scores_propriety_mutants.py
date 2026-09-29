"""Strict-propriety spot mutants for the proper score functions.

Quick sanity mutants, not a full suite: a proper score must strictly prefer
the true distribution/quantile over perturbed candidates. If a mutation of a
correct forecast scores *better*, the score is not proper — a defect.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.scoring import (
    crps_empirical,
    crps_from_quantiles,
    crps_gaussian,
    mean_crps_gaussian,
    mean_pinball,
    pinball_loss,
    pit_values,
    qlike,
)


def _norm_ppf(taus: np.ndarray) -> np.ndarray:
    """Standard-normal quantiles via the error function (no scipy dependency)."""
    from scipy.special import ndtri

    return ndtri(taus)


def test_pinball_penalizes_asymmetric_error() -> None:
    """tau > 0.5 must charge an under-forecast tau/(1-tau) times an over-forecast."""
    rng = np.random.default_rng(20240910)
    y = rng.normal(size=4000)
    for tau in (0.9, 0.95, 0.99):
        d = 0.5
        under = pinball_loss(y + d, y, tau)  # forecast sits d below realized
        over = pinball_loss(y - d, y, tau)  # forecast sits d above realized
        ratio = float(under.mean() / over.mean())
        assert ratio == pytest.approx(tau / (1 - tau), rel=1e-9)


def test_pinball_true_quantile_minimizes_expected_loss() -> None:
    """A proper score's minimizer on a candidate grid is the true quantile."""
    rng = np.random.default_rng(20240911)
    y = rng.normal(size=20000)
    for tau in (0.1, 0.5, 0.9, 0.95):
        grid = np.linspace(-3.0, 3.0, 601)
        losses = np.array([mean_pinball(y, np.full(y.size, q), tau) for q in grid])
        best = grid[int(np.argmin(losses))]
        truth = float(_norm_ppf(np.array([tau]))[0])
        assert abs(best - truth) <= 0.02, f"tau={tau}: minimizer {best} != {truth}"
        # any off-quantile candidate must score strictly worse than truth
        for bad in (truth - 0.5, truth + 0.5):
            assert mean_pinball(y, np.full(y.size, bad), tau) > mean_pinball(
                y, np.full(y.size, truth), tau
            )


def test_crps_gaussian_favors_true_distribution() -> None:
    """Mean CRPS over seeded draws: truth beats shifted and rescaled mutants."""
    rng = np.random.default_rng(20240912)
    y = rng.normal(size=10000)
    truth = float(mean_crps_gaussian(y, np.zeros(y.size), np.ones(y.size)))
    for mu, sigma in [(0.25, 1.0), (-0.5, 1.0), (0.0, 0.5), (0.0, 2.0), (0.1, 0.9)]:
        mutant = float(mean_crps_gaussian(y, np.full(y.size, mu), np.full(y.size, sigma)))
        assert mutant > truth + 1e-4, f"mu={mu} sigma={sigma} scored {mutant} <= {truth}"


def test_crps_gaussian_perfect_forecast_floor() -> None:
    """Degenerate point mass: CRPS -> |y - mu| as sigma -> 0, never negative."""
    y = np.array([0.0, 1.0, -2.0])
    tiny = crps_gaussian(y, np.array([0.0, 1.0, -2.0]), np.full(3, 1e-9))
    assert np.isfinite(tiny).all()
    assert np.all(tiny >= 0.0)
    assert np.max(tiny) < 1e-6
    assert not np.isfinite(crps_gaussian(y, np.zeros(3), np.zeros(3))).all()


def test_crps_empirical_favors_true_samples() -> None:
    """Sampled truth beats sampled mutants under the empirical CRPS."""
    rng = np.random.default_rng(20240913)
    n = 400
    true_draws = rng.normal(size=n)
    shifted = true_draws + 0.5
    widened = true_draws * 2.0
    narrowed = true_draws * 0.4
    ys = rng.normal(size=600)
    score_t = float(np.mean([crps_empirical(y, true_draws) for y in ys]))
    for name, sample in [("shifted", shifted), ("widened", widened), ("narrowed", narrowed)]:
        score_m = float(np.mean([crps_empirical(y, sample) for y in ys]))
        assert score_m > score_t, f"{name} mutant scored {score_m} <= {score_t}"


def test_crps_from_quantiles_rejects_misassignment() -> None:
    """Truth quantiles beat tail-swapped and spread-mutated candidates."""
    rng = np.random.default_rng(20240914)
    taus = np.linspace(0.05, 0.95, 19)
    truth_q = _norm_ppf(taus)
    y = rng.normal(size=4000)
    q = np.tile(truth_q, (y.size, 1))
    truth = crps_from_quantiles(y, q, taus)
    # mutant 1: symmetric tails swapped (median body intact, extremes wrong)
    swapped = q.copy()
    swapped[:, [0, -1]] = q[:, [-1, 0]]
    # mutant 2: uniformly inflated tails
    widened = q * 1.5
    # mutant 3: all mass at the median
    collapsed = np.zeros_like(q)
    for name, cand in [("tail_swap", swapped), ("widened", widened), ("collapsed", collapsed)]:
        mutant = crps_from_quantiles(y, cand, taus)
        assert mutant > truth + 1e-4, f"{name} scored {mutant} <= {truth}"


def test_qlike_favors_true_variance() -> None:
    """QLIKE is proper: realized-variance truth beats scaled variance mutants."""
    rng = np.random.default_rng(20240915)
    realized_var = rng.normal(size=20000) ** 2  # mean 1 == true variance
    truth = qlike(realized_var, np.ones(realized_var.size))
    for var in (0.5, 2.0, 1.5, 0.75):
        mutant = qlike(realized_var, np.full(realized_var.size, var))
        assert mutant > truth, f"variance {var} scored {mutant} <= {truth}"


def test_pit_uniform_under_truth_not_under_mutants() -> None:
    """PIT values are ~uniform under the true model, skewed under mutants."""
    rng = np.random.default_rng(20240916)
    taus = np.linspace(0.05, 0.95, 19)
    y = rng.normal(size=4000)
    q_true = np.tile(_norm_ppf(taus), (y.size, 1))
    pits_true = pit_values(y, q_true, taus)
    assert abs(float(np.mean(pits_true)) - 0.5) < 0.02
    # a systematically shifted forecast overstates low quantiles -> PIT skew
    q_shift = np.tile(_norm_ppf(taus) + 0.8, (y.size, 1))
    pits_mut = pit_values(y, q_shift, taus)
    assert abs(float(np.mean(pits_mut)) - 0.5) > 0.1
