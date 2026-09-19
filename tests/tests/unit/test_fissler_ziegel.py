"""DayWave3: Fissler–Ziegel joint VaR/ES scoring — closed-form + edges.

Research-diagnostic only — live_pnl_claim=false; not a live capital claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.scoring import fissler_ziegel_loss, mean_fissler_ziegel

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def test_fz0_closed_form_hit_and_miss() -> None:
    """Hand-check canonical Nolde–Ziegel FZ0: S = (1/(1-α))·1_{L>v}(L-v)/e + v/e − 1 + log(e)."""
    alpha = 0.95
    v, e = 1.0, 2.0
    losses = np.array([3.0, 0.5])  # hit, miss
    var = np.full(2, v)
    es = np.full(2, e)
    scores = fissler_ziegel_loss(losses, var, es, alpha)
    pb = 1.0 - alpha
    s_hit = (3.0 - v) / e / pb + v / e - 1.0 + math.log(e)
    s_miss = v / e - 1.0 + math.log(e)
    assert scores[0] == pytest.approx(s_hit)
    assert scores[1] == pytest.approx(s_miss)
    assert mean_fissler_ziegel(losses, var, es, alpha) == pytest.approx(0.5 * (s_hit + s_miss))


def test_fz0_properness_minimizes_at_true_es() -> None:
    """Bugbot regression: expected FZ0 must be minimized at the TRUE ES, not (1-α)·ES.

    Two-point tail: L=1 w.p. 0.05 (breach), L=0 otherwise. α=0.95 coverage →
    true VaR=0, true ES=1. A proper joint (VaR, ES) score has its expectation
    minimized at e=ES=1.0. The earlier un-scaled variant minimized at
    e=(1-α)·ES=0.05, i.e. was not a proper scoring rule.
    """
    alpha = 0.95
    rng = np.random.default_rng(0)
    losses = (rng.random(400_000) < 0.05).astype(float)
    var_true = np.full(losses.size, 0.0)
    grid = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    means = [mean_fissler_ziegel(losses, var_true, np.full(losses.size, e), alpha) for e in grid]
    argmin = grid[int(np.argmin(means))]
    assert argmin == pytest.approx(1.0), f"FZ0 minimized at e={argmin}, expected ES=1.0"
    # Explicitly exclude the old improper minimizer
    assert means[grid.index(0.05)] > means[grid.index(1.0)], (
        "score at (1-α)·ES must exceed score at true ES"
    )


def test_fz0_empty_mismatch_bad_alpha() -> None:
    empty = np.array([])
    assert fissler_ziegel_loss(empty, empty, empty, 0.95).size == 0
    assert math.isnan(mean_fissler_ziegel(empty, empty, empty, 0.95))
    with pytest.raises(ValueError, match="length mismatch"):
        fissler_ziegel_loss(np.array([1.0, 2.0]), np.array([1.0]), np.array([2.0, 2.0]), 0.95)
    with pytest.raises(ValueError, match="alpha"):
        fissler_ziegel_loss(np.array([1.0]), np.array([1.0]), np.array([2.0]), 0.0)
    with pytest.raises(ValueError, match="alpha"):
        mean_fissler_ziegel(np.array([1.0]), np.array([1.0]), np.array([2.0]), 1.0)
    with pytest.raises(ValueError, match="alpha"):
        mean_fissler_ziegel(np.array([1.0]), np.array([1.0]), np.array([2.0]), float("nan"))


def test_fz0_nonpositive_es_and_nonfinite_nan() -> None:
    losses = np.array([0.2, 0.3, 0.1])
    var = np.full(3, 0.15)
    es_bad = np.array([0.2, 0.0, -0.1])
    scores = fissler_ziegel_loss(losses, var, es_bad, 0.95)
    assert np.isfinite(scores[0])
    assert math.isnan(scores[1]) and math.isnan(scores[2])
    # mean ignores NaN indices → finite only over the good row
    assert mean_fissler_ziegel(losses, var, es_bad, 0.95) == pytest.approx(scores[0])
    # all non-positive ES → NaN mean
    assert math.isnan(mean_fissler_ziegel(losses, var, np.zeros(3), 0.95))
    # non-finite loss → NaN at that index
    scores2 = fissler_ziegel_loss(
        np.array([np.nan, 0.3]), np.array([0.1, 0.1]), np.array([0.2, 0.2]), 0.9
    )
    assert math.isnan(scores2[0]) and np.isfinite(scores2[1])
