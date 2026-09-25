"""Tests for metrics/concordance.py — rank-association measures."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.concordance import (
    concordance_index,
    goodman_kruskal_gamma,
    kendall_tau_b,
    somers_d,
)


def test_perfect_concordance() -> None:
    x = np.arange(50, dtype=float)
    y = 2.0 * x + 1.0  # strictly increasing
    assert goodman_kruskal_gamma(x, y) == pytest.approx(1.0)
    assert somers_d(x, y) == pytest.approx(1.0)
    assert kendall_tau_b(x, y) == pytest.approx(1.0)


def test_perfect_discordance() -> None:
    x = np.arange(50, dtype=float)
    y = -x
    assert goodman_kruskal_gamma(x, y) == pytest.approx(-1.0)
    assert kendall_tau_b(x, y) == pytest.approx(-1.0)


def test_independence_near_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400)
    y = rng.standard_normal(400)
    assert abs(kendall_tau_b(x, y)) < 0.1
    assert abs(goodman_kruskal_gamma(x, y)) < 0.1


def test_c_index_equals_auc() -> None:
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(1.0, 1.0, 200), rng.normal(-1.0, 1.0, 200)])
    outcome = np.concatenate([np.ones(200), np.zeros(200)])
    c = concordance_index(scores, outcome)
    assert 0.7 < c <= 1.0
    # a perfect separator scores 1
    perfect = concordance_index(np.array([3.0, 2.0, 1.0, 0.0]), np.array([1, 1, 0, 0]))
    assert perfect == pytest.approx(1.0)


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        kendall_tau_b(np.arange(2.0), np.arange(2.0))
    with pytest.raises(ValueError):
        concordance_index(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))  # one class
