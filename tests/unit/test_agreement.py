"""Tests for metrics/agreement.py — Cohen's kappa and weighted kappa."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.agreement import cohen_kappa, weighted_kappa


def test_perfect_agreement() -> None:
    y = np.array([0, 1, 2, 1, 0, 2, 1, 0, 2, 2])
    assert cohen_kappa(y, y) == pytest.approx(1.0)
    assert weighted_kappa(y, y, "quadratic") == pytest.approx(1.0)


def test_chance_agreement_near_zero() -> None:
    rng = np.random.default_rng(0)
    y1 = rng.integers(0, 3, size=2000)
    y2 = rng.integers(0, 3, size=2000)
    assert abs(cohen_kappa(y1, y2)) < 0.1


def test_weighted_kappa_rewards_near_misses() -> None:
    rng = np.random.default_rng(1)
    y1 = rng.integers(0, 5, size=500)
    # y2 mostly off by one -> unweighted low, quadratic-weighted higher
    y2 = np.clip(y1 + rng.integers(-1, 2, size=500), 0, 4)
    kw = weighted_kappa(y1, y2, "quadratic")
    ku = cohen_kappa(y1, y2)
    assert kw > ku


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        cohen_kappa(np.array([1]), np.array([1]))  # too few
    with pytest.raises(ValueError):
        weighted_kappa(np.array([0, 1, 2]), np.array([0, 1, 2]), weights="cubic")


@pytest.mark.parametrize(
    "invalid", [np.array([0.0, 1.5]), np.array([0.0, np.nan]), np.array([0.0, np.inf])]
)
def test_invalid_numeric_labels_rejected(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="labels"):
        cohen_kappa(np.array([0, 1]), invalid)
    with pytest.raises(ValueError, match="labels"):
        weighted_kappa(np.array([0, 1]), invalid)
