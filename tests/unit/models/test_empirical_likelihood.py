"""Tests for metrics/empirical_likelihood.py — Owen empirical likelihood."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.empirical_likelihood import (
    empirical_likelihood_ci,
    empirical_likelihood_mean,
)


def test_el_accepts_true_mean() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    out = empirical_likelihood_mean(x, 0.0)
    assert out["pvalue"] > 0.1  # true mean not rejected
    assert out["statistic"] >= 0.0


def test_el_rejects_far_mean() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    out = empirical_likelihood_mean(x, 0.4)
    assert out["pvalue"] < 0.05


def test_el_ci_contains_mean_and_ordered() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(400) + 1.0
    ci = empirical_likelihood_ci(x, level=0.95)
    assert ci["lower"] < ci["mean"] < ci["upper"]
    assert ci["lower"] < 1.0 < ci["upper"]


def test_fail_closed() -> None:
    x = np.random.default_rng(0).standard_normal(100)
    with pytest.raises(ValueError):
        empirical_likelihood_mean(x, 100.0)  # outside data range
    with pytest.raises(ValueError):
        empirical_likelihood_mean(np.arange(3.0), 1.0)  # too few obs
