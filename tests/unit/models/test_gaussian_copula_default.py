"""Tests for models/gaussian_copula_default.py — one-factor default distribution."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.gaussian_copula_default import (
    conditional_default_prob,
    portfolio_default_distribution,
)


def test_conditional_prob_monotone_in_factor() -> None:
    pd = np.array([0.05])
    hi = conditional_default_prob(pd, rho=0.2, m=-3.0)[0]  # bad factor -> more defaults
    lo = conditional_default_prob(pd, rho=0.2, m=3.0)[0]
    assert hi > lo


def test_pmf_sums_to_one_and_mean() -> None:
    pd = np.full(20, 0.05)
    out = portfolio_default_distribution(pd, rho=0.15)
    pmf = np.asarray(out["pmf"])
    assert abs(float(pmf.sum()) - 1.0) < 1e-9
    assert abs(out["mean"] - 20 * 0.05) < 0.05  # mean defaults = n * pd


def test_higher_correlation_fattens_tail() -> None:
    pd = np.full(30, 0.05)
    low = portfolio_default_distribution(pd, rho=0.05)
    high = portfolio_default_distribution(pd, rho=0.4)
    # both share the same mean, but high rho has larger variance (tail risk)
    assert high["var"] > low["var"]
    assert abs(high["mean"] - low["mean"]) < 0.1


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        portfolio_default_distribution(np.array([1.2]), rho=0.1)
    with pytest.raises(ValueError):
        portfolio_default_distribution(np.array([0.05]), rho=1.5)
