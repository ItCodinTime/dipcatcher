"""Tests for models/expectile.py — Newey-Powell expectiles & regression."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.expectile import expectile, expectile_regression, expectile_var


def test_expectile_half_equals_mean() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500) * 2.0 + 1.0
    assert abs(expectile(x, 0.5) - float(x.mean())) < 1e-6


def test_expectile_monotone_in_tau() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1000)
    vals = [expectile(x, tau) for tau in (0.1, 0.25, 0.5, 0.75, 0.9)]
    assert all(b > a for a, b in zip(vals, vals[1:], strict=False))


def test_expectile_regression_recovers_slope() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(400)
    y = 1.0 + 2.0 * x + 0.3 * rng.standard_normal(x.size)
    out = expectile_regression(x, y, tau=0.5)
    coef = np.asarray(out["coef"])
    assert coef.shape == (2,)
    assert abs(coef[0] - 1.0) < 0.1
    assert abs(coef[1] - 2.0) < 0.1


def test_expectile_var_coherent_range() -> None:
    rng = np.random.default_rng(3)
    losses = rng.standard_normal(1000)
    ev = expectile_var(losses, tau=0.95)
    assert ev > float(np.median(losses))
    with pytest.raises(ValueError):
        expectile_var(losses, tau=0.3)


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        expectile(np.array([1.0]), 0.5)
    with pytest.raises(ValueError):
        expectile(np.arange(10.0), tau=0.0)
    with pytest.raises(ValueError):
        expectile_regression(np.ones((5, 2)), np.arange(5.0))  # rank-deficient + const
