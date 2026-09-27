"""Tests for models/gam.py — backfitting generalized additive model."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.gam import gam_fit


def test_recovers_additive_structure() -> None:
    rng = np.random.default_rng(0)
    n = 600
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    truth1 = np.sin(1.5 * x1)
    truth2 = 0.5 * x2
    y = 2.0 + truth1 + truth2 + 0.1 * rng.standard_normal(n)
    out = gam_fit(np.column_stack([x1, x2]), y)
    contrib = np.asarray(out["contributions"])
    assert out["r2"] > 0.85
    # each recovered smooth correlates with its true component
    assert np.corrcoef(contrib[:, 0], truth1)[0, 1] > 0.9
    assert np.corrcoef(contrib[:, 1], truth2)[0, 1] > 0.9
    assert abs(out["intercept"] - 2.0) < 0.1


def test_contributions_centered() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, (300, 2))
    y = x[:, 0] ** 2 + x[:, 1] + 0.1 * rng.standard_normal(300)
    out = gam_fit(x, y)
    contrib = np.asarray(out["contributions"])
    assert abs(contrib[:, 0].mean()) < 1e-6
    assert abs(contrib[:, 1].mean()) < 1e-6


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        gam_fit(np.ones((10, 2)), np.ones(10))  # too few obs
    with pytest.raises(ValueError):
        gam_fit(np.ones((30, 2)), np.ones(29))  # misaligned
