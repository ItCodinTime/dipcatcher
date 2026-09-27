"""Tests for models/creditrisk_plus.py — CreditRisk+ Panjer recursion."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.creditrisk_plus import creditrisk_plus_distribution


def test_pmf_normalises_and_p0() -> None:
    pd = np.array([0.02, 0.03, 0.05, 0.01])
    ev = np.array([1, 1, 2, 3])
    out = creditrisk_plus_distribution(pd, ev, max_units=60)
    pmf = np.asarray(out["pmf"])
    assert abs(float(pmf.sum()) - 1.0) < 1e-6
    assert abs(pmf[0] - np.exp(-float(pd.sum()))) < 1e-9


def test_mean_matches_expected_loss() -> None:
    pd = np.array([0.02, 0.04, 0.03, 0.05, 0.02])
    ev = np.array([1, 2, 1, 3, 2])
    out = creditrisk_plus_distribution(pd, ev, max_units=80)
    pmf = np.asarray(out["pmf"])
    units = np.asarray(out["units"])
    mean = float(np.sum(units * pmf))
    var = float(np.sum((units - mean) ** 2 * pmf))
    assert abs(mean - out["expected_loss"]) < 1e-3
    assert abs(var - out["loss_var"]) < 1e-2  # compound Poisson var = sum pd v^2


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        creditrisk_plus_distribution(np.array([0.02, 0.03]), np.array([1]))  # misaligned
    with pytest.raises(ValueError):
        creditrisk_plus_distribution(np.array([1.2]), np.array([1]))  # pd >= 1
    with pytest.raises(ValueError):
        creditrisk_plus_distribution(np.array([0.02]), np.array([0]))  # non-positive exposure
