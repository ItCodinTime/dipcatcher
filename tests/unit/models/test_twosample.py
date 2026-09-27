"""Tests for metrics/twosample.py — nonparametric two-sample tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.twosample import (
    cramer_von_mises_2samp,
    energy_distance,
    energy_test,
    ks_two_sample,
    permutation_test,
)


def test_ks_same_distribution_high_p() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = rng.standard_normal(500)
    out = ks_two_sample(x, y)
    assert 0.0 <= out["statistic"] <= 1.0
    assert out["pvalue"] > 0.1


def test_ks_shifted_distribution_low_p() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    y = rng.standard_normal(500) + 1.0
    assert ks_two_sample(x, y)["pvalue"] < 0.01


def test_energy_distance_nonneg_and_test() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(300)
    y = rng.standard_normal(300) + 1.5
    assert energy_test(x, y, n_perm=199, rng=rng)["pvalue"] < 0.05
    # energy distance is much larger for shifted than for same-distribution samples
    shifted_ed = energy_distance(x, y)
    same_ed = energy_distance(rng.standard_normal(300), rng.standard_normal(300))
    assert shifted_ed > 0
    assert same_ed < 0.25 * shifted_ed


def test_cvm_and_generic_permutation() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(200)
    y = rng.standard_normal(200) + 1.0
    assert cramer_von_mises_2samp(x, y, n_perm=199, rng=rng)["pvalue"] < 0.05
    diff_means = permutation_test(
        x, y, lambda a, b: abs(float(a.mean() - b.mean())), n_perm=199, rng=rng
    )
    assert diff_means["pvalue"] < 0.05


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        ks_two_sample(np.arange(3.0), np.arange(10.0))
    with pytest.raises(ValueError):
        energy_distance(np.arange(10.0), np.arange(2.0))
