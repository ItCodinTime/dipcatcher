"""Permutation-invariance pins: rank-derived statistics must not depend on
input row order. Ordinal ranks (argsort of argsort) break ties by position;
midranks assign every tied element the group mean rank."""

from __future__ import annotations

import numpy as np

from quant_fund.metrics.twosample import _cvm_statistic
from quant_fund.utils.numeric import midrank


def test_midrank_known_answers() -> None:
    np.testing.assert_allclose(midrank(np.array([1.0, 1.0, 3.0])), [1.5, 1.5, 3.0])
    np.testing.assert_allclose(midrank(np.array([4.0, 1.0, 1.0, 1.0])), [4.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(midrank(np.array([2.0, 5.0, 1.0])), [2.0, 3.0, 1.0])


def test_midrank_matches_scipy() -> None:
    from scipy.stats import rankdata

    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, size=200).astype(float)  # heavy ties
    np.testing.assert_allclose(midrank(x), rankdata(x, method="average"))


def test_midrank_permutation_invariant() -> None:
    rng = np.random.default_rng(1)
    x = np.array([0.5, 0.5, 0.5, 0.1, 0.9])
    r0 = midrank(x)
    for _ in range(20):
        perm = rng.permutation(x.size)
        np.testing.assert_allclose(midrank(x[perm])[np.argsort(perm)], r0)


def test_cvm_statistic_permutation_invariant_under_ties() -> None:
    """Ordinal ranks made the CvM statistic depend on where cross-sample ties
    sit in the pooled array; midranks remove that order dependence."""
    x = np.array([1.0, 2.0, 2.0, 5.0, 9.0])
    y = np.array([2.0, 3.0, 8.0])
    base = _cvm_statistic(x, y)
    rng = np.random.default_rng(7)
    for _ in range(50):
        px, py = rng.permutation(x.size), rng.permutation(y.size)
        assert _cvm_statistic(x[px], y[py]) == base


def test_cvm_statistic_unchanged_without_ties() -> None:
    """No ties → midranks equal ordinal ranks → statistic bit-identical."""
    x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    y = np.array([2.0, 4.0, 6.0])
    pooled = np.concatenate([x, y])
    ordinal = np.argsort(np.argsort(pooled)) + 1
    nx = x.size
    i, j = np.arange(1, nx + 1), np.arange(1, y.size + 1)
    ref = (
        nx * float(np.sum((np.sort(ordinal[:nx]) - i) ** 2))
        + y.size * float(np.sum((np.sort(ordinal[nx:]) - j) ** 2))
    ) / (nx * y.size * (nx + y.size)) - (4.0 * nx * y.size - 1.0) / (6.0 * (nx + y.size))
    assert _cvm_statistic(x, y) == ref
