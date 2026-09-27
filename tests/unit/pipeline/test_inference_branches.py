"""Inference branch coverage: DM ties, alternative directions, bootstrap blocks.

Complements test_inference / test_inference_edges / test_inference_helpers_edges
by pinning the remaining fail-closed and alternative-direction branches.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.inference import (
    benjamini_hochberg,
    bootstrap_mean_ci,
    bootstrap_sharpe_ci,
    circular_block_indices,
    diebold_mariano,
    grouped_mean_tstat,
    jobson_korkie_memmel,
    mean_difference_t,
    pairwise_diebold_mariano,
    two_proportion_test,
)


def test_grouped_mean_tstat_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="values and groups must have the same length"):
        grouped_mean_tstat(np.array([1.0, 2.0]), np.array(["a"]), target=0.0)


def test_grouped_mean_tstat_all_nonfinite_returns_zero_groups() -> None:
    mu, t, p, n = grouped_mean_tstat(np.array([np.nan, np.inf]), np.array(["a", "b"]))
    assert np.isnan(mu) and np.isnan(t) and np.isnan(p)
    assert n == 0


def test_diebold_mariano_tie_preferred() -> None:
    loss = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = diebold_mariano(loss, loss.copy(), name_a="m1", name_b="m2")
    assert result.mean_loss_diff == 0.0
    assert result.preferred == "tie"


@pytest.mark.parametrize(
    ("alternative", "expect_lt_half"),
    [("greater", True), ("less", False), ("two-sided", None)],
)
def test_mean_difference_t_alternative_directions(
    alternative: str, expect_lt_half: bool | None
) -> None:
    t, p = mean_difference_t(0.05, 30, sd=0.1, alternative=alternative)
    assert np.isfinite(t) and 0.0 <= p <= 1.0
    if expect_lt_half is True:
        assert p < 0.5
    elif expect_lt_half is False:
        assert p > 0.5


def test_mean_difference_t_invalid_scale_and_alternative() -> None:
    assert all(math.isnan(v) for v in mean_difference_t(0.05, 30, sd=0.0))
    assert all(math.isnan(v) for v in mean_difference_t(0.05, 30, sd=float("inf")))
    with pytest.raises(ValueError, match="alternative must be"):
        mean_difference_t(0.05, 30, sd=0.1, alternative="up")


def test_two_proportion_test_sparse_uses_fisher() -> None:
    stat, p = two_proportion_test(2, 5, 4, 5, alternative="two-sided")
    assert np.isfinite(stat) and 0.0 <= p <= 1.0  # Fisher odds ratio + exact p


@pytest.mark.parametrize(
    ("alternative", "expect_lt_half"),
    [("greater", True), ("less", False)],
)
def test_two_proportion_test_large_sample_directions(
    alternative: str, expect_lt_half: bool
) -> None:
    stat, p = two_proportion_test(60, 100, 30, 100, alternative=alternative)
    assert np.isfinite(stat) and 0.0 <= p <= 1.0
    if expect_lt_half:
        assert p < 0.5
    else:
        assert p > 0.5


def test_two_proportion_test_invalid_counts_and_alternative() -> None:
    assert all(math.isnan(v) for v in two_proportion_test(0, 0, 1, 10))
    assert all(math.isnan(v) for v in two_proportion_test(11, 10, 1, 10))
    with pytest.raises(ValueError, match="alternative must be"):
        two_proportion_test(1, 10, 1, 10, alternative="up")


def test_circular_block_indices_block_shapes() -> None:
    rng = np.random.default_rng(3)
    assert circular_block_indices(0, 4, rng).size == 0
    small = circular_block_indices(10, 3, rng)
    assert small.shape == (10,)
    assert small.min() >= 0 and small.max() < 10
    wide = circular_block_indices(4, 10, rng)
    assert wide.shape == (4,)
    exact = circular_block_indices(6, 6, rng)
    assert exact.shape == (6,)
    with pytest.raises(ValueError, match="block must be >= 1"):
        circular_block_indices(5, 0, rng)
    with pytest.raises(ValueError, match="n must be non-negative"):
        circular_block_indices(-1, 2, rng)


def test_bootstrap_mean_ci_explicit_block_deterministic() -> None:
    x = np.linspace(-0.02, 0.03, 40)
    lo1, hi1, mean1 = bootstrap_mean_ci(x, n_boot=200, block=5, seed=11)
    lo2, hi2, mean2 = bootstrap_mean_ci(x, n_boot=200, block=5, seed=11)
    assert (lo1, hi1, mean1) == (lo2, hi2, mean2)
    assert np.isfinite(lo1) and np.isfinite(hi1)
    assert lo1 <= mean1 <= hi1


def test_bootstrap_sharpe_ci_explicit_block_and_degenerate() -> None:
    rng = np.random.default_rng(5)
    r = rng.normal(0.0005, 0.01, size=60)
    lo, hi, point = bootstrap_sharpe_ci(r, n_boot=200, block=7, seed=13)
    assert np.isfinite(lo) and np.isfinite(hi) and np.isfinite(point)
    assert lo <= point <= hi
    # Zero returns: zero dispersion → Sharpe is exactly zero, not NaN/inf.
    lo_c, hi_c, point_c = bootstrap_sharpe_ci(np.zeros(30), n_boot=50, block=5, seed=1)
    assert point_c == 0.0
    assert lo_c == 0.0 and hi_c == 0.0


def test_jobson_korkie_memmel_valid_and_degenerate() -> None:
    theta, p = jobson_korkie_memmel(1.0, 0.5, 120, 0.3)
    assert np.isfinite(theta) and 0.0 <= p <= 1.0
    assert theta > 0  # higher Sharpe prefers a
    assert all(math.isnan(v) for v in jobson_korkie_memmel(1.0, 0.5, 120, 1.5))
    assert all(math.isnan(v) for v in jobson_korkie_memmel(1.0, 0.5, 120, float("nan")))
    # corr=1 with equal Sharpes → zero variance denominator → fail closed.
    assert all(math.isnan(v) for v in jobson_korkie_memmel(0.8, 0.8, 120, 1.0))


def test_pairwise_diebold_mariano_branches() -> None:
    with pytest.raises(ValueError, match="loss series must align"):
        pairwise_diebold_mariano({"a": np.zeros(6), "b": np.zeros(5)})

    short = pairwise_diebold_mariano({"a": np.zeros(4), "b": np.ones(4)}, include_e_process=True)
    assert len(short) == 1
    row = short[0]
    assert row["preferred"] == "inconclusive"
    assert np.isnan(float(row["e_final"]))
    assert row["e_reject"] is False
    assert row["e_n"] == 4

    rng = np.random.default_rng(2)
    long = pairwise_diebold_mariano(
        {"a": rng.normal(0, 0.01, 60), "b": rng.normal(0.001, 0.01, 60)},
        include_e_process=True,
    )
    assert len(long) == 1
    assert {"e_final", "e_reject", "e_n"}.issubset(long[0].keys())
    assert isinstance(long[0]["e_reject"], bool)


def test_benjamini_hochberg_cutoff_matches_smallest_rejection() -> None:
    p = np.array([0.001, 0.02, 0.5, 0.9])
    reject, cutoff = benjamini_hochberg(p, alpha=0.05)
    assert reject.tolist() == [True, True, False, False]
    assert cutoff == pytest.approx(0.025)
