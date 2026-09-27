import numpy as np
import pytest

from quant_fund.models.conformal_rank import (
    bench_conformal_topk,
    conformal_selection_pvalues,
    conformal_topk,
    oracle_topk_mask,
    planted_rank_panel,
    within_date_percentiles,
)

SEEDED = 18
SLACK = 0.10


def test_planted_fdr_or_oracle_coverage() -> None:
    row = bench_conformal_topk(
        n_dates=80,
        n_names=40,
        k=5,
        alpha=0.20,
        noise=0.20,
        seed=SEEDED,
        guarantee="fdr",
    )
    alpha = float(row["alpha"])
    fdr_ok = float(row["fdr"]) <= alpha + SLACK
    cov_ok = float(row["coverage"]) >= 1.0 - alpha - SLACK
    assert fdr_ok or cov_ok
    assert float(row["set_size"]) > 0.0
    assert int(row["n_dates"]) >= 20
    assert row["guarantee"] == "fdr"
    assert float(row["seed"]) == float(SEEDED)


def test_set_coverage_on_planted_scores() -> None:
    row = bench_conformal_topk(
        n_dates=80,
        n_names=40,
        k=5,
        alpha=0.20,
        noise=0.20,
        seed=SEEDED,
        guarantee="set_coverage",
    )
    alpha = float(row["alpha"])
    assert float(row["coverage"]) >= 1.0 - alpha - SLACK
    assert float(row["set_size"]) >= 5.0
    assert row["guarantee"] == "set_coverage"


def test_groups_by_date_never_stacks() -> None:
    """Date-level score drift must not make only late dates selectable."""
    scores, labels, dates = planted_rank_panel(
        n_dates=40, n_names=24, noise=0.15, seed=SEEDED, date_shift=8.0
    )
    result = conformal_topk(scores, labels, k=4, alpha=0.20, dates=dates, guarantee="fdr")
    assert result.n_dates > 0
    unique = np.unique(dates)
    n_cal = len(unique) - result.n_dates
    test_dates = unique[n_cal:]
    mid = n_cal + len(test_dates) // 2
    early = np.isin(dates, unique[n_cal:mid])
    late = np.isin(dates, unique[mid:])
    n_early = int(result.selected[early].sum())
    n_late = int(result.selected[late].sum())
    # Stacking raw scores would pick only the drifted late names.
    assert n_early > 0
    assert n_late > 0


def test_within_date_percentiles_not_global() -> None:
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([100.0, 101.0, 102.0, 103.0])
    pa = within_date_percentiles(a)
    pb = within_date_percentiles(b)
    assert pa == pytest.approx(pb)
    stacked = within_date_percentiles(np.concatenate([a, b]))
    assert stacked[0] < stacked[-1]
    assert stacked[0] != pytest.approx(pa[0])


def test_oracle_topk_and_ranks() -> None:
    y = np.array([0.1, 0.9, 0.3, 0.8])
    mask = oracle_topk_mask(y, k=2, higher_is_better=True)
    assert list(mask) == [False, True, False, True]
    ranks = np.array([4.0, 1.0, 3.0, 2.0])
    assert np.array_equal(oracle_topk_mask(ranks, k=2, higher_is_better=False), mask)


def test_conformal_pvalues_monotone_in_score() -> None:
    null = np.linspace(0.0, 0.8, 40)
    p = conformal_selection_pvalues(np.array([0.1, 0.9]), null)
    assert p[1] < p[0]
    assert 0.0 < p[1] <= 1.0


def test_bench_keys_no_sharpe() -> None:
    row = bench_conformal_topk(seed=SEEDED)
    assert {"set_size", "fdr", "coverage", "n_dates"} <= set(row)
    assert all("sharpe" not in str(key).lower() for key in row)


def test_rejects_bad_alpha_and_k() -> None:
    scores, labels, dates = planted_rank_panel(8, 6, 0.2, seed=0)
    with pytest.raises(ValueError):
        conformal_topk(scores, labels, k=2, alpha=0.0, dates=dates)
    with pytest.raises(ValueError):
        conformal_topk(scores, labels, k=0, alpha=0.2, dates=dates)


def test_larger_alpha_selects_more() -> None:
    scores, labels, dates = planted_rank_panel(60, 30, 0.25, seed=SEEDED)
    tight = conformal_topk(scores, labels, k=4, alpha=0.10, dates=dates, guarantee="fdr")
    loose = conformal_topk(scores, labels, k=4, alpha=0.30, dates=dates, guarantee="fdr")
    assert loose.set_size >= tight.set_size - 1e-12
