"""Wave 16: conformal_rank edge extremes (empty dates, k/alpha bounds, planted honesty)."""

from __future__ import annotations

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


def test_empty_dates_raises() -> None:
    with pytest.raises(ValueError, match="at least two dates"):
        conformal_topk(
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            k=1,
            alpha=0.2,
            dates=[],
        )


def test_single_date_raises() -> None:
    scores = np.linspace(0.0, 1.0, 6)
    labels = scores.copy()
    dates = [0] * 6
    with pytest.raises(ValueError, match="at least two dates"):
        conformal_topk(scores, labels, k=2, alpha=0.2, dates=dates)


def test_bad_alpha_bounds_raise() -> None:
    scores, labels, dates = planted_rank_panel(8, 6, 0.2, seed=0)
    for alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            conformal_topk(scores, labels, k=2, alpha=alpha, dates=dates)


def test_bad_k_raises() -> None:
    scores, labels, dates = planted_rank_panel(8, 6, 0.2, seed=1)
    for k in (0, -1, -10):
        with pytest.raises(ValueError, match="k must be"):
            conformal_topk(scores, labels, k=k, alpha=0.2, dates=dates)


def test_bad_cal_frac_and_guarantee_raise() -> None:
    scores, labels, dates = planted_rank_panel(8, 6, 0.2, seed=2)
    with pytest.raises(ValueError, match="cal_frac"):
        conformal_topk(scores, labels, k=2, alpha=0.2, dates=dates, cal_frac=0.0)
    with pytest.raises(ValueError, match="cal_frac"):
        conformal_topk(scores, labels, k=2, alpha=0.2, dates=dates, cal_frac=1.0)
    with pytest.raises(ValueError, match="guarantee"):
        conformal_topk(
            scores,
            labels,
            k=2,
            alpha=0.2,
            dates=dates,
            guarantee="not_a_mode",  # type: ignore[arg-type]
        )


def test_length_mismatch_raises() -> None:
    scores, labels, dates = planted_rank_panel(6, 4, 0.2, seed=3)
    with pytest.raises(ValueError, match="align"):
        conformal_topk(scores[:-1], labels, k=2, alpha=0.2, dates=dates)
    with pytest.raises(ValueError, match="align"):
        conformal_topk(scores, labels, k=2, alpha=0.2, dates=dates[:-1])


def test_k_greater_than_n_names_clips_oracle() -> None:
    """oracle_topk_mask clips to n_finite; conformal_topk still runs."""
    scores, labels, dates = planted_rank_panel(10, 5, 0.15, seed=4)
    # k=50 >> 5 names per date
    result = conformal_topk(scores, labels, k=50, alpha=0.25, dates=dates, guarantee="fdr")
    assert result.k == 50
    assert result.n_dates >= 1
    assert np.isfinite(result.set_size)
    # Every finite name is in oracle top-k when k >= n
    for d in np.unique(dates):
        mask = dates == d
        ora = oracle_topk_mask(labels[mask], k=50, higher_is_better=True)
        assert int(ora.sum()) == int(mask.sum())


def test_oracle_topk_k_le_zero_or_all_nan() -> None:
    y = np.array([0.1, 0.9, 0.3])
    assert not oracle_topk_mask(y, k=0).any()
    assert not oracle_topk_mask(y, k=-2).any()
    assert not oracle_topk_mask(np.array([np.nan, np.nan]), k=1).any()


def test_within_date_percentiles_all_nan() -> None:
    out = within_date_percentiles(np.array([np.nan, np.nan, np.nan]))
    assert out.size == 3
    assert np.isnan(out).all()


def test_conformal_pvalues_empty_null_are_ones() -> None:
    p = conformal_selection_pvalues(np.array([0.1, 0.9]), np.asarray([], dtype=float))
    assert np.allclose(p, 1.0)


def test_planted_edge_low_noise_coverage_honesty() -> None:
    """Near-perfect scores: set_coverage should be high; we record not over-claim."""
    row = bench_conformal_topk(
        n_dates=60,
        n_names=30,
        k=4,
        alpha=0.20,
        noise=0.01,
        seed=18,
        guarantee="set_coverage",
    )
    assert row["guarantee"] == "set_coverage"
    cov = float(row["coverage"])
    assert 0.0 <= cov <= 1.0
    # Strong planted edge → coverage near nominal (honesty: allow slack, no Sharpe)
    assert cov >= 1.0 - float(row["alpha"]) - 0.10
    assert float(row["set_size"]) >= 4.0
    assert all("sharpe" not in str(k).lower() for k in row)


def test_planted_fdr_edge_finite_and_no_sharpe() -> None:
    row = bench_conformal_topk(
        n_dates=60,
        n_names=30,
        k=4,
        alpha=0.20,
        noise=0.05,
        seed=21,
        guarantee="fdr",
    )
    assert np.isfinite(float(row["fdr"])) or np.isnan(float(row["fdr"]))
    assert float(row["set_size"]) >= 0.0
    assert int(row["n_dates"]) >= 10
    assert all("sharpe" not in str(k).lower() for k in row)
    # Forbidden portfolio keys must stay out of the bench dict
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav")
    assert all(not any(f in str(k).lower() for f in forbidden) for k in row)


def test_high_noise_does_not_claim_perfect_fdr() -> None:
    """Honesty: noisy planted edge is not asserted as FDR ≤ α without slack."""
    row = bench_conformal_topk(
        n_dates=40,
        n_names=20,
        k=3,
        alpha=0.10,
        noise=2.0,
        seed=99,
        guarantee="fdr",
    )
    fdr = float(row["fdr"])
    # Only require finite reporting — do NOT claim fdr <= alpha on this path
    assert np.isfinite(fdr)
    assert 0.0 <= fdr <= 1.0
