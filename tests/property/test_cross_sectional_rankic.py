"""P3.4 rank-IC invariants: rank-based scoring must be permutation-stable."""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.research.cross_sectional import run_cross_sectional_bench


@settings(max_examples=30, deadline=None, derandomize=True)
@given(
    block=st.integers(min_value=0, max_value=10_000),
    perm=st.permutations(range(6)),
)
def test_rank_ic_invariant_to_asset_order(block: int, perm: tuple[int, ...]) -> None:
    """Permuting assets within every date leaves the Spearman series unchanged."""
    n_assets = 6
    n_dates = 24
    rng = np.random.default_rng(block)
    score = rng.normal(0.0, 1.0, size=(n_dates, n_assets))
    target = rng.normal(0.0, 1.0, size=(n_dates, n_assets))
    dates = np.repeat(np.arange(n_dates), n_assets)

    perm = list(perm)
    result = date_ic_series(score.reshape(-1), target.reshape(-1), dates, min_names=5)
    permuted = date_ic_series(
        score[:, perm].reshape(-1), target[:, perm].reshape(-1), dates, min_names=5
    )
    np.testing.assert_allclose(
        result.spearman, permuted.spearman, rtol=1e-12, atol=1e-12, equal_nan=True
    )


@settings(max_examples=20, deadline=None, derandomize=True)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_inversion_flips_every_ic(seed: int) -> None:
    """Negating the score negates every per-date Spearman value."""
    rng = np.random.default_rng(seed)
    score = rng.normal(0.0, 1.0, size=(40, 10))
    target = rng.normal(0.0, 1.0, size=(40, 10))
    dates = np.repeat(np.arange(40), 10)
    fwd = date_ic_series(score.reshape(-1), target.reshape(-1), dates, min_names=5)
    inv = date_ic_series((-score).reshape(-1), target.reshape(-1), dates, min_names=5)
    np.testing.assert_allclose(fwd.spearman, -inv.spearman, rtol=1e-12, atol=1e-12)


def test_shuffle_breaks_planted_signal_statistically() -> None:
    """Within-date permutation destroys the planted edge on every shard."""
    frame, _ = run_cross_sectional_bench(challengers=["shuffled"], horizons=(1,), seed=19)
    for row in frame.iter_rows(named=True):
        if row["shard"] in {"pure_noise", "regime_flip"}:
            continue
        assert abs(row["mean_spearman"]) < 0.15, row


def test_zero_ic_null_coverage() -> None:
    """On pure_noise + shuffled, the NW t-stat rejects at roughly nominal rate."""
    rejects = 0
    cells = 0
    for seed in range(40, 52):
        frame, _ = run_cross_sectional_bench(
            challengers=["shuffled"], horizons=(1,), n_dates=96, seed=seed
        )
        row = frame.filter(pl.col("shard") == "pure_noise").row(0, named=True)
        cells += 1
        if abs(row["t_spearman"]) > 1.96:
            rejects += 1
    assert rejects <= 3, f"null over-rejects: {rejects}/{cells}"
