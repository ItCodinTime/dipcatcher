"""Wave 39: cross_section date_ic / decile empty, mismatch, single-name, n<10 edges.

Research metrics only — live_pnl_claim=false; not a live edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.cross_section import date_ic_series, decile_portfolios

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def test_date_ic_empty_returns_nan_summary() -> None:
    empty = np.array([])
    out = date_ic_series(empty, empty, empty, min_names=5)
    assert out.n_dates == 0
    assert out.dates == []
    assert out.pearson.size == 0
    assert np.isnan(out.mean_pearson)
    assert np.isnan(out.t_pearson)
    assert np.isnan(out.icir_pearson)


def test_date_ic_length_mismatch_fail_closed() -> None:
    with pytest.raises(ValueError, match="align"):
        date_ic_series(np.array([1.0, 2.0]), np.array([1.0]), np.array([0, 1]))
    with pytest.raises(ValueError, match="align"):
        date_ic_series(np.array([1.0]), np.array([1.0, 2.0]), np.array([0]))
    with pytest.raises(ValueError, match="align"):
        date_ic_series(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0]))


def test_date_ic_single_name_dates_skipped() -> None:
    # One name per date — below min_names=5 → no kept dates, NaN summary.
    dates = np.arange(20)
    scores = np.linspace(-1, 1, 20)
    y = scores + 0.01
    out = date_ic_series(scores, y, dates, min_names=5)
    assert out.n_dates == 0
    assert out.dates == []
    assert np.isnan(out.mean_pearson)


def test_date_ic_mixed_single_and_enough_names() -> None:
    # Dates 0..4: 1 name each (skip); dates 5..14: 8 names (keep) → 10 ICs but <3? wait 10>=3
    single = np.arange(5)
    multi_dates = np.repeat(np.arange(5, 15), 8)
    dates = np.concatenate([single, multi_dates])
    scores = np.linspace(-1, 1, dates.size)
    y = scores + 0.02 * np.sin(np.arange(dates.size))
    out = date_ic_series(scores, y, dates, min_names=5)
    assert out.n_dates == 10
    assert all(str(d) not in {"0", "1", "2", "3", "4"} for d in map(str, out.dates))
    assert np.isfinite(out.mean_pearson)


def test_decile_empty_returns_nan() -> None:
    empty = np.array([])
    out = decile_portfolios(empty, empty, empty, n_buckets=5, min_names=5)
    assert out.dates == []
    assert out.long_short.size == 0
    assert all(np.isnan(m) for m in out.mean_returns)
    assert np.isnan(out.mean_ls)
    assert np.isnan(out.hit_rate)
    assert np.isnan(out.monotonicity)


def test_decile_length_mismatch_fail_closed() -> None:
    with pytest.raises(ValueError, match="align"):
        decile_portfolios(np.array([1.0, 2.0]), np.array([1.0]), np.array([0, 1]))
    with pytest.raises(ValueError, match="align"):
        decile_portfolios(np.array([1.0]), np.array([1.0]), np.array([0, 1]))


def test_decile_n_buckets_and_min_names_fail_closed() -> None:
    scores = np.array([1.0, 2.0, 3.0])
    y = np.array([0.1, 0.2, 0.3])
    dates = np.array([0, 0, 0])
    with pytest.raises(ValueError, match="n_buckets"):
        decile_portfolios(scores, y, dates, n_buckets=1)
    with pytest.raises(ValueError, match="min_names"):
        decile_portfolios(scores, y, dates, min_names=0)


def test_decile_n_less_than_10_skipped_when_min_names_high() -> None:
    # 9 names per date, min_names=10, n_buckets=5 → skip all (need max(10,5)=10).
    n_dates, n_names = 15, 9
    dates = np.repeat(np.arange(n_dates), n_names)
    rng = np.random.default_rng(39)
    scores = rng.normal(size=dates.size)
    y = scores + 0.1 * rng.normal(size=dates.size)
    out = decile_portfolios(scores, y, dates, n_buckets=5, min_names=10)
    assert out.dates == []
    assert out.long_short.size == 0
    assert np.isnan(out.mean_ls)


def test_decile_exactly_10_names_kept() -> None:
    n_dates, n_names = 12, 10
    dates = np.repeat(np.arange(n_dates), n_names)
    rng = np.random.default_rng(40)
    scores = rng.normal(size=dates.size)
    y = scores + 0.15 * rng.normal(size=dates.size)
    out = decile_portfolios(scores, y, dates, n_buckets=5, min_names=10)
    assert out.n_buckets == 5
    assert len(out.dates) == n_dates
    assert out.long_short.size == n_dates
    assert np.isfinite(out.mean_ls)


def test_decile_single_name_dates_skipped() -> None:
    dates = np.arange(30)
    scores = np.linspace(-1, 1, 30)
    y = scores
    out = decile_portfolios(scores, y, dates, n_buckets=5, min_names=5)
    assert out.dates == []
    assert np.isnan(out.mean_ls)
