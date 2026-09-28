"""Benchmark alignment + POSIX RAM honesty in hedge_lab.

The IR benchmark pair must be date-aligned on the common calendar (a gap
in the benchmark is a multi-day return matched by the book's own multi-day
return), never a positional tail overlap. Benchmark bars missing on a book
date must drop the pair, not fabricate a 0.0 buy-hold return.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from quant_fund.hedge_lab.lightspeed_book import _bench_window_returns
from quant_fund.hedge_lab.resources import physical_memory
from quant_fund.hedge_lab.runner import _aligned_book_and_benchmark

_D = [datetime(2024, 1, i) for i in (2, 3, 4, 5, 8)]


def _equity(nav: list[float], dates: list[datetime] | None = None) -> pl.DataFrame:
    return pl.DataFrame({"event_time": dates or _D[: len(nav)], "nav": nav})


def _bench(close: list[float], dates: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({"event_time": dates, "security_id": ["SPY"] * len(dates), "close": close})


def test_aligned_pair_computes_both_legs_on_joined_calendar() -> None:
    # Benchmark misses 2024-01-04: both legs span 01-03 -> 01-05 as one return.
    bench_dates = [_D[0], _D[1], _D[3], _D[4]]
    aligned = _aligned_book_and_benchmark(
        _bench([100.0, 101.0, 102.0, 103.0], bench_dates),
        _equity([1000.0, 1010.0, 999.0, 1030.0, 1020.0]),
        "SPY",
    )
    assert aligned is not None
    book_r, bench_r = aligned
    assert book_r.shape == bench_r.shape == (3,)
    # Book leg over the gap uses the book's own joined dates — 1010 -> 1030,
    # not a positional tail of the per-day equity returns.
    np.testing.assert_allclose(book_r, [1010 / 1000 - 1.0, 1030 / 1010 - 1.0, 1020 / 1030 - 1.0])
    np.testing.assert_allclose(bench_r, [101 / 100 - 1.0, 102 / 101 - 1.0, 103 / 102 - 1.0])


def test_aligned_pair_none_without_benchmark() -> None:
    assert _aligned_book_and_benchmark(_bench([], []), _equity([1.0, 1.1, 1.2]), "") is None


def test_aligned_pair_fails_closed_on_nonfinite_nav() -> None:
    aligned = _aligned_book_and_benchmark(
        _bench([100.0] * 5, _D), _equity([1000.0, np.inf, 999.0, 1030.0, 1020.0]), "SPY"
    )
    assert aligned is None


def test_bench_window_returns_drops_unpaired_dates() -> None:
    bench_dates = [_D[0], _D[1], _D[3]]  # benchmark has no bar on _D[4]
    dates, rets = _bench_window_returns(
        book_dates=[_D[1], _D[3], _D[4]],
        bench_dates=bench_dates,
        bench_closes=np.asarray([50.0, 55.0, 60.0]),
    )
    # Only the _D[1]->_D[3] pair has both bars; the _D[3]->_D[4] leg is dropped.
    assert rets.tolist() == [60.0 / 55.0 - 1.0]
    assert len(dates) == len(rets) + 1


def test_bench_window_returns_empty_when_no_overlap() -> None:
    dates, rets = _bench_window_returns([_D[0], _D[1]], [_D[3], _D[4]], np.asarray([1.0, 1.1]))
    assert dates == [] and rets.size == 0


def test_physical_memory_reports_a_measured_pair() -> None:
    total, available = physical_memory()
    assert total > 0
    assert 0 < available <= total
