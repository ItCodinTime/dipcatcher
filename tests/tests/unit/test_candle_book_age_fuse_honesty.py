"""Candle-book PIT book-age honesty (mean/max ≥0; max ≥ mean) — fuse residual polish."""

from __future__ import annotations

from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure import bench_candle_order_book
from quant_fund.research.catalog import book_age_seconds_honesty_errors


def test_synth_candle_stamps_mean_and_max_book_age() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    receipt = bench_candle_order_book(bars, book=None, depth=5, seed=7, label="SYNTHETIC")
    assert "mean_book_age_seconds" in receipt
    assert "max_book_age_seconds" in receipt
    assert book_age_seconds_honesty_errors(receipt) == []


def test_candle_max_book_age_lt_mean_fail_closed() -> None:
    errs = book_age_seconds_honesty_errors(
        {"mean_book_age_seconds": 8.0, "max_book_age_seconds": 2.0}
    )
    assert "max_book_age_seconds_lt_mean" in errs


def test_candle_max_book_age_inf_fail_closed() -> None:
    errs = book_age_seconds_honesty_errors({"max_book_age_seconds": float("inf")})
    assert "max_book_age_seconds_non_finite_fail_closed" in errs
