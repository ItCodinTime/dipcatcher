"""Soft-verify book_snaps >0; session means dispatcher fans into helpers."""

from __future__ import annotations

from quant_fund.research.catalog import (
    northset_session_book_snaps_honesty_errors,
    northset_session_means_honesty_errors,
)


def test_book_snaps_positive_when_finite() -> None:
    assert northset_session_book_snaps_honesty_errors({"mean_session_book_snaps": 8.0}) == []
    assert (
        northset_session_book_snaps_honesty_errors({"mean_session_book_snaps": float("nan")}) == []
    )
    assert northset_session_book_snaps_honesty_errors({"mean_session_book_snaps": 0.0}) == [
        "mean_session_book_snaps_non_positive_or_non_finite"
    ]
    assert northset_session_book_snaps_honesty_errors({"mean_session_book_snaps": -1.0}) == [
        "mean_session_book_snaps_non_positive_or_non_finite"
    ]


def test_dispatcher_empty_on_clean_blob() -> None:
    blob = {
        "mean_session_imbalance_mean": 0.1,
        "mean_session_close_micro_bps": 0.5,
        "mean_session_close_bid_depth": 1.0,
        "mean_session_close_ask_depth": 2.0,
        "mean_session_imbalance_std": 0.05,
        "mean_session_close_imbalance": -0.2,
        "mean_session_close_mid": 100.0,
        "mean_session_close_spread_bps": 4.0,
        "mean_session_spread_bps_mean": 3.0,
        "mean_session_book_snaps": 6.0,
        "n_session_book_rows": 6.0,
    }
    assert northset_session_means_honesty_errors(blob) == []


def test_dispatcher_collects_multiple_errors() -> None:
    blob = {
        "mean_session_close_imbalance": 2.0,  # out of [-1,1]
        "mean_session_book_snaps": 0.0,  # non-positive
        "mean_session_close_bid_depth": -1.0,  # negative
    }
    errs = northset_session_means_honesty_errors(blob)
    assert "mean_session_close_imbalance_out_of_unit_interval" in errs
    assert "mean_session_book_snaps_non_positive_or_non_finite" in errs
    assert "mean_session_close_bid_depth_negative_or_non_finite" in errs
