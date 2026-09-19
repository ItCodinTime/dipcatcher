"""Soft-verify: session close/path spread_bps ≥0 when finite (zero allowed)."""

from __future__ import annotations

from quant_fund.research.catalog import (
    northset_session_close_spread_bps_honesty_errors,
    northset_session_spread_bps_mean_honesty_errors,
)


def test_close_spread_bps_nonnegative_allows_zero() -> None:
    assert (
        northset_session_close_spread_bps_honesty_errors({"mean_session_close_spread_bps": 4.0})
        == []
    )
    # Zero-spread possible (vendor_book_map otherwise(0.0) / locked book)
    assert (
        northset_session_close_spread_bps_honesty_errors({"mean_session_close_spread_bps": 0.0})
        == []
    )
    assert (
        northset_session_close_spread_bps_honesty_errors(
            {"mean_session_close_spread_bps": float("nan")}
        )
        == []
    )
    assert northset_session_close_spread_bps_honesty_errors(
        {"mean_session_close_spread_bps": -0.01}
    ) == ["mean_session_close_spread_bps_negative_or_non_finite"]


def test_path_spread_bps_mean_nonnegative() -> None:
    assert (
        northset_session_spread_bps_mean_honesty_errors({"mean_session_spread_bps_mean": 3.5}) == []
    )
    assert (
        northset_session_spread_bps_mean_honesty_errors({"mean_session_spread_bps_mean": 0.0}) == []
    )
    assert (
        northset_session_spread_bps_mean_honesty_errors(
            {"mean_session_spread_bps_mean": float("nan")}
        )
        == []
    )
    assert northset_session_spread_bps_mean_honesty_errors(
        {"mean_session_spread_bps_mean": -1.0}
    ) == ["mean_session_spread_bps_mean_negative_or_non_finite"]


def test_spread_bps_rejects_inf() -> None:
    assert northset_session_close_spread_bps_honesty_errors(
        {"mean_session_close_spread_bps": float("inf")}
    ) == ["mean_session_close_spread_bps_negative_or_non_finite"]
    assert northset_session_spread_bps_mean_honesty_errors(
        {"mean_session_spread_bps_mean": float("-inf")}
    ) == ["mean_session_spread_bps_mean_negative_or_non_finite"]
