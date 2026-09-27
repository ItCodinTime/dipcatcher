"""Soft-verify session_ofi_sum_mean (finite-only) + session_book_vpin_mean ∈[0,1]."""

from __future__ import annotations

from quant_fund.research.catalog import (
    NORTHSET_SESSION_MEANS_HONESTY_HELPERS,
    northset_session_book_vpin_mean_honesty_errors,
    northset_session_means_honesty_errors,
    northset_session_ofi_sum_mean_honesty_errors,
)


def test_ofi_sum_mean_finite_only() -> None:
    assert northset_session_ofi_sum_mean_honesty_errors({"session_ofi_sum_mean": -12.5}) == []
    assert (
        northset_session_ofi_sum_mean_honesty_errors({"session_ofi_sum_mean": float("nan")}) == []
    )
    assert northset_session_ofi_sum_mean_honesty_errors({"session_ofi_sum_mean": float("inf")}) == [
        "session_ofi_sum_mean_non_finite"
    ]


def test_book_vpin_mean_unit_interval() -> None:
    assert northset_session_book_vpin_mean_honesty_errors({"session_book_vpin_mean": 0.0}) == []
    assert northset_session_book_vpin_mean_honesty_errors({"session_book_vpin_mean": 1.0}) == []
    assert (
        northset_session_book_vpin_mean_honesty_errors({"session_book_vpin_mean": float("nan")})
        == []
    )
    assert northset_session_book_vpin_mean_honesty_errors({"session_book_vpin_mean": 1.01}) == [
        "session_book_vpin_mean_out_of_unit_interval"
    ]


def test_dispatcher_includes_ofi_and_vpin() -> None:
    names = {fn.__name__ for fn in NORTHSET_SESSION_MEANS_HONESTY_HELPERS}
    assert "northset_session_ofi_sum_mean_honesty_errors" in names
    assert "northset_session_book_vpin_mean_honesty_errors" in names
    # clean blob with ofi + vpin
    assert (
        northset_session_means_honesty_errors(
            {
                "session_ofi_sum_mean": 3.0,
                "session_book_vpin_mean": 0.4,
                "mean_session_imbalance_mean": 0.0,
                "mean_session_close_micro_bps": 0.0,
                "mean_session_close_bid_depth": 1.0,
                "mean_session_close_ask_depth": 1.0,
                "mean_session_imbalance_std": 0.1,
                "mean_session_close_imbalance": 0.0,
                "mean_session_close_mid": 10.0,
                "mean_session_close_spread_bps": 1.0,
                "mean_session_spread_bps_mean": 1.0,
                "mean_session_book_snaps": 4.0,
                "n_session_book_rows": 4.0,
            }
        )
        == []
    )
