"""Soft-verify: |session_ofi_sum_mean| ≤ mean_session_ofi_abs_sum; never equate VPIN to ratio-of-means."""

from __future__ import annotations

from quant_fund.research.catalog import (
    NORTHSET_SESSION_MEANS_HONESTY_HELPERS,
    northset_session_ofi_abs_dominates_sum_honesty_errors,
)


def test_ofi_abs_dominates_ok_and_flags() -> None:
    assert (
        northset_session_ofi_abs_dominates_sum_honesty_errors(
            {"session_ofi_sum_mean": -3.0, "mean_session_ofi_abs_sum": 5.0}
        )
        == []
    )
    assert (
        northset_session_ofi_abs_dominates_sum_honesty_errors(
            {"session_ofi_sum_mean": 2.0, "mean_session_ofi_abs_sum": 2.0}
        )
        == []
    )
    assert (
        northset_session_ofi_abs_dominates_sum_honesty_errors(
            {"session_ofi_sum_mean": float("nan"), "mean_session_ofi_abs_sum": 1.0}
        )
        == []
    )
    assert northset_session_ofi_abs_dominates_sum_honesty_errors(
        {"session_ofi_sum_mean": 5.0, "mean_session_ofi_abs_sum": 4.0}
    ) == ["session_ofi_abs_sum_less_than_abs_ofi_sum_mean"]


def test_dispatcher_includes_dominates_helper() -> None:
    names = {fn.__name__ for fn in NORTHSET_SESSION_MEANS_HONESTY_HELPERS}
    assert "northset_session_ofi_abs_dominates_sum_honesty_errors" in names


def test_vpin_ratio_of_means_not_forced_equal() -> None:
    """Honest: VPIN mean need not equal |ofi_sum_mean|/ofi_abs_mean (Jensen)."""
    # Clean receipt where ratio-of-means would be 0.5 but we do not soft-fail VPIN
    blob = {
        "session_ofi_sum_mean": 2.0,
        "mean_session_ofi_abs_sum": 4.0,
        "session_book_vpin_mean": 0.7,  # ≠ 0.5; still honest unit interval
    }
    assert northset_session_ofi_abs_dominates_sum_honesty_errors(blob) == []
