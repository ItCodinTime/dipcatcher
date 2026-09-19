"""Soft-verify: close_micro finite ⇒ close_mid finite (paired honesty)."""

from __future__ import annotations

from quant_fund.research.catalog import (
    northset_session_close_mid_micro_pair_honesty_errors,
)


def test_pair_ok_when_both_finite_or_both_nan() -> None:
    assert (
        northset_session_close_mid_micro_pair_honesty_errors(
            {"mean_session_close_micro_bps": 0.5, "mean_session_close_mid": 100.0}
        )
        == []
    )
    assert (
        northset_session_close_mid_micro_pair_honesty_errors(
            {
                "mean_session_close_micro_bps": float("nan"),
                "mean_session_close_mid": float("nan"),
            }
        )
        == []
    )
    # micro NaN, mid finite — skip (asymmetric; only require mid when micro finite)
    assert (
        northset_session_close_mid_micro_pair_honesty_errors(
            {"mean_session_close_micro_bps": float("nan"), "mean_session_close_mid": 100.0}
        )
        == []
    )


def test_pair_flags_mid_nan_while_micro_finite() -> None:
    assert northset_session_close_mid_micro_pair_honesty_errors(
        {"mean_session_close_micro_bps": 0.1, "mean_session_close_mid": float("nan")}
    ) == ["mean_session_close_mid_nan_while_micro_finite"]
    assert northset_session_close_mid_micro_pair_honesty_errors(
        {"mean_session_close_micro_bps": 0.1}  # mid missing
    ) == ["mean_session_close_mid_nan_while_micro_finite"]


def test_pair_flags_mid_inf_while_micro_finite() -> None:
    assert northset_session_close_mid_micro_pair_honesty_errors(
        {"mean_session_close_micro_bps": 0.1, "mean_session_close_mid": float("inf")}
    ) == ["mean_session_close_mid_nan_while_micro_finite"]
