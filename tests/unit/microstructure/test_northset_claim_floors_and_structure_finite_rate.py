"""Northset claim/floors honesty + structure_finite_rate."""

from __future__ import annotations

from quant_fund.research.catalog import (
    northset_shape_and_session_l2_floors_honesty_errors,
    northset_top_level_claim_honesty_errors,
    structure_finite_rate_honesty_errors,
)


def test_northset_claim_honesty() -> None:
    assert (
        northset_top_level_claim_honesty_errors(
            {"research_only": True, "claim": "research_diagnostic_only"}
        )
        == []
    )
    assert "northset_research_only_missing_or_false" in (
        northset_top_level_claim_honesty_errors({"research_only": False})
    )


def test_shape_and_session_l2_floors_honesty() -> None:
    assert (
        northset_shape_and_session_l2_floors_honesty_errors(
            {
                "session_l2_identity_floor": 0.99,
                "session_l2_identity_gate": "skipped",
                "queue_priority_finite_floor": 0.5,
                "depth_shape_finite_floor": None,
            }
        )
        == []
    )
    assert "tob_size_share_finite_floor_out_of_unit_interval" in (
        northset_shape_and_session_l2_floors_honesty_errors({"tob_size_share_finite_floor": 1.1})
    )


def test_structure_finite_rate_honesty() -> None:
    assert (
        structure_finite_rate_honesty_errors({"finite_rate_bid_size_concentration_top": 1.0}) == []
    )
    assert "finite_rate_ask_size_concentration_top_out_of_unit_interval" in (
        structure_finite_rate_honesty_errors({"finite_rate_ask_size_concentration_top": 1.2})
    )
