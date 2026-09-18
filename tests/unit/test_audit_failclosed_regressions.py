"""Fail-closed regressions for the 2026-09-17 audit of the overnight wave.

Pins: no fabricated bound floors in the H-table, no silent momentum fallback on
a stale ranker, no zero-reset on ``w_prev`` shape mismatch, no malformed
evidence read as absent in the honesty helpers, and the optional candle family
being verifiable (not required). Research diagnostic only; never live Sharpe.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.pipeline.forecast import _align_w_prev
from quant_fund.research.agent import _build_hypotheses
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    REQUIRED_BENCHMARK_FAMILIES,
    candle_order_book_claim_honesty_errors,
    candle_order_book_dgp_data_source_honesty_errors,
    gap_finite_rate_honesty_errors,
    join_coverage_honesty_errors,
    kyle_ofi_join_coverage_honesty_errors,
    mean_microprice_weight_balance_honesty_errors,
    mean_tob_size_share_honesty_errors,
    northset_log_size_slope_honesty_errors,
    northset_metrics_required_finite_ok_rates_honesty_errors,
    northset_qlike_means_honesty_errors,
    northset_queue_priority_bid_ask_pair_honesty_errors,
    northset_receipt_dgp_data_source_honesty_errors,
    northset_session_book_snaps_honesty_errors,
    northset_shape_columns_ensured_book_panel_path_honesty_errors,
    northset_sweep_evidence_blob_honesty_errors,
    northset_top_level_claim_honesty_errors,
)


def _hyps(**families: object) -> dict[str, object]:
    return {h.id: h for h in _build_hypotheses(dict(families), [])}


# --- money/evidence path -----------------------------------------------------


def test_align_w_prev_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="w_prev length"):
        _align_w_prev(["A", "B"], np.array([0.1]))
    assert _align_w_prev(["A"], np.array([0.25])).tolist() == [0.25]


def test_broker_from_state_requires_cash_key(tmp_path: Path) -> None:
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    with pytest.raises(ValueError, match="required 'cash'"):
        SimulatedBroker.from_state(cfg, {"initial_cash": 100_000.0, "shares": {"A": 5.0}})


def test_h10_mints_unavailable_not_fabricated_floor() -> None:
    hyps = _hyps(jackknife_plus={"coverage": 0.85})
    h10 = hyps["H10_jackknife_coverage"]
    assert h10.meets_floor is None
    assert "unavailable" in h10.decision.lower()


def test_h19_mints_unavailable_not_assumed_alpha() -> None:
    hyps = _hyps(conformal_rank={"fdr": 0.15, "dgp": "panel"})
    h19 = hyps["H19_conformal_rank"]
    assert h19.meets_floor is None
    assert "unavailable" in h19.decision.lower()


def test_h41_h42_mint_unavailable_without_stability_floor() -> None:
    hyps = _hyps(
        northset={
            "sweep_reject_fold_positive_fraction": 0.9,
            "sweep_follow_fold_positive_fraction": 0.9,
        }
    )
    for hid in ("H41_northset_reject_stability", "H42_northset_follow_stability"):
        assert hyps[hid].meets_floor is None
        assert "unavailable" in hyps[hid].decision.lower()


# --- honesty helpers ---------------------------------------------------------


def test_queue_priority_nan_ask_companion_flagged() -> None:
    assert northset_queue_priority_bid_ask_pair_honesty_errors(
        {"mean_queue_priority_proxy": 0.5, "mean_ask_queue_priority_proxy": float("nan")}
    ) == ["mean_ask_queue_priority_proxy_nan_while_bid_proxy_finite"]
    assert (
        northset_queue_priority_bid_ask_pair_honesty_errors(
            {"mean_queue_priority_proxy": 0.5, "mean_ask_queue_priority_proxy": 0.4}
        )
        == []
    )


def test_sweep_reject_fdr_non_bool_flagged() -> None:
    errs = northset_sweep_evidence_blob_honesty_errors(
        {"sweep_evidence": {"event_studies": [{"sample_adequate": False, "reject_fdr": "true"}]}}
    )
    assert "sweep_evidence_event_studies_0_reject_fdr_not_bool" in errs


def test_claim_without_research_only_flagged() -> None:
    assert "northset_research_only_missing_or_false" in northset_top_level_claim_honesty_errors(
        {"claim": "research_diagnostic_only"}
    )
    assert "candle_research_only_missing_or_false" in candle_order_book_claim_honesty_errors(
        {"family": "candle_order_book", "claim": "research_diagnostic_only"}
    )


def test_session_book_snaps_non_numeric_flagged() -> None:
    assert northset_session_book_snaps_honesty_errors({"mean_session_book_snaps": "oops"}) == [
        "mean_session_book_snaps_non_numeric"
    ]


def test_dgp_data_source_non_str_flagged() -> None:
    assert "data_source_non_str" in northset_receipt_dgp_data_source_honesty_errors(
        {"family": "northset", "data_source": 123}
    )
    assert "data_source_non_str" in candle_order_book_dgp_data_source_honesty_errors(
        {"family": "candle_order_book", "data_source": 123}
    )


def test_kyle_join_coverage_non_numeric_flagged() -> None:
    assert kyle_ofi_join_coverage_honesty_errors({"kyle_ofi": {"join_coverage": "abc"}}) == [
        "kyle_ofi_join_coverage_non_numeric"
    ]


def test_shape_path_non_str_flagged() -> None:
    assert northset_shape_columns_ensured_book_panel_path_honesty_errors(
        {"family": "northset", "shape_columns_ensured": True, "book_panel_path": 123}
    ) == ["book_panel_path_non_string_while_shape_columns_ensured"]


def test_log_slope_and_qlike_non_numeric_flagged() -> None:
    assert "mean_bid_log_size_slope_non_numeric" in northset_log_size_slope_honesty_errors(
        {"mean_bid_log_size_slope": "oops"}
    )
    assert "parkinson_qlike_vs_cc_non_numeric" in northset_qlike_means_honesty_errors(
        {"parkinson_qlike_vs_cc": "oops"}
    )


def test_rate_and_coverage_helpers_reject_bool_coercion() -> None:
    """Bools are ints in Python: True must not pass as 1.0 in unit-interval gates."""
    assert northset_metrics_required_finite_ok_rates_honesty_errors(
        {
            "metrics_required_finite_ok": True,
            "depth_shape_finite_rate": True,
            "concentration_top_finite_rate": 0.5,
            "queue_priority_finite_rate": 0.5,
            "side_notional_finite_rate": 0.5,
            "tob_size_share_finite_rate": 0.5,
            "structure_finite_rate": 0.5,
        }
    ) == ["depth_shape_finite_rate_non_numeric_while_metrics_required_finite_ok"]
    assert join_coverage_honesty_errors({"join_coverage": True}) == ["join_coverage_non_numeric"]
    assert gap_finite_rate_honesty_errors({"gap_finite_rate": True}) == [
        "gap_finite_rate_non_numeric"
    ]
    assert mean_microprice_weight_balance_honesty_errors(
        {"mean_microprice_weight_balance": "0.5"}
    ) == ["mean_microprice_weight_balance_non_numeric"]
    assert mean_tob_size_share_honesty_errors({"mean_tob_size_share": True}) == [
        "mean_tob_size_share_non_numeric"
    ]


# --- wiring ------------------------------------------------------------------


def test_candle_family_is_optional_and_verifiable() -> None:
    assert "candle_order_book" in OPTIONAL_BENCHMARK_FAMILIES
    assert "candle_order_book" not in REQUIRED_BENCHMARK_FAMILIES
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "OPTIONAL_BENCHMARK_FAMILIES" in src


def test_candle_depth_shape_rate_nan_when_shape_cols_absent() -> None:
    from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
    from quant_fund.microstructure.bench import bench_candle_order_book
    from quant_fund.microstructure.book_metrics import DEPTH_SHAPE_FIELDS
    from quant_fund.microstructure.synthetic_lob import synthesize_l2_from_bars

    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()
    panel = synthesize_l2_from_bars(bars, depth=5, seed=5)
    drop = [c for c in DEPTH_SHAPE_FIELDS if c in panel.columns]
    thin = panel.drop(drop) if drop else panel
    receipt = bench_candle_order_book(bars, book=thin)
    assert math.isnan(float(receipt["depth_shape_finite_rate"]))
