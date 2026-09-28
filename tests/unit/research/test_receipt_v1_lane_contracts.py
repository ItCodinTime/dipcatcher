"""v1-sealed lane receipts get the same contract scrutiny as ``fleet_eval.v1``."""

from __future__ import annotations

from typing import Any

from quant_fund.research.receipt_v2 import seal_receipt, verify_receipt_payload


def _vol_bench_body() -> dict[str, Any]:
    return {
        "schema": "vol_bench.v1",
        "kind": "vol_bench",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "n_rows": 2,
        "n_error_rows": 1,
        "results": [
            {"shard": "garch", "model": "har", "horizon": 1, "status": "ok"},
            {"shard": "garch", "model": "har", "horizon": 5, "status": "error"},
        ],
    }


def _rankic_body() -> dict[str, Any]:
    return {
        "schema": "cross_sectional_rankic.v1",
        "kind": "cross_sectional_rankic_eval",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "n_rows": 1,
        "n_error_rows": 0,
        "results": [
            {
                "shard": "panel_a",
                "challenger": "momentum",
                "horizon": 5,
                "n_dates": 40,
                "status": "ok",
            }
        ],
    }


def _capacity_body() -> dict[str, Any]:
    return {
        "schema": "capacity_overlay.v1",
        "kind": "capacity_overlay_eval",
        "data_label": "SYNTHETIC",
        "live_pnl_claim": False,
        "dev_only": True,
        "n_rows": 1,
        "results": [
            {
                "book": "sleeve_a",
                "aum": 1e9,
                "participation_cap": 0.1,
                "status": "ok",
            }
        ],
    }


def test_v1_vol_bench_receipt_verifies() -> None:
    result = verify_receipt_payload(seal_receipt(_vol_bench_body()))
    assert result["valid"] is True, result["errors"]


def test_v1_vol_bench_error_row_count_cannot_be_zeroed() -> None:
    """Zeroing ``n_error_rows`` (which the verdict recomputes from) must trip
    the row re-derivation even under an honest re-seal."""
    body = {**_vol_bench_body(), "n_error_rows": 0}
    result = verify_receipt_payload(seal_receipt(body))
    assert result["valid"] is False
    assert "n_error_rows_mismatch" in result["errors"]


def test_v1_vol_bench_rejects_wrong_kind() -> None:
    body = {**_vol_bench_body(), "kind": "vol_bench_eval"}
    result = verify_receipt_payload(seal_receipt(body))
    assert result["valid"] is False
    assert "kind_not_vol_bench" in result["errors"]


def test_v1_rankic_receipt_verifies() -> None:
    result = verify_receipt_payload(seal_receipt(_rankic_body()))
    assert result["valid"] is True, result["errors"]


def test_v1_rankic_n_rows_must_match_results() -> None:
    body = {**_rankic_body(), "n_rows": 99}
    result = verify_receipt_payload(seal_receipt(body))
    assert result["valid"] is False
    assert "n_rows_mismatch" in result["errors"]


def test_v1_capacity_receipt_verifies() -> None:
    result = verify_receipt_payload(seal_receipt(_capacity_body()))
    assert result["valid"] is True, result["errors"]


def test_v1_capacity_requires_dev_only_flag() -> None:
    body = {**_capacity_body(), "dev_only": False}
    result = verify_receipt_payload(seal_receipt(body))
    assert result["valid"] is False
    assert "dev_only_not_true" in result["errors"]


def test_v1_capacity_n_rows_must_match_results() -> None:
    body = {**_capacity_body(), "n_rows": 7}
    result = verify_receipt_payload(seal_receipt(body))
    assert result["valid"] is False
    assert "n_rows_mismatch" in result["errors"]


def test_v1_lane_receipt_still_rejects_forbidden_metrics() -> None:
    body = {**_rankic_body(), "headline_sharpe": 2.0}
    result = verify_receipt_payload(seal_receipt(body))
    assert result["valid"] is False
    assert "forbidden_metric_keys" in result["errors"]
