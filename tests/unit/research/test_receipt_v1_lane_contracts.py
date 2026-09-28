"""v1-sealed lane receipts get the same contract scrutiny as ``fleet_eval.v1``."""

from __future__ import annotations

from typing import Any

from quant_fund.research.ranker_probability import _digest as _strict_digest
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


def _ranker_report(*, gate_pass: bool = True) -> dict[str, Any]:
    """A measured ranker_probability receipt whose embedded pieces agree."""
    return {
        "kind": "ranker_probability_experiment",
        "schema_version": 1,
        "spec": {
            "label": "future_idio_return_1",
            "label_end": "label_end_time_1",
            "features": ["cs_pct_mom_20"],
            "train_dates": 252,
            "cal_dates": 63,
            "test_dates": 63,
            "embargo_dates": 1,
            "ridge_alpha": 1.0,
            "n_boot": 2000,
            "mean_block": 10.0,
            "min_test_dates": 250,
            "seed": 17,
        },
        "feature_set_version": "v7",
        "input_rows": 9000,
        "common_rows": 8000,
        "folds": [
            {"status": "scored", "prediction_sha256": "a" * 64},
            {"status": "scored", "prediction_sha256": "b" * 64},
        ],
        "n_scored_dates": 300,
        "n_untrainable_folds": 0,
        "metric_unit": "equal_weight_per_decision_date",
        "research_only": True,
        "synthetic_is_correctness_only": True,
        "production_promotion": False,
        "forward_evidence_accepted": False,
        "status": "measured",
        "losses": {
            "ranker_platt": {"brier": 0.18, "log_loss": 0.51},
            "momentum_platt": {"brier": 0.20, "log_loss": 0.55},
            "train_base_rate": {"brier": 0.25, "log_loss": 0.69},
        },
        "paired_brier": {
            "momentum_platt": {"ci_low": -0.05, "ci_high": -0.01},
            "train_base_rate": {"ci_low": -0.09, "ci_high": -0.03},
        },
        "gate_pass": gate_pass,
        "gate_definition": "test gate",
        "input_sha256": {"features": "c" * 64, "labels": "d" * 64},
        "bronze_input_sha256": None,
        "gold_build_config_sha256": None,
        "code_sha256": {"ranker_probability.py": "e" * 64},
        "data_scope": "exploratory_previously_inspected_or_unverified",
        "holdout_previously_inspected_or_unverified": True,
    }


def _seal_strict(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": _strict_digest(body)}


def test_v1_ranker_probability_measured_receipt_verifies() -> None:
    result = verify_receipt_payload(_seal_strict(_ranker_report()))
    assert result["valid"] is True, result["errors"]


def test_v1_ranker_probability_gate_forgery_caught() -> None:
    """A report that fails its own gate but claims ``gate_pass: true`` is
    rejected even though the digest stays self-consistent."""
    body = _ranker_report(gate_pass=True)
    body["paired_brier"]["momentum_platt"] = {"ci_low": -0.05, "ci_high": 0.01}
    result = verify_receipt_payload(_seal_strict(body))
    assert result["valid"] is False
    assert "gate_pass_mismatch" in result["errors"]


def test_v1_ranker_probability_gate_claimed_false_when_met() -> None:
    """The reverse tamper — gate met but reported false — is also caught."""
    result = verify_receipt_payload(_seal_strict(_ranker_report(gate_pass=False)))
    assert result["valid"] is False
    assert "gate_pass_mismatch" in result["errors"]


def test_v1_ranker_probability_untrainable_count_rederived() -> None:
    body = _ranker_report()
    body["folds"].append({"status": "untrainable", "reason": "insufficient"})
    result = verify_receipt_payload(_seal_strict(body))
    assert result["valid"] is False
    # both the count and the gate (which requires zero untrainable folds)
    assert "n_untrainable_folds_mismatch" in result["errors"]
    assert "gate_pass_mismatch" in result["errors"]


def test_v1_ranker_probability_unmeasured_requires_gate_false() -> None:
    body = _ranker_report()
    body.update({"status": "unmeasured", "gate_pass": False, "reason": "x"})
    body.pop("losses")
    body.pop("paired_brier")
    result = verify_receipt_payload(_seal_strict(body))
    assert result["valid"] is True, result["errors"]
    bad = {**body, "gate_pass": True}
    result = verify_receipt_payload(_seal_strict(bad))
    assert result["valid"] is False
    assert "unmeasured_gate_must_be_false" in result["errors"]
