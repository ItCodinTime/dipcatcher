"""Coverage pack for ``research.verify`` fail-closed edges not exercised elsewhere.

Covers the receipt envelope (unreadable JSON, malformed header fields,
hypothesis/provenance/runtime shape), unknown-family and non-dict payloads,
the northset / candle_order_book / ranking soft-verify fan-in, and the
immutable-artifact hash/path checks.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from quant_fund.research.catalog import (
    BENCHMARK_CATALOG_VERSION,
    RESEARCH_RECEIPT_SCHEMA_VERSION,
)
from quant_fund.research.verify import (
    REQUIRED_BENCHMARK_FAMILIES,
    _receipt_digest,
    verify_research_artifact,
)
from quant_fund.utils.hashing import hash_file

_INF = float("inf")
_RUN_ID = "a" * 64


def _base_payload(tmp_path: Path) -> dict[str, Any]:
    immutable = tmp_path / "immutable"
    immutable.mkdir(exist_ok=True)
    (immutable / f"{_RUN_ID}.md").write_text("# research")
    return {
        "schema_version": RESEARCH_RECEIPT_SCHEMA_VERSION,
        "firm": "Artificial Hedge",
        "product": "Dipcatcher",
        "version": "1.0.0",
        "generated_at": "2026-09-16T00:00:00+00:00",
        "data_source": "SYNTHETIC",
        "synthetic": True,
        "disclaimer": "research only",
        "ranking_target": "future_return_1",
        "claim": "research_only",
        "rankers": [],
        "hypotheses": [],
        "provenance": {
            "run_id": _RUN_ID,
            "git_revision": "HEAD",
            "git_worktree_sha256": "e" * 64,
            "config_sha256": "b" * 64,
            "dataset_sha256": "c" * 64,
            "dataset_content_sha256": "d" * 64,
            "northset_inputs_sha256": "e" * 64,
            "row_count": 10,
            "column_count": 3,
            "point_in_time": True,
            "execution_claim": "research_only",
            "benchmark_catalog_version": BENCHMARK_CATALOG_VERSION,
            "runtime": {
                "python": "3.12.0",
                "implementation": "CPython",
                "platform": "test",
                "machine": "test",
                "byteorder": "little",
                "packages": {
                    "numpy": "2.0.0",
                    "polars": "1.0.0",
                    "scipy": "1.0.0",
                    "scikit-learn": "1.0.0",
                },
            },
        },
        "scorecard": {
            name: {
                "executed": True,
                "nonempty": True,
                "finite_observation": True,
                "forbidden_metrics_absent": True,
                "claim": "research_metric_only",
            }
            for name in REQUIRED_BENCHMARK_FAMILIES
        },
        "families": {name: {"executed": True} for name in REQUIRED_BENCHMARK_FAMILIES},
        "artifacts": {
            "immutable_json": str(immutable / f"{_RUN_ID}.json"),
            "immutable_markdown": str(immutable / f"{_RUN_ID}.md"),
            "immutable_markdown_sha256": hash_file(immutable / f"{_RUN_ID}.md"),
        },
    }


def _write_receipt(path: Path, payload: dict[str, Any]) -> Path:
    """Persist the notebook and its immutable JSON twin from one payload."""
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts["immutable_json_sha256"] = _receipt_digest(payload)
    text = json.dumps(payload)
    path.write_text(text)
    if isinstance(artifacts, dict):
        immutable = artifacts.get("immutable_json")
        if isinstance(immutable, str):
            twin = Path(immutable)
            if not twin.is_absolute():
                twin = path.parent / twin
            twin.parent.mkdir(parents=True, exist_ok=True)
            twin.write_text(text)
    return path


def _receipt(tmp_path: Path) -> Path:
    return _write_receipt(tmp_path / "latest.json", _base_payload(tmp_path))


def _verify_mutated(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    mutate(payload)
    _write_receipt(path, payload)
    return verify_research_artifact(path)


def _hypothesis(hyp_id: str, family: str) -> dict[str, Any]:
    return {
        "id": hyp_id,
        "statement": "s",
        "test": "t",
        "statistic": 1.0,
        "p_value": 0.5,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "d",
        "family": family,
    }


def test_rejects_unreadable_and_missing_notebook(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    path.write_text("{broken json")
    result = verify_research_artifact(path)
    assert result["valid"] is False
    assert result["errors"][0].startswith("unreadable:")

    missing = verify_research_artifact(tmp_path / "ghost.json")
    assert missing["valid"] is False
    assert missing["errors"][0].startswith("unreadable:")


def test_rejects_blank_and_non_string_top_level_fields(tmp_path: Path) -> None:
    result = _verify_mutated(
        tmp_path,
        lambda p: p.update({"version": " ", "disclaimer": 7, "ranking_target": None}),
    )
    errors = result["errors"]
    assert result["valid"] is False
    assert "invalid_notebook_version" in errors
    assert "invalid_notebook_disclaimer" in errors
    assert "invalid_notebook_ranking_target" in errors


def test_non_string_data_source_skips_synthetic_crosscheck(tmp_path: Path) -> None:
    # data_source: non-str -> field error, and the synthetic cross-check elif is
    # skipped (a non-string source cannot be compared to "SYNTHETIC").
    result = _verify_mutated(tmp_path, lambda p: p.update({"data_source": 123}))
    errors = result["errors"]
    assert "invalid_notebook_data_source" in errors
    assert "notebook_source_synthetic_mismatch" not in errors


def test_rejects_non_bool_synthetic_flag(tmp_path: Path) -> None:
    result = _verify_mutated(tmp_path, lambda p: p.update({"synthetic": "yes"}))
    assert "invalid_notebook_synthetic" in result["errors"]


@pytest.mark.parametrize("bad", [123, "not-a-timestamp"])
def test_rejects_non_iso_generated_at(tmp_path: Path, bad: object) -> None:
    result = _verify_mutated(tmp_path, lambda p: p.update({"generated_at": bad}))
    assert "invalid_notebook_generated_at" in result["errors"]


def test_rejects_non_list_rankers_and_hypotheses(tmp_path: Path) -> None:
    result = _verify_mutated(
        tmp_path,
        lambda p: p.update({"rankers": "oracle_raw", "hypotheses": "H1"}),
    )
    errors = result["errors"]
    assert "invalid_notebook_rankers" in errors
    assert "invalid_notebook_hypotheses" in errors


def test_rejects_non_dict_and_incomplete_hypotheses(tmp_path: Path) -> None:
    result = _verify_mutated(
        tmp_path,
        lambda p: p.update({"hypotheses": ["not-a-hypothesis", {"id": "H2", "statement": "x"}]}),
    )
    errors = result["errors"]
    assert "invalid_hypothesis:0" in errors
    assert any(e.startswith("hypothesis_fields_missing:1:") for e in errors)


def test_rejects_blank_hypothesis_fields(tmp_path: Path) -> None:
    hyp = {
        "id": "  ",
        "statement": "",
        "test": 9,
        "statistic": float("nan"),
        "p_value": 0.4,
        "reject_raw": False,
        "reject_fdr": False,
        "decision": "x",
        "family": "weird",
    }
    result = _verify_mutated(tmp_path, lambda p: p.update({"hypotheses": [hyp]}))
    errors = result["errors"]
    assert "invalid_hypothesis_id:0" in errors
    assert "invalid_hypothesis_family:0" in errors
    assert "invalid_hypothesis_statement:0" in errors
    assert "invalid_hypothesis_test:0" in errors


def test_rejects_non_numeric_hypothesis_p_value(tmp_path: Path) -> None:
    hyp = _hypothesis("H1", "calibration")
    hyp["p_value"] = "0.05"
    result = _verify_mutated(tmp_path, lambda p: p.update({"hypotheses": [hyp]}))
    assert "invalid_hypothesis_p_value:0" in result["errors"]


def test_rejects_missing_provenance_object(tmp_path: Path) -> None:
    # NB: provenance=None (key present) crashes verify.py's immutable compare
    # at ``immutable.get("provenance", {})`` returning None — flagged in the PR;
    # the absent-key path exercises the same fail-closed envelope checks.
    result = _verify_mutated(tmp_path, lambda p: p.pop("provenance"))
    errors = result["errors"]
    assert "provenance_missing" in errors
    assert any(e.startswith("provenance_fields_missing:") for e in errors)
    assert "invalid_run_id" in errors


def test_rejects_fail_closed_provenance_flags(tmp_path: Path) -> None:
    def mutate(p: dict[str, Any]) -> None:
        p["provenance"]["point_in_time"] = False
        p["provenance"]["execution_claim"] = "live_trading"

    result = _verify_mutated(tmp_path, mutate)
    assert "not_point_in_time" in result["errors"]
    assert "invalid_execution_claim" in result["errors"]


def test_rejects_bad_runtime_field_types(tmp_path: Path) -> None:
    result = _verify_mutated(
        tmp_path,
        lambda p: p["provenance"]["runtime"].update({"python": "", "machine": 7}),
    )
    errors = result["errors"]
    assert "runtime_field_invalid:python" in errors
    assert "runtime_field_invalid:machine" in errors


@pytest.mark.parametrize(
    ("packages", "expected"),
    [
        ({}, "runtime_packages_missing"),
        ({"numpy": 2.0}, "runtime_packages_invalid"),
        (
            {
                "numpy": "2.0.0",
                "polars": "1.0.0",
                "scipy": "UNAVAILABLE",
                "scikit-learn": "1.0.0",
            },
            "runtime_packages_unavailable:scipy",
        ),
    ],
)
def test_rejects_invalid_runtime_packages(
    tmp_path: Path, packages: dict[str, Any], expected: str
) -> None:
    result = _verify_mutated(
        tmp_path, lambda p: p["provenance"]["runtime"].update({"packages": packages})
    )
    assert expected in result["errors"]


def test_rejects_unknown_family_and_non_dict_blob(tmp_path: Path) -> None:
    def mutate(p: dict[str, Any]) -> None:
        p["families"]["mystery_family"] = {"executed": True}
        p["families"]["tail"] = "not-a-dict"

    result = _verify_mutated(tmp_path, mutate)
    errors = result["errors"]
    assert "families_unknown:mystery_family" in errors
    assert "families_invalid_payload" in errors


def test_northset_soft_verify_failures_fail_closed(tmp_path: Path) -> None:
    """A hostile northset blob must fail closed on every soft-verify lane."""

    def mutate(p: dict[str, Any]) -> None:
        p["families"]["northset"] = {
            "research_only": True,  # claim key absent -> coupling error
            "ohlc_identity_rate": 0.9,
            "book_uncrossed_rate": 0.8,
            "imbalance_top_p_ic": 0.3,
            "session_reconstructs_daily_rate": 0.7,
            "microprice_p_ic": 0.2,
            "session_book_vpin_p_ic": 0.2,
            "mean_session_book_snaps": -1.0,
            "depth_shape_finite_floor": 1.5,
            "session_l2_identity_gate": "maybe",
            "mean_microprice_weight_balance": 1.5,
            "structure_finite_rate": 1.5,
            "join_coverage": 2.0,
            "mean_microprice_minus_mid": _INF,
            "mean_book_age_seconds": -1.0,
            "mean_tob_size_share": 0.0,
            "mean_tob_notional_share": 2.0,
            "mean_close_mid_abs_rel": -0.1,
            "mean_queue_priority_proxy": 1.5,
            "mean_notional_imbalance": 2.0,
            "mean_bid_size_concentration_top": 2.0,
            "mean_depth_imbalance": -2.0,
            "mean_depth_imbalance_abs": 1.5,
            "mean_spread_bps": -1.0,
            "metrics_required_finite_ok": True,
            "shape_columns_ensured": True,
            "mean_imbalance_top": 5.0,
            "mean_bid_depth": -1.0,
            "mean_top_bid_size": -0.5,
            "mean_fwd_ret_after_high_reclaim": _INF,
            "mid": "not-numeric",
            "kyle_ofi": {"n_fused": 5.0},
        }

    result = _verify_mutated(tmp_path, mutate)
    errors = result["errors"]
    assert result["valid"] is False
    expected = [
        "northset_claim_missing_while_research_only_true",
        "hypothesis_h21_missing_despite_finite_book_uncrossed_rate",
        "hypothesis_h22_missing_despite_finite_imbalance_p_ic",
        "hypothesis_h23_missing_despite_finite_session_reconstructs_daily_rate",
        "hypothesis_h25_missing_despite_finite_microprice_p_ic",
        "hypothesis_h43_missing_despite_finite_session_book_vpin_p_ic",
        "mean_session_book_snaps_non_positive_or_non_finite",
        "depth_shape_finite_floor_out_of_unit_interval",
        "session_l2_identity_gate_invalid",
        "mean_microprice_weight_balance_out_of_unit_interval",
        "structure_finite_rate_out_of_unit_interval",
        "join_coverage_outside_open_unit_interval_fail_closed",
        "mean_microprice_minus_mid_non_finite_fail_closed",
        "mean_book_age_seconds_negative",
        "mean_tob_size_share_out_of_open_unit_interval",
        "mean_tob_notional_share_out_of_open_unit_interval",
        "mean_close_mid_abs_rel_negative",
        "mean_queue_priority_proxy_out_of_unit_interval",
        "mean_notional_imbalance_out_of_unit_interval",
        "mean_bid_size_concentration_top_out_of_open_unit_interval",
        "mean_depth_imbalance_out_of_unit_interval",
        "mean_depth_imbalance_abs_out_of_unit_interval",
        "mean_spread_bps_negative_or_non_finite",
        "depth_shape_finite_rate_missing_while_metrics_required_finite_ok",
        "depth_shape_finite_rate_missing_while_shape_columns_ensured",
        "mean_imbalance_top_out_of_unit_interval",
        "mean_bid_depth_negative",
        "mean_top_bid_size_negative",
        "mean_fwd_ret_after_high_reclaim_non_finite",
        "mid_non_numeric_metrics_required",
        "kyle_ofi_min_names_missing",
    ]
    for token in expected:
        assert token in errors, token
    assert any(e.startswith("northset_receipt_unclassified_keys:") for e in errors)


def test_northset_hypothesis_row_satisfies_minting(tmp_path: Path) -> None:
    """Minted H21 row silences the soft-verify gate on a clean receipt."""

    def mutate(p: dict[str, Any]) -> None:
        p["families"]["northset"]["book_uncrossed_rate"] = 0.8
        p["hypotheses"] = [_hypothesis("H21_northset_book", "bound")]

    result = _verify_mutated(tmp_path, mutate)
    assert result["errors"] == []
    assert result["valid"] is True


def test_candle_order_book_soft_verify_failures_fail_closed(tmp_path: Path) -> None:
    """A hostile candle_order_book blob must fail closed on every lane."""

    def mutate(p: dict[str, Any]) -> None:
        p["families"]["candle_order_book"] = {
            "family": "candle_order_book",
            "research_only": False,
            "claim": "live_execution",
            "label": "",
            "book_source": 7,
            "dgp": "synthetic_lob",
            "book_dgp": "real_lob",
            "data_source": "parquet",
            "min_names": 0,
            "depth": -1,
            "n_bars": 5,
            "n_fused": 10,
            "n_scored": 12,
            "join_coverage": 2.0,
            "ic_method": "median_ic",
            "ic_ofi": 0.5,
            "ic_ofi_t": _INF,
            "ic_ofi_pearson": 1.5,
            "mean_ofi": _INF,
            "mean_queue_imbalance": 2.0,
            "ic_microprice_weight_balance": 2.0,
            "ic_microprice_weight_balance_p": 1.5,
            "ic_microprice_weight_balance_n_dates": -1,
            "mean_microprice_weight_balance": 1.5,
            "ic_notional_imbalance": 0.1,
            "mean_notional_imbalance": 5.0,
            "ic_spread_over_mid": 0.2,
            "mean_spread_over_mid": -0.5,
            "ic_imbalance_depth": 0.2,
            "mean_depth_imbalance": 2.0,
            "ic_depth_imbalance_abs": 0.2,
            "mean_depth_imbalance_abs": 1.5,
            "ic_imbalance_top": 0.2,
            "mean_imbalance_top": 5.0,
            "ic_spread_bps": 0.9,
            "finite_rate_microprice_minus_mid": 1.5,
            "finite_rate_bid_size_concentration_top": 0.5,
            "finite_rate_ask_size_concentration_top": 0.5,
            "structure_finite_rate": 0.9,
            "mean_quoted_spread": -1.0,
            "mean_spread_bps": -0.5,
            "mean_candle_direction": 2.0,
            "mean_wick_skew": _INF,
            "mean_signed_vol_x_imbalance": _INF,
            "mean_candle_range_frac": 1.5,
            "mean_spread_x_range": -0.5,
            "mean_candle_dir_x_imbalance": 2.0,
            "mean_bid_log_size_slope": _INF,
            "mean_bid_mean_log_tick_spacing": _INF,
            "mean_microprice_minus_mid": _INF,
            "mean_tob_size_share": 1.5,
            "mean_close_mid_abs_rel": -0.1,
            "mean_queue_priority_proxy": 1.5,
            "mean_bid_size_concentration_top": 2.0,
            "mean_book_age_seconds": -1.0,
            "depth_shape_finite_rate": 1.5,
        }

    result = _verify_mutated(tmp_path, mutate)
    errors = result["errors"]
    assert result["valid"] is False
    expected = [
        "candle_research_only_missing_or_false",
        "candle_claim_not_research_diagnostic_only",
        "candle_label_empty_or_not_str",
        "book_source_non_str",
        "candle_dgp_book_dgp_mismatch",
        "candle_min_names_lt_one_or_not_int",
        "candle_depth_lt_one_or_not_int",
        "candle_n_fused_gt_n_bars",
        "candle_n_scored_gt_n_fused",
        "n_fused_gt_n_bars",
        "n_scored_gt_n_fused",
        "join_coverage_out_of_open_unit_interval",
        "join_coverage_outside_open_unit_interval_fail_closed",
        "candle_order_book_ic_method_invalid",
        "ic_microprice_weight_balance_out_of_unit_interval",
        "ic_microprice_weight_balance_p_out_of_unit_interval",
        "ic_microprice_weight_balance_n_dates_negative",
        "ic_ofi_t_non_finite_fail_closed",
        "ic_ofi_pearson_out_of_unit_interval",
        "mean_ofi_non_finite",
        "mean_queue_imbalance_out_of_signed_unit",
        "mean_microprice_weight_balance_out_of_unit_interval",
        "mean_notional_imbalance_out_of_unit_interval",
        "mean_spread_over_mid_negative_or_non_finite",
        "mean_depth_imbalance_out_of_unit_interval",
        "mean_depth_imbalance_abs_out_of_unit_interval",
        "mean_imbalance_top_out_of_unit_interval",
        "ic_spread_bps_diverges_from_ic_spread_over_mid",
        "finite_rate_microprice_minus_mid_out_of_unit_interval",
        "candle_structure_finite_rate_not_nanmean_of_finite_rate_companions",
        "mean_quoted_spread_negative_or_non_finite",
        "mean_spread_bps_negative_or_non_finite",
        "mean_candle_direction_out_of_signed_unit",
        "mean_wick_skew_non_finite",
        "mean_signed_vol_x_imbalance_non_finite",
        "mean_candle_range_frac_out_of_unit_interval",
        "mean_spread_x_range_negative",
        "mean_candle_dir_x_imbalance_out_of_signed_unit",
        "mean_bid_log_size_slope_non_finite",
        "mean_bid_mean_log_tick_spacing_non_finite",
        "mean_microprice_minus_mid_non_finite_fail_closed",
        "mean_tob_size_share_out_of_open_unit_interval",
        "mean_close_mid_abs_rel_negative",
        "mean_queue_priority_proxy_out_of_unit_interval",
        "mean_bid_size_concentration_top_out_of_open_unit_interval",
        "mean_book_age_seconds_negative",
        "depth_shape_finite_rate_out_of_unit_interval",
    ]
    for token in expected:
        assert token in errors, token
    assert any(e.endswith("_missing_from_candle_feature_cols_receipt") for e in errors)


def test_robinhood_plus_claim_fail_closed(tmp_path: Path) -> None:
    result = _verify_mutated(
        tmp_path,
        lambda p: p["families"].update(
            {
                "robinhood_plus": {
                    "family": "robinhood_plus",
                    "research_only": False,
                    "execution_claim": "live",
                    "claim": "live_performance",
                    "backend": "jax",
                }
            }
        ),
    )
    errors = result["errors"]
    assert "robinhood_plus_research_only_invalid" in errors
    assert "robinhood_plus_execution_claim_invalid" in errors
    assert "robinhood_plus_claim_invalid" in errors
    assert "robinhood_plus_backend_invalid" in errors


def test_ranking_data_snooping_fail_closed(tmp_path: Path) -> None:
    def mutate(p: dict[str, Any]) -> None:
        p["families"]["ranking"]["data_snooping"] = {
            "research_only": False,
            "claim": "live",
            "spa_p_consistent": 0.3,
            "n_trials": 0,
            "stepm_rejected": ["a"],
            "stepm_n_rejected": 3,
            "mcs_included": ["x"],
            "mcs_n_included": 0,
        }

    result = _verify_mutated(tmp_path, mutate)
    errors = result["errors"]
    assert "data_snooping_research_only_missing_or_false" in errors
    assert "data_snooping_claim_not_research_diagnostic_only" in errors
    assert "data_snooping_n_trials_invalid" in errors
    assert "data_snooping_stepm_n_rejected_mismatch" in errors
    assert "data_snooping_mcs_n_included_mismatch" in errors
    assert "hypothesis_h99_missing_despite_finite_spa_p" in errors


def test_rejects_non_dict_scorecard_entry(tmp_path: Path) -> None:
    result = _verify_mutated(tmp_path, lambda p: p["scorecard"].update({"ranking": "corrupt"}))
    assert "scorecard_invalid:ranking" in result["errors"]


def test_rejects_missing_artifacts_block(tmp_path: Path) -> None:
    result = _verify_mutated(tmp_path, lambda p: p.pop("artifacts"))
    assert "artifacts_missing" in result["errors"]


def test_rejects_absent_immutable_artifact_files(tmp_path: Path) -> None:
    # Empty artifacts dict: no resolvable immutable paths -> outside root +
    # missing file for both keys, and the immutable-compare block is skipped.
    result = _verify_mutated(tmp_path, lambda p: p.update({"artifacts": {}}))
    errors = result["errors"]
    assert "artifact_outside_receipt_root:immutable_json" in errors
    assert "artifact_outside_receipt_root:immutable_markdown" in errors
    assert "artifact_missing:immutable_json" in errors
    assert "artifact_missing:immutable_markdown" in errors


def test_rejects_latest_pointer_mismatch_and_bad_hash(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["artifacts"] = {
        "json": "latest.json",  # canonical name -> pointer loop continues
        "markdown": "archive/old.md",  # wrong name -> mismatch token
        "immutable_json_sha256": "xyz",  # malformed digest shape
    }
    path.write_text(json.dumps(payload))
    result = verify_research_artifact(path)
    errors = result["errors"]
    assert "latest_markdown_artifact_pointer_mismatch" in errors
    assert "invalid_artifact_hash:immutable_json_sha256" in errors
    assert "latest_json_artifact_pointer_mismatch" not in errors


def test_rejects_unverifiable_markdown_and_missing_json_hash(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    payload["artifacts"]["immutable_markdown"] = "immutable/absent.md"
    payload["artifacts"]["immutable_markdown_sha256"] = "f" * 64
    payload["artifacts"].pop("immutable_json_sha256")
    text = json.dumps(payload)
    path.write_text(text)
    Path(payload["artifacts"]["immutable_json"]).write_text(text)
    result = verify_research_artifact(path)
    errors = result["errors"]
    assert "artifact_missing:immutable_markdown" in errors
    assert "immutable_json_hash_missing" in errors
    assert "immutable_markdown_hash_unverifiable" in errors


def test_rejects_unreadable_immutable_json(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    payload = json.loads(path.read_text())
    Path(payload["artifacts"]["immutable_json"]).write_text("{corrupt")
    result = verify_research_artifact(path)
    assert "immutable_json_unreadable" in result["errors"]


def test_artifact_root_resolution_oserror_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(self: Path, other: Path) -> bool:
        raise OSError("simulated resolution failure")

    monkeypatch.setattr(Path, "is_relative_to", _boom)
    result = verify_research_artifact(_receipt(tmp_path))
    assert result["valid"] is False
    assert "artifact_outside_receipt_root:immutable_json" in result["errors"]
    assert "artifact_outside_receipt_root:immutable_markdown" in result["errors"]
