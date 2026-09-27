"""Fail-closed verification of persisted research receipts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

from quant_fund.northset.benches import (
    northset_receipt_key_classification_honesty_errors,
)
from quant_fund.research.catalog import (
    BENCHMARK_CATALOG_VERSION,
    OPTIONAL_BENCHMARK_FAMILIES,
    REQUIRED_BENCHMARK_FAMILIES,
    RESEARCH_RECEIPT_SCHEMA_VERSION,
    book_age_seconds_honesty_errors,
    candle_all_finite_rate_prefix_honesty_errors,
    candle_all_ic_n_dates_nonneg_honesty_errors,
    candle_all_ic_p_unit_honesty_errors,
    candle_all_ic_pearson_unit_honesty_errors,
    candle_all_ic_t_finite_honesty_errors,
    candle_depth_imbalance_ic_implies_mean_honesty_errors,
    candle_direction_mean_honesty_errors,
    candle_feature_cols_ic_completeness_honesty_errors,
    candle_feature_cols_ic_honesty_errors,
    candle_feature_cols_ic_implies_mean_honesty_errors,
    candle_feature_ofi_finite_honesty_errors,
    candle_frac_and_spread_x_honesty_errors,
    candle_join_coverage_and_chain_honesty_errors,
    candle_log_slopes_finite_honesty_errors,
    candle_log_tick_spacing_finite_honesty_errors,
    candle_microprice_weight_balance_ic_honesty_errors,
    candle_mwb_scored_implies_mean_unit_honesty_errors,
    candle_notional_imbalance_ic_implies_mean_honesty_errors,
    candle_ofi_and_queue_imbalance_means_honesty_errors,
    candle_ofi_qp_slope_ic_implies_mean_honesty_errors,
    candle_order_book_claim_honesty_errors,
    candle_order_book_dgp_data_source_honesty_errors,
    candle_order_book_family_provenance_honesty_errors,
    candle_order_book_ic_method_honesty_errors,
    candle_order_book_sizing_honesty_errors,
    candle_signed_vol_x_imbalance_mean_honesty_errors,
    candle_spread_alias_honesty_errors,
    candle_spread_bps_ic_matches_spread_over_mid_ic_honesty_errors,
    candle_spread_bps_nonneg_honesty_errors,
    candle_spread_over_mid_ic_implies_mean_honesty_errors,
    candle_structure_finite_rate_covers_companions_honesty_errors,
    candle_structure_ic_implies_mean_honesty_errors,
    candle_wick_skew_and_body_ret_means_honesty_errors,
    coverage_guarantee_scope_consistency_errors,
    depth_shape_finite_rate_honesty_errors,
    dist_crps_eprocess_keys_present,
    dist_crps_eprocess_missing_keys,
    family_blob_executed,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
    family_blob_nonempty,
    h1_hypothesis_consistency_errors,
    h2_hypothesis_consistency_errors,
    h3_hypothesis_consistency_errors,
    h4_hypothesis_consistency_errors,
    h4b_hypothesis_consistency_errors,
    h7_hypothesis_consistency_errors,
    h8_hypothesis_consistency_errors,
    h9_hypothesis_consistency_errors,
    h10_hypothesis_consistency_errors,
    h11_hypothesis_consistency_errors,
    h12_hypothesis_consistency_errors,
    h15_hypothesis_consistency_errors,
    h16_h18_panel_kupiec_consistency_errors,
    h19_hypothesis_consistency_errors,
    h20_hypothesis_consistency_errors,
    h21_hypothesis_consistency_errors,
    h22_hypothesis_consistency_errors,
    h43_hypothesis_consistency_errors,
    h99_hypothesis_consistency_errors,
    join_coverage_honesty_errors,
    kyle_ofi_nest_honesty_errors,
    mean_candle_dir_x_imbalance_honesty_errors,
    mean_close_mid_abs_rel_honesty_errors,
    mean_depth_imbalance_abs_honesty_errors,
    mean_depth_imbalance_honesty_errors,
    mean_imbalance_top_honesty_errors,
    mean_microprice_minus_mid_honesty_errors,
    mean_microprice_weight_balance_honesty_errors,
    mean_notional_imbalance_honesty_errors,
    mean_queue_priority_honesty_errors,
    mean_tob_notional_share_honesty_errors,
    mean_tob_size_share_honesty_errors,
    northset_depth_notional_spread_over_mid_honesty_errors,
    northset_fwd_ret_after_sweep_honesty_errors,
    northset_h23_h28_consistency_errors,
    northset_metrics_required_finite_ok_rates_honesty_errors,
    northset_metrics_required_keys_finite_when_present_honesty_errors,
    northset_price_slope_tick_top_levels_honesty_errors,
    northset_receipt_honesty_errors,
    northset_session_means_honesty_errors,
    northset_shape_and_session_l2_floors_honesty_errors,
    northset_shape_columns_ensured_rates_honesty_errors,
    northset_top_level_claim_honesty_errors,
    ranking_data_snooping_honesty_errors,
    robinhood_plus_claim_honesty_errors,
    size_concentration_top_honesty_errors,
    structure_finite_rate_honesty_errors,
    tail_es_battery_keys_present,
    tail_es_battery_missing_keys,
    tail_var_battery_keys_present,
    tail_var_battery_missing_keys,
)
from quant_fund.utils.hashing import SHA256_HEX_LENGTH, hash_bytes, hash_file


def _receipt_digest(payload: dict[str, Any]) -> str:
    """Hash a receipt canonically while excluding its self-referential digest."""
    normalized = json.loads(json.dumps(payload))
    artifacts = normalized.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts.pop("immutable_json_sha256", None)
    return hash_bytes(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode())


REQUIRED_PROVENANCE = {
    "run_id",
    "git_revision",
    "config_sha256",
    "dataset_sha256",
    "git_worktree_sha256",
    "dataset_content_sha256",
    "northset_inputs_sha256",
    "row_count",
    "column_count",
    "point_in_time",
    "execution_claim",
}

REQUIRED_NOTEBOOK_IDENTITY = {
    "firm": "Artificial Hedge",
    "product": "Dipcatcher",
    "claim": "research_only",
}
REQUIRED_HYPOTHESIS_FIELDS = {
    "id",
    "statement",
    "test",
    "statistic",
    "p_value",
    "reject_raw",
    "reject_fdr",
    "decision",
    "family",
}
REQUIRED_RUNTIME_PACKAGES = frozenset({"numpy", "polars", "scipy", "scikit-learn"})


def _finite_or_nan_number(value: object) -> bool:
    """Accept numeric research statistics, including explicit unavailable NaN."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    numeric = cast(float, value)
    return math.isnan(numeric) or math.isfinite(numeric)


def _p_value_valid(value: object) -> bool:
    """Accept a p-value in [0, 1], or NaN/None for an unavailable test.

    ``None`` is accepted because agent ``_jsonable`` maps non-finite floats to
    JSON null (strict JSON has no NaN token). Bound hyps (H10/H15/H19) mint
    ``p_value=nan`` which round-trips as null — still an unavailable test.
    """
    if value is None:
        return True
    if not _finite_or_nan_number(value):
        return False
    numeric = cast(float, value)
    return math.isnan(numeric) or 0.0 <= numeric <= 1.0


def _is_sha256(value: object) -> bool:
    """Accept only lowercase hexadecimal SHA-256 digests."""
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _timestamp_valid(value: object) -> bool:
    """Require an ISO-8601 timestamp carrying an explicit timezone."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


class UnreadableResearchVerification(TypedDict):
    """Early fail-closed receipt result: the notebook was not an object."""

    valid: bool
    path: str
    errors: list[str]


class ResearchVerification(TypedDict):
    """Receipt verification result. ``run_id`` is untrusted until the errors list is empty."""

    valid: bool
    path: str
    run_id: object
    scorecard_families: int
    errors: list[str]
    claim: str


def _tag(prefix: str, keys: Iterable[object]) -> list[str]:
    """Prefix each catalog missing-key token. Order follows ``keys``."""
    return [f"{prefix}:{key}" for key in keys]


def _notebook_identity_errors(notebook: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, expected in REQUIRED_NOTEBOOK_IDENTITY.items():
        if notebook.get(key) != expected:
            errors.append(f"invalid_notebook_{key}")
    schema_version = notebook.get("schema_version")
    # Bools are ints in Python; reject them explicitly so True cannot pass as 1.
    if isinstance(schema_version, bool) or schema_version != RESEARCH_RECEIPT_SCHEMA_VERSION:
        errors.append("invalid_research_receipt_schema_version")
    for key in ("version", "data_source", "disclaimer", "ranking_target"):
        value = notebook.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"invalid_notebook_{key}")
    if not _timestamp_valid(notebook.get("generated_at")):
        errors.append("invalid_notebook_generated_at")
    synthetic = notebook.get("synthetic")
    if not isinstance(synthetic, bool):
        errors.append("invalid_notebook_synthetic")
    elif isinstance(notebook.get("data_source"), str):
        # Strip before comparing so a whitespace-padded " SYNTHETIC " still
        # counts as the synthetic source and a false synthetic flag is rejected.
        is_synthetic_source = notebook["data_source"].strip().upper() == "SYNTHETIC"
        if synthetic != is_synthetic_source:
            errors.append("notebook_source_synthetic_mismatch")
    return errors


def _ranker_errors(notebook: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rankers = notebook.get("rankers")
    if not isinstance(rankers, list):
        errors.append("invalid_notebook_rankers")
    else:
        ranker_names: set[str] = set()
        for index, ranker in enumerate(rankers):
            if not isinstance(ranker, dict):
                errors.append(f"invalid_ranker:{index}")
                continue
            if not isinstance(ranker.get("name"), str) or not ranker["name"].strip():
                errors.append(f"invalid_ranker_name:{index}")
            else:
                ranker_name = ranker["name"].strip()
                if ranker_name in ranker_names:
                    errors.append(f"duplicate_ranker_name:{ranker_name}")
                ranker_names.add(ranker_name)
            for field in ("n_dates", "n_folds"):
                if field not in ranker:
                    continue
                value = ranker[field]
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or int(value) != float(value)
                    or int(value) < 0
                ):
                    errors.append(f"invalid_ranker_{field}:{index}")
    return errors


def _hypothesis_record_errors(notebook: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    hypotheses = notebook.get("hypotheses")
    if not isinstance(hypotheses, list):
        errors.append("invalid_notebook_hypotheses")
    else:
        hypothesis_ids: set[str] = set()
        for index, hypothesis in enumerate(hypotheses):
            if not isinstance(hypothesis, dict):
                errors.append(f"invalid_hypothesis:{index}")
                continue
            missing_fields = sorted(REQUIRED_HYPOTHESIS_FIELDS - hypothesis.keys())
            if missing_fields:
                errors.append(f"hypothesis_fields_missing:{index}:{','.join(missing_fields)}")
            hypothesis_id = hypothesis.get("id")
            if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
                errors.append(f"invalid_hypothesis_id:{index}")
            elif hypothesis_id in hypothesis_ids:
                errors.append(f"duplicate_hypothesis_id:{hypothesis_id}")
            else:
                hypothesis_ids.add(hypothesis_id)
            if hypothesis.get("family") not in {"calibration", "discovery", "bound"}:
                errors.append(f"invalid_hypothesis_family:{index}")
            for field in ("statement", "test", "decision"):
                value = hypothesis.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"invalid_hypothesis_{field}:{index}")
            if not _finite_or_nan_number(hypothesis.get("statistic")):
                errors.append(f"invalid_hypothesis_statistic:{index}")
            if not _p_value_valid(hypothesis.get("p_value")):
                errors.append(f"invalid_hypothesis_p_value:{index}")
            for field in ("reject_raw", "reject_fdr"):
                if type(hypothesis.get(field)) is not bool:
                    errors.append(f"invalid_hypothesis_{field}:{index}")
            p_value = hypothesis.get("p_value")
            unavailable_p = p_value is None or (
                _finite_or_nan_number(p_value) and math.isnan(float(cast(float, p_value)))
            )
            if unavailable_p and (hypothesis.get("reject_raw") or hypothesis.get("reject_fdr")):
                errors.append(f"hypothesis_rejection_without_p_value:{index}")
    return errors


def _provenance_state(notebook: dict[str, Any]) -> tuple[dict[str, Any], object, list[str]]:
    errors: list[str] = []
    provenance = notebook.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance_missing")
        provenance = {}
    missing = sorted(REQUIRED_PROVENANCE - provenance.keys())
    if missing:
        errors.append(f"provenance_fields_missing:{','.join(missing)}")
    run_id = provenance.get("run_id")
    if not _is_sha256(run_id):
        errors.append("invalid_run_id")
    git_revision = provenance.get("git_revision")
    if not isinstance(git_revision, str) or not git_revision.strip():
        errors.append("invalid_git_revision")
    for key in (
        "config_sha256",
        "dataset_sha256",
        "git_worktree_sha256",
        "dataset_content_sha256",
        "northset_inputs_sha256",
    ):
        value = provenance.get(key)
        if not _is_sha256(value):
            errors.append(f"invalid_{key}")
    for key in ("row_count", "column_count"):
        value = provenance.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"invalid_{key}")
    if provenance.get("point_in_time") is not True:
        errors.append("not_point_in_time")
    if provenance.get("execution_claim") != "research_only":
        errors.append("invalid_execution_claim")
    if provenance.get("benchmark_catalog_version") != BENCHMARK_CATALOG_VERSION:
        errors.append("invalid_benchmark_catalog_version")
    runtime = provenance.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("runtime_missing")
    else:
        for field in ("python", "implementation", "platform", "machine", "byteorder"):
            if not isinstance(runtime.get(field), str) or not runtime[field]:
                errors.append(f"runtime_field_invalid:{field}")
        packages = runtime.get("packages")
        if not isinstance(packages, dict) or not packages:
            errors.append("runtime_packages_missing")
        elif any(not isinstance(name, str) or not name for name in packages) or any(
            not isinstance(value, str) or not value for value in packages.values()
        ):
            errors.append("runtime_packages_invalid")
        else:
            missing_packages = sorted(REQUIRED_RUNTIME_PACKAGES - packages.keys())
            unavailable_packages = sorted(
                name for name in REQUIRED_RUNTIME_PACKAGES if packages.get(name) == "UNAVAILABLE"
            )
            if missing_packages:
                errors.append(f"runtime_packages_missing_required:{','.join(missing_packages)}")
            if unavailable_packages:
                errors.append(f"runtime_packages_unavailable:{','.join(unavailable_packages)}")
    return provenance, run_id, errors


def _family_shape_errors(families: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    allowed_families = REQUIRED_BENCHMARK_FAMILIES | OPTIONAL_BENCHMARK_FAMILIES
    missing_families = sorted(REQUIRED_BENCHMARK_FAMILIES - families.keys())
    unknown_families = sorted(set(families.keys()) - allowed_families)
    if missing_families:
        errors.append(f"families_missing:{','.join(missing_families)}")
    if unknown_families:
        errors.append(f"families_unknown:{','.join(unknown_families)}")
    if any(not isinstance(blob, dict) for blob in families.values()):
        errors.append("families_invalid_payload")
    for fam_name, blob in families.items():
        if not family_blob_forbidden_metrics_absent(blob):
            errors.append(f"families_forbidden_metrics:{fam_name}")
    return errors


def _family_fanin_errors(notebook: dict[str, Any], families: dict[str, Any]) -> list[str]:
    """Catalog honesty fan-in, in the historical receipt order.

    Each checker is pure. Building the error list here does not interpret a
    metric as a trading signal.
    """
    errors: list[str] = []
    # Soft VaR-battery honesty (Day Wave 18): nonempty tail with kupiec_p or
    # kupiec_lr must expose Christoffersen CC (+ preferred ind) keys so
    # DayWave16 cannot silently regress. Values may be NaN; empty {} skips.
    # Research-receipt fail-closed on missing key *presence* only — not a
    # live capital / promotion gate.
    errors.extend(
        _tag("tail_var_battery_incomplete", tail_var_battery_missing_keys(families.get("tail")))
    )
    # Soft ES-battery honesty (Day Wave 21): nonempty tail with es_95 /
    # realized_es / var_95 must expose Acerbi Z1/Z2 + FZ mean + es_hit_count.
    # Orthogonal to VaR-battery. Research-receipt fail-closed on key presence.
    errors.extend(
        _tag("tail_es_battery_incomplete", tail_es_battery_missing_keys(families.get("tail")))
    )
    # Soft distribution CRPS e-process honesty (Day Wave 20): nonempty
    # distribution with dm_crps_p / dm_crps_scaled_p must expose matching
    # e_dm_crps_* / e_dm_crps_scaled_* keys so DayWave18/19 cannot silently
    # regress. Values may be NaN; empty {} skips. Research-receipt fail-closed
    # on missing key *presence* only — not a live capital / promotion gate.
    errors.extend(
        _tag(
            "dist_crps_eprocess_incomplete",
            dist_crps_eprocess_missing_keys(families.get("distribution")),
        )
    )
    # Soft H4b notebook consistency (Day Wave 26): finite tail.christoffersen_cc_p
    # must mint H4b_var_christoffersen_cc (calibration). Non-finite/missing → skip.
    # Research-receipt fail-closed — not a live capital / promotion gate.
    errors.extend(
        h4b_hypothesis_consistency_errors(families.get("tail"), notebook.get("hypotheses"))
    )
    # Soft H4 Kupiec notebook consistency (Day Wave 29 research diagnostic):
    # finite tail.kupiec_p must mint H4_var_kupiec (calibration). Non-finite/
    # missing → skip. Not a live capital / promotion gate.
    errors.extend(
        h4_hypothesis_consistency_errors(families.get("tail"), notebook.get("hypotheses"))
    )
    # Soft H3 vol-DM notebook consistency (Day Wave 30 research diagnostic):
    # finite volatility.dm_p must mint H3_vol_dm (discovery). Non-finite/
    # missing → skip. Not a live capital / promotion gate.
    errors.extend(
        h3_hypothesis_consistency_errors(families.get("volatility"), notebook.get("hypotheses"))
    )
    # Soft H7 ACI coverage notebook consistency (Day Wave 32 research diagnostic):
    # finite conformal.aci.kupiec_p must mint H7_aci_coverage (calibration).
    # Non-finite/missing/no aci → skip. Not a live capital / promotion gate.
    errors.extend(
        h7_hypothesis_consistency_errors(families.get("conformal"), notebook.get("hypotheses"))
    )
    # Soft H8 Mondrian high-vol notebook consistency (Day Wave 33 research diagnostic):
    # finite conformal.mondrian_aci.high_x_kupiec_p must mint H8_mondrian_high_vol
    # (calibration). Non-finite/missing/no mondrian_aci → skip. Not a live promotion gate.
    errors.extend(
        h8_hypothesis_consistency_errors(families.get("conformal"), notebook.get("hypotheses"))
    )
    # Soft H11 CRC notebook consistency (Day Wave 34 research diagnostic):
    # finite crc.kupiec_p must mint H11_crc_var (calibration).
    errors.extend(
        h11_hypothesis_consistency_errors(families.get("crc"), notebook.get("hypotheses"))
    )
    # Soft H12 weighted CQR notebook consistency (Day Wave 35 research diagnostic):
    # finite weighted_conformal.kupiec_p must mint H12_weighted_cqr (calibration).
    # Non-finite/missing → skip. Not a live capital / promotion gate.
    errors.extend(
        h12_hypothesis_consistency_errors(
            families.get("weighted_conformal"), notebook.get("hypotheses")
        )
    )
    # Soft H9 e-process ACI notebook consistency (Day Wave 36 research diagnostic):
    # finite evalues.e_sup must mint H9_eprocess_aci (calibration).
    # Non-finite/missing → skip. Not a live capital / promotion gate.
    errors.extend(
        h9_hypothesis_consistency_errors(families.get("evalues"), notebook.get("hypotheses"))
    )
    # Soft H10 Jackknife+ coverage notebook consistency (Day Wave 37 research diagnostic):
    # finite jackknife_plus.coverage must mint H10_jackknife_coverage (bound).
    # Non-finite/missing → skip. Not a live capital / promotion gate.
    errors.extend(
        h10_hypothesis_consistency_errors(
            families.get("jackknife_plus"), notebook.get("hypotheses")
        )
    )
    # Soft H16–H18 panel Kupiec notebook consistency (Day Wave 39 research diagnostic):
    # finite localized_conformal / online_crc / portfolio_conformal kupiec_p (non-fixture)
    # must mint H16/H17/H18 (calibration). Fixture dgp / non-finite / missing → skip.
    # Not a live capital / promotion gate.
    errors.extend(h16_h18_panel_kupiec_consistency_errors(families, notebook.get("hypotheses")))
    # Soft H15 CV+ floor notebook consistency (Day Wave 38 research diagnostic):
    # finite cv_plus.coverage + coverage_floor must mint H15_cv_plus_floor (bound).
    # Non-finite/missing either → skip. Not a live capital / promotion gate.
    errors.extend(
        h15_hypothesis_consistency_errors(families.get("cv_plus"), notebook.get("hypotheses"))
    )
    # Soft H19 conformal-rank FDR bound notebook consistency (Day Wave 40 research diagnostic):
    # finite conformal_rank.fdr (non-fixture) must mint H19_conformal_rank (bound).
    # Fixture dgp / non-finite / missing → skip. Not a live capital / promotion gate.
    errors.extend(
        h19_hypothesis_consistency_errors(
            families.get("conformal_rank"), notebook.get("hypotheses")
        )
    )
    # Soft H20–H22 Northset notebook consistency (catalog v2 research diagnostic):
    # finite ohlc_identity_rate / book_uncrossed_rate / imbalance_top_p_ic must
    # mint H20 (bound) / H21 (bound) / H22 (discovery). Non-finite/missing → skip.
    # Not a live capital / promotion gate.
    errors.extend(
        h20_hypothesis_consistency_errors(families.get("northset"), notebook.get("hypotheses"))
    )
    errors.extend(
        h21_hypothesis_consistency_errors(families.get("northset"), notebook.get("hypotheses"))
    )
    errors.extend(
        h22_hypothesis_consistency_errors(families.get("northset"), notebook.get("hypotheses"))
    )
    errors.extend(
        northset_h23_h28_consistency_errors(families.get("northset"), notebook.get("hypotheses"))
    )
    errors.extend(
        h43_hypothesis_consistency_errors(families.get("northset"), notebook.get("hypotheses"))
    )
    # Soft Jackknife+/CV+ marginal coverage_guarantee_scope honesty (Day Wave 42):
    # nonempty jp/cv with coverage and/or coverage_floor keys must expose
    # coverage_guarantee_scope=marginal_exchangeable (Wave41). Empty/missing keys → skip.
    # Not a live capital / promotion gate; not H-table sprawl.
    errors.extend(coverage_guarantee_scope_consistency_errors(families))
    errors.extend(northset_session_means_honesty_errors(families.get("northset")))
    errors.extend(northset_receipt_key_classification_honesty_errors(families.get("northset")))
    # Catalog receipt soft-verify fan-in (vpin_mean / gap_finite_rate /
    # ohlc_identity_rate / spread means / sweep+IC / book age / …).
    # Dispatcher already existed in catalog; wire into verify-research.
    for receipt_err in northset_receipt_honesty_errors(families.get("northset")):
        errors.append(receipt_err)
    errors.extend(northset_top_level_claim_honesty_errors(families.get("northset")))
    errors.extend(northset_shape_and_session_l2_floors_honesty_errors(families.get("northset")))
    # Soft kyle_ofi nest residual_flow / dispersion honesty (research diagnostic):
    # finite residual_* or dispersion stamps → research_only + claim + no Sharpe keys.
    errors.extend(kyle_ofi_nest_honesty_errors(families.get("northset")))
    # mean_microprice_weight_balance ∈[0,1] — northset + candle_order_book parity
    errors.extend(mean_microprice_weight_balance_honesty_errors(families.get("northset")))
    errors.extend(mean_microprice_weight_balance_honesty_errors(families.get("candle_order_book")))
    # structure_finite_rate ∈[0,1] — northset aggregate + candle finite_rate_* companions
    errors.extend(structure_finite_rate_honesty_errors(families.get("northset")))
    errors.extend(structure_finite_rate_honesty_errors(families.get("candle_order_book")))
    errors.extend(
        candle_order_book_dgp_data_source_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(
        candle_order_book_family_provenance_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(candle_order_book_sizing_honesty_errors(families.get("candle_order_book")))
    errors.extend(candle_order_book_ic_method_honesty_errors(families.get("candle_order_book")))
    errors.extend(candle_feature_cols_ic_honesty_errors(families.get("candle_order_book")))
    errors.extend(
        candle_feature_cols_ic_completeness_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(
        candle_microprice_weight_balance_ic_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(
        candle_mwb_scored_implies_mean_unit_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(
        candle_notional_imbalance_ic_implies_mean_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(mean_notional_imbalance_honesty_errors(families.get("candle_order_book")))
    errors.extend(
        candle_spread_over_mid_ic_implies_mean_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(
        candle_depth_imbalance_ic_implies_mean_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(
        candle_structure_ic_implies_mean_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(candle_all_finite_rate_prefix_honesty_errors(families.get("candle_order_book")))
    errors.extend(
        candle_structure_finite_rate_covers_companions_honesty_errors(
            families.get("candle_order_book")
        )
    )
    errors.extend(
        candle_ofi_qp_slope_ic_implies_mean_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(
        candle_ofi_and_queue_imbalance_means_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(candle_feature_ofi_finite_honesty_errors(families.get("candle_order_book")))
    errors.extend(
        candle_feature_cols_ic_implies_mean_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(candle_order_book_claim_honesty_errors(families.get("candle_order_book")))
    errors.extend(robinhood_plus_claim_honesty_errors(families.get("robinhood_plus")))
    errors.extend(candle_all_ic_pearson_unit_honesty_errors(families.get("candle_order_book")))
    errors.extend(candle_all_ic_p_unit_honesty_errors(families.get("candle_order_book")))
    errors.extend(candle_all_ic_t_finite_honesty_errors(families.get("candle_order_book")))
    errors.extend(candle_all_ic_n_dates_nonneg_honesty_errors(families.get("candle_order_book")))
    errors.extend(candle_join_coverage_and_chain_honesty_errors(families.get("candle_order_book")))
    errors.extend(candle_spread_alias_honesty_errors(families.get("candle_order_book")))
    errors.extend(
        candle_spread_bps_ic_matches_spread_over_mid_ic_honesty_errors(
            families.get("candle_order_book")
        )
    )
    errors.extend(candle_frac_and_spread_x_honesty_errors(families.get("candle_order_book")))
    errors.extend(candle_direction_mean_honesty_errors(families.get("candle_order_book")))
    errors.extend(
        candle_wick_skew_and_body_ret_means_honesty_errors(families.get("candle_order_book"))
    )
    errors.extend(
        candle_signed_vol_x_imbalance_mean_honesty_errors(families.get("candle_order_book"))
    )
    # join_coverage ∈(0,1] or [book_join_coverage_floor,1] — northset + candle_order_book
    # fail-closed fuse honesty (research diagnostic only; never live Sharpe).
    errors.extend(join_coverage_honesty_errors(families.get("northset")))
    errors.extend(join_coverage_honesty_errors(families.get("candle_order_book")))
    errors.extend(mean_microprice_minus_mid_honesty_errors(families.get("northset")))
    errors.extend(mean_microprice_minus_mid_honesty_errors(families.get("candle_order_book")))
    errors.extend(book_age_seconds_honesty_errors(families.get("candle_order_book")))
    errors.extend(depth_shape_finite_rate_honesty_errors(families.get("candle_order_book")))
    errors.extend(mean_tob_size_share_honesty_errors(families.get("northset")))
    errors.extend(mean_tob_size_share_honesty_errors(families.get("candle_order_book")))
    errors.extend(candle_log_tick_spacing_finite_honesty_errors(families.get("candle_order_book")))
    # mean_tob_notional_share ∈(0,1] when finite — ≠ mean_tob_size_share (Commander #62)
    for mtn_err in mean_tob_notional_share_honesty_errors(families.get("northset")):
        errors.append(mtn_err)
    errors.extend(mean_close_mid_abs_rel_honesty_errors(families.get("northset")))
    errors.extend(mean_close_mid_abs_rel_honesty_errors(families.get("candle_order_book")))
    errors.extend(mean_candle_dir_x_imbalance_honesty_errors(families.get("candle_order_book")))
    errors.extend(mean_queue_priority_honesty_errors(families.get("northset")))
    errors.extend(mean_queue_priority_honesty_errors(families.get("candle_order_book")))
    errors.extend(mean_notional_imbalance_honesty_errors(families.get("northset")))
    errors.extend(size_concentration_top_honesty_errors(families.get("northset")))
    errors.extend(size_concentration_top_honesty_errors(families.get("candle_order_book")))
    errors.extend(mean_depth_imbalance_honesty_errors(families.get("northset")))
    errors.extend(mean_depth_imbalance_abs_honesty_errors(families.get("northset")))
    errors.extend(mean_depth_imbalance_honesty_errors(families.get("candle_order_book")))
    errors.extend(candle_spread_bps_nonneg_honesty_errors(families.get("candle_order_book")))
    errors.extend(candle_spread_bps_nonneg_honesty_errors(families.get("northset")))
    errors.extend(candle_log_slopes_finite_honesty_errors(families.get("candle_order_book")))
    errors.extend(mean_depth_imbalance_abs_honesty_errors(families.get("candle_order_book")))
    errors.extend(
        northset_metrics_required_keys_finite_when_present_honesty_errors(families.get("northset"))
    )
    errors.extend(mean_imbalance_top_honesty_errors(families.get("northset")))
    errors.extend(mean_imbalance_top_honesty_errors(families.get("candle_order_book")))
    errors.extend(northset_shape_columns_ensured_rates_honesty_errors(families.get("northset")))
    errors.extend(
        northset_metrics_required_finite_ok_rates_honesty_errors(families.get("northset"))
    )
    errors.extend(northset_depth_notional_spread_over_mid_honesty_errors(families.get("northset")))
    errors.extend(northset_price_slope_tick_top_levels_honesty_errors(families.get("northset")))
    errors.extend(northset_fwd_ret_after_sweep_honesty_errors(families.get("northset")))
    return errors


def _family_state(notebook: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    families = notebook.get("families")
    if not isinstance(families, dict) or not families:
        return {}, ["families_missing"]
    errors = _family_shape_errors(families)
    errors.extend(_family_fanin_errors(notebook, families))
    return families, errors


def _ranking_link_errors(notebook: dict[str, Any], families: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    # Soft H1/H2 oracle ranking consistency (Day Wave 31 research diagnostic):
    # finite oracle_raw p_ic / ls_p must mint H1/H2 discovery rows; the two
    # gates are independent. Non-finite / missing oracle → skip.
    errors.extend(
        h1_hypothesis_consistency_errors(notebook.get("rankers"), notebook.get("hypotheses"))
    )
    errors.extend(
        h2_hypothesis_consistency_errors(notebook.get("rankers"), notebook.get("hypotheses"))
    )
    # Data-snooping battery over the ranker universe: structure/ordering honesty
    # and the H46 consistency gate (finite consistent SPA p ⇒ H46 discovery).
    errors.extend(ranking_data_snooping_honesty_errors(families.get("ranking")))
    errors.extend(
        h99_hypothesis_consistency_errors(families.get("ranking"), notebook.get("hypotheses"))
    )
    return errors


def _scorecard_errors(notebook: dict[str, Any], families: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    # Benchmark scorecard: required families, honest flags, and forged-flag
    # detection against the family blobs (a True scorecard flag on an empty or
    # forbidden-key blob fails closed instead of validating a hollow bench).
    scorecard = notebook.get("scorecard")
    if not isinstance(scorecard, dict):
        errors.append("scorecard_missing")
    else:
        allowed_scorecard = REQUIRED_BENCHMARK_FAMILIES | OPTIONAL_BENCHMARK_FAMILIES
        missing_scorecard = sorted(REQUIRED_BENCHMARK_FAMILIES - scorecard.keys())
        unknown_scorecard = sorted(set(scorecard.keys()) - allowed_scorecard)
        if missing_scorecard:
            errors.append(f"scorecard_families_missing:{','.join(missing_scorecard)}")
        if unknown_scorecard:
            errors.append(f"scorecard_families_unknown:{','.join(unknown_scorecard)}")
        for fam_name, item in scorecard.items():
            if not isinstance(item, dict):
                errors.append(f"scorecard_invalid:{fam_name}")
                continue
            if item.get("claim") != "research_metric_only":
                errors.append(f"scorecard_claim_invalid:{fam_name}")
            flags = ("executed", "nonempty", "finite_observation", "forbidden_metrics_absent")
            if any(item.get(flag) is not True for flag in flags):
                errors.append(f"scorecard_invalid:{fam_name}")
            blob = families.get(fam_name) if isinstance(families, dict) else None
            if item.get("executed") is True and not family_blob_executed(blob):
                errors.append(f"scorecard_executed_flag_forged:{fam_name}")
            if item.get("nonempty") is True and not family_blob_nonempty(blob):
                errors.append(f"scorecard_nonempty_flag_forged:{fam_name}")
            if item.get("finite_observation") is True and not family_blob_has_finite_observation(
                blob
            ):
                errors.append(f"scorecard_finite_observation_flag_forged:{fam_name}")
            if item.get("forbidden_metrics_absent") is True and not (
                family_blob_forbidden_metrics_absent(blob)
            ):
                errors.append(f"scorecard_forbidden_flag_forged:{fam_name}")
            if (
                fam_name == "tail"
                and item.get("tail_var_battery_ok") is True
                and not tail_var_battery_keys_present(blob)
            ):
                errors.append("scorecard_tail_var_battery_flag_forged:tail")
            if (
                fam_name == "tail"
                and item.get("tail_es_battery_ok") is True
                and not tail_es_battery_keys_present(blob)
            ):
                errors.append("scorecard_tail_es_battery_flag_forged:tail")
            if (
                fam_name == "distribution"
                and item.get("dist_crps_eprocess_ok") is True
                and not dist_crps_eprocess_keys_present(blob)
            ):
                errors.append("scorecard_dist_crps_eprocess_flag_forged:distribution")
    return errors


def _artifact_errors(
    path: Path,
    notebook: dict[str, Any],
    run_id: object,
    scorecard: object,
) -> list[str]:
    errors: list[str] = []
    artifacts = notebook.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts_missing")
    else:
        receipt_root = path.resolve().parent
        # Canonical latest pointers must stay named latest.json / latest.md; a
        # malformed or repointed value fails closed instead of validating a
        # different artifact as the latest receipt.
        for key, expected_name, mismatch_token in (
            ("json", "latest.json", "latest_json_artifact_pointer_mismatch"),
            ("markdown", "latest.md", "latest_markdown_artifact_pointer_mismatch"),
        ):
            pointer = artifacts.get(key)
            if pointer is None:
                continue
            if not isinstance(pointer, str) or not pointer.strip():
                errors.append(f"invalid_artifact_pointer:{key}")
                continue
            if Path(pointer).name != expected_name:
                errors.append(mismatch_token)
        # Hash fields are checked for shape wherever present (file existence is
        # checked separately below).
        for digest_key in ("immutable_json_sha256", "immutable_markdown_sha256"):
            digest = artifacts.get(digest_key)
            if digest is None:
                continue
            if not _is_sha256(digest):
                errors.append(f"invalid_artifact_hash:{digest_key}")
        for key in ("immutable_json", "immutable_markdown"):
            artifact = artifacts.get(key)
            raw_path = Path(artifact) if isinstance(artifact, str) else None
            artifact_path = (
                raw_path if raw_path is None or raw_path.is_absolute() else receipt_root / raw_path
            )
            try:
                inside_receipt = (
                    artifact_path is not None
                    and artifact_path.resolve().is_relative_to(receipt_root)
                )
            except OSError:
                inside_receipt = False
            if not inside_receipt:
                errors.append(f"artifact_outside_receipt_root:{key}")
            if artifact_path is None or not artifact_path.is_file():
                errors.append(f"artifact_missing:{key}")
        immutable_json = artifacts.get("immutable_json")
        immutable_path = (
            receipt_root / immutable_json
            if isinstance(immutable_json, str) and not Path(immutable_json).is_absolute()
            else Path(immutable_json)
            if isinstance(immutable_json, str)
            else None
        )
        if isinstance(immutable_json, str) and run_id and Path(immutable_json).stem != run_id:
            errors.append("immutable_artifact_run_id_mismatch")
        if immutable_path is not None and immutable_path.is_file():
            expected_json_hash = artifacts.get("immutable_json_sha256")
            if not _is_sha256(expected_json_hash):
                errors.append("immutable_json_hash_missing")
            expected_md_hash = artifacts.get("immutable_markdown_sha256")
            markdown_value = artifacts.get("immutable_markdown")
            markdown_path = (
                receipt_root / markdown_value
                if isinstance(markdown_value, str) and not Path(markdown_value).is_absolute()
                else Path(markdown_value)
                if isinstance(markdown_value, str)
                else None
            )
            if not _is_sha256(expected_md_hash):
                errors.append("immutable_markdown_hash_missing")
            elif markdown_path is None or not markdown_path.is_file():
                errors.append("immutable_markdown_hash_unverifiable")
            elif hash_file(markdown_path) != expected_md_hash:
                errors.append("immutable_markdown_hash_mismatch")
            try:
                immutable = json.loads(immutable_path.read_text())
            except (OSError, json.JSONDecodeError):
                errors.append("immutable_json_unreadable")
            else:
                if (
                    isinstance(expected_json_hash, str)
                    and _is_sha256(expected_json_hash)
                    and _receipt_digest(immutable) != expected_json_hash
                ):
                    errors.append("immutable_json_hash_mismatch")
                if immutable.get("provenance", {}).get("run_id") != run_id:
                    errors.append("immutable_provenance_mismatch")
                if immutable.get("scorecard") != scorecard:
                    errors.append("immutable_scorecard_mismatch")
                if immutable != notebook:
                    errors.append("immutable_receipt_mismatch")
    return errors


def verify_research_artifact(path: Path) -> dict[str, Any]:
    """Validate a persisted notebook without interpreting metrics as alpha."""
    path = Path(path)
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        unreadable: UnreadableResearchVerification = {
            "valid": False,
            "path": str(path),
            "errors": [f"unreadable:{exc}"],
        }
        return cast(dict[str, Any], unreadable)
    if not isinstance(loaded, dict):
        not_object: UnreadableResearchVerification = {
            "valid": False,
            "path": str(path),
            "errors": ["notebook_not_object"],
        }
        return cast(dict[str, Any], not_object)
    notebook: dict[str, Any] = loaded
    errors: list[str] = []
    errors.extend(_notebook_identity_errors(notebook))
    errors.extend(_ranker_errors(notebook))
    errors.extend(_hypothesis_record_errors(notebook))
    _provenance, run_id, provenance_errors = _provenance_state(notebook)
    errors.extend(provenance_errors)
    families, family_errors = _family_state(notebook)
    errors.extend(family_errors)
    errors.extend(_ranking_link_errors(notebook, families))
    errors.extend(_scorecard_errors(notebook, families))
    errors.extend(_artifact_errors(path, notebook, run_id, notebook.get("scorecard")))
    scorecard = notebook.get("scorecard")
    verified: ResearchVerification = {
        "valid": not errors,
        "path": str(path),
        "run_id": run_id,
        "scorecard_families": len(scorecard) if isinstance(scorecard, dict) else 0,
        "errors": errors,
        "claim": "research_only",
    }
    return cast(dict[str, Any], verified)
