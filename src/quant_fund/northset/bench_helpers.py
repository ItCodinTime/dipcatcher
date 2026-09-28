"""Pure helpers extracted from ``bench_northset`` (McCabe ratchet).

Behaviour-preserving splits only — no scoring changes. Unit-tested in
``tests/unit/microstructure/test_northset_bench_helpers.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
import polars as pl

from quant_fund.microstructure import book_metrics as book_metrics_mod


def nanmean_finite(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """NaN-safe mean; empty / all-non-finite → NaN."""
    finite = np.asarray(arr, dtype=float)
    if finite.size == 0:
        return float("nan")
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def nanmean_col(frame: pl.DataFrame, column: str) -> float:
    """Mean of ``column`` when present; NaN if missing or all-non-finite."""
    if column not in frame.columns:
        return float("nan")
    return nanmean_finite(frame[column].to_numpy().astype(float))


def float_col_or_empty(frame: pl.DataFrame, column: str) -> np.ndarray:
    """Float ndarray for ``column``, or empty array when absent."""
    if column not in frame.columns:
        return np.array([])
    return frame[column].to_numpy().astype(float)


def rate_or_nan(
    columns: set[str] | list[str] | tuple[str, ...],
    required: tuple[str, ...],
    rows: list[dict[str, float]],
    rate_fn: Callable[[list[dict[str, float]]], float],
) -> float:
    """Finite-rate helper: missing column class → NaN (never a fake 0.0)."""
    if not set(required).issubset(columns):
        return float("nan")
    return float(rate_fn(rows))


def structure_finite_rates(
    book: pl.DataFrame,
    shape_rows: list[dict[str, float]],
) -> dict[str, float]:
    """LOB structure finite-rate pack used by the Northset receipt."""
    cols = book.columns
    return {
        "depth_shape_finite_rate": rate_or_nan(
            cols,
            (*book_metrics_mod.DEPTH_SHAPE_FIELDS, "n_bid_levels", "n_ask_levels"),
            shape_rows,
            book_metrics_mod.depth_shape_finite_rate,
        ),
        "concentration_top_finite_rate": rate_or_nan(
            cols,
            book_metrics_mod.SIDE_STRUCTURE_FIELDS,
            shape_rows,
            book_metrics_mod.concentration_top_finite_rate,
        ),
        "queue_priority_finite_rate": rate_or_nan(
            cols,
            book_metrics_mod.QUEUE_STRUCTURE_FIELDS,
            shape_rows,
            book_metrics_mod.queue_priority_finite_rate,
        ),
        "side_notional_finite_rate": rate_or_nan(
            cols,
            book_metrics_mod.SIDE_NOTIONAL_FIELDS,
            shape_rows,
            book_metrics_mod.side_notional_finite_rate,
        ),
        "tob_size_share_finite_rate": rate_or_nan(
            cols,
            book_metrics_mod.TOB_SHARE_FIELDS,
            shape_rows,
            book_metrics_mod.tob_size_share_finite_rate,
        ),
    }


def enforce_structure_floors(
    rates: Mapping[str, float],
    floors: Mapping[str, float | None],
) -> None:
    """Fail-closed when a measured rate sits below its configured floor."""
    for rate_name, floor_value in floors.items():
        if floor_value is None:
            continue
        rate_value = float(rates[rate_name])
        if not np.isfinite(rate_value) or rate_value + 1e-12 < float(floor_value):
            raise ValueError(
                f"{rate_name}={rate_value} below floor {floor_value} (fail-closed data contract)"
            )


def metrics_required_finite_ok(shape_rows: list[dict[str, float]]) -> bool:
    """True when the first shape row passes ``assert_metrics_required_finite``."""
    if not shape_rows:
        return False
    try:
        book_metrics_mod.assert_metrics_required_finite(shape_rows[0])
    except (TypeError, ValueError):
        return False
    return True


def join_age_metrics(fused: pl.DataFrame) -> tuple[float, float, float]:
    """Return ``(join_coverage, mean_book_age_seconds, max_book_age_seconds)``."""
    if "join_coverage" in fused.columns and fused.height:
        join_coverage = float(fused["join_coverage"][0])
    else:
        join_coverage = float("nan")
    if "book_age_seconds" in fused.columns and fused.height:
        ages = fused["book_age_seconds"].to_numpy().astype(float)
        mean_age = float(np.nanmean(ages)) if ages.size else float("nan")
        max_age = float(np.nanmax(ages)) if ages.size else float("nan")
    else:
        mean_age = float("nan")
        max_age = float("nan")
    return join_coverage, mean_age, max_age


def evidence_provenance(
    *,
    bar_source: str,
    book_source: str,
    book_dgp: str,
    session_l2_enabled: bool,
) -> tuple[str, str]:
    """Return ``(evidence_label, family_dgp)`` honesty stamps for the receipt."""
    bars_synthetic = bar_source.lower() == "synthetic"
    book_synthetic = book_source.lower() in {
        "synthetic",
        "synthetic_lob",
        "synthetic_reconstruction",
    }
    session_synthetic = bool(session_l2_enabled)
    if bars_synthetic and book_synthetic:
        return "SYNTHETIC", "synthetic_lob"
    if not book_synthetic:
        # Candle label may stay SYNTHETIC; book provenance is authoritative.
        evidence_label = "SYNTHETIC" if bars_synthetic else bar_source
        return evidence_label, book_dgp
    if bars_synthetic or book_synthetic or session_synthetic:
        return "MIXED_SYNTHETIC_DERIVED", "mixed_sources"
    return bar_source, "empirical"


def cond_fwd_mean(scored: pl.DataFrame, flag: str) -> float:
    """Mean ``fwd_ret_1`` on rows where ``flag == 1.0``; NaN if empty/missing."""
    if flag not in scored.columns:
        return float("nan")
    sub = scored.filter(pl.col(flag) == 1.0)
    if sub.height == 0:
        return float("nan")
    mean = sub["fwd_ret_1"].mean()
    if mean is None:
        return float("nan")
    return float(cast(Any, mean))


def book_structure_mean_fields(book: pl.DataFrame) -> dict[str, float]:
    """Receipt means drawn from the L2 book panel (missing → NaN)."""
    keys = (
        "tob_size_share",
        "tob_notional_share",
        "notional_imbalance",
        "bid_size_concentration_top",
        "ask_size_concentration_top",
        "bid_depth",
        "ask_depth",
        "side_notional_proxy_bid",
        "side_notional_proxy_ask",
        "top_of_book_notional_proxy",
        "spread_over_mid",
        "bid_log_price_slope",
        "ask_log_price_slope",
        "bid_mean_log_tick_spacing",
        "ask_mean_log_tick_spacing",
        "top_bid_size",
        "top_ask_size",
        "n_bid_levels",
        "n_ask_levels",
        "queue_priority_proxy",
        "ask_queue_priority_proxy",
    )
    return {f"mean_{key}": nanmean_col(book, key) for key in keys}


def sweep_evidence_receipt_fields(
    sweep_evidence: Mapping[str, Any],
    *,
    primary_test_id_default: str,
) -> dict[str, Any]:
    """Flatten sweep-evidence nests into top-level receipt keys (pure)."""
    out: dict[str, Any] = {}
    for row in sweep_evidence["event_studies"]:
        if row["horizon"] != 1:
            continue
        prefix = "sweep_reject" if row["signal"] == "sweep_reject_signed" else "sweep_follow"
        out[f"{prefix}_event_p"] = row["p_value"]
        out[f"{prefix}_event_t"] = row["hac_t"]
        out[f"{prefix}_event_mean_bps"] = row["mean_excess_bps"]
        out[f"{prefix}_cost_adjusted_mean_bps"] = row["cost_adjusted_mean_bps"]
        out[f"{prefix}_cost_adjusted_p"] = row["cost_adjusted_p_greater"]
        out[f"{prefix}_fold_positive_fraction"] = row["positive_fraction"]
    placebos = sweep_evidence["permutation_placebos"]
    out["sweep_reject_placebo_p"] = placebos["sweep_reject_signed"]["placebo_p_value"]
    out["sweep_reject_placebo_observed_ic"] = placebos["sweep_reject_signed"]["observed_mean_ic"]
    out["sweep_follow_placebo_p"] = placebos["sweep_follow_signed"]["placebo_p_value"]
    out["sweep_follow_placebo_observed_ic"] = placebos["sweep_follow_signed"]["observed_mean_ic"]
    matched_controls = sweep_evidence["matched_controls"]
    for prefix, signal in (
        ("sweep_reject", "sweep_reject_signed"),
        ("sweep_follow", "sweep_follow_signed"),
    ):
        control = matched_controls.get(signal) or {}
        out[f"{prefix}_control_diff_p"] = float(control.get("p_value", float("nan")))
        out[f"{prefix}_control_diff_t"] = float(control.get("hac_t", float("nan")))
        out[f"{prefix}_control_diff_mean_bps"] = float(control.get("mean_diff_bps", float("nan")))
        out[f"{prefix}_control_sample_adequate"] = bool(control.get("sample_adequate", False))
        out[f"{prefix}_control_n_dates"] = int(control.get("n_dates", 0))
        liq = (sweep_evidence.get("liquidity_matched_controls") or {}).get(signal) or {}
        out[f"{prefix}_liq_control_diff_p"] = float(liq.get("p_value", float("nan")))
        out[f"{prefix}_liq_control_diff_t"] = float(liq.get("hac_t", float("nan")))
        out[f"{prefix}_liq_control_diff_mean_bps"] = float(liq.get("mean_diff_bps", float("nan")))
        out[f"{prefix}_liq_control_sample_adequate"] = bool(liq.get("sample_adequate", False))
        clustered = (sweep_evidence.get("name_clustered") or {}).get(signal) or {}
        out[f"{prefix}_name_cluster_p"] = float(clustered.get("p_value", float("nan")))
        out[f"{prefix}_name_cluster_t"] = float(clustered.get("t_stat", float("nan")))
        two_way = (sweep_evidence.get("two_way_clustered") or {}).get(signal) or {}
        out[f"{prefix}_two_way_cluster_p"] = float(two_way.get("p_value", float("nan")))
        out[f"{prefix}_two_way_cluster_t"] = float(two_way.get("t_stat", float("nan")))
        out[f"{prefix}_two_way_wild_p"] = float(two_way.get("wild_bootstrap_p", float("nan")))
        gap = (sweep_evidence.get("overnight_gaps") or {}).get(signal) or {}
        out[f"{prefix}_overnight_gap_p"] = float(gap.get("p_value", float("nan")))
        out[f"{prefix}_overnight_gap_t"] = float(gap.get("hac_t", float("nan")))
        out[f"{prefix}_overnight_gap_mean_bps"] = float(gap.get("mean_gap_bps", float("nan")))
    follow_oot = (sweep_evidence.get("oot_holdouts") or {}).get("sweep_follow_signed") or {}
    out["sweep_follow_oot_holdout_mean_bps"] = float(
        follow_oot.get("holdout_mean_bps", float("nan"))
    )
    out["sweep_follow_oot_insample_mean_bps"] = float(
        follow_oot.get("insample_mean_bps", float("nan"))
    )
    out["sweep_follow_oot_same_sign"] = float(follow_oot.get("same_sign", float("nan")))
    primary = sweep_evidence.get("primary_test") or {}
    out["sweep_primary_test_id"] = str(primary.get("id") or primary_test_id_default)
    ledger = sweep_evidence.get("trial_ledger") or {}
    out["sweep_n_counted_trials"] = int(ledger.get("n_counted_trials", 0))
    adv = sweep_evidence.get("adv_participation") or {}
    out["sweep_median_event_adv_participation"] = float(
        adv.get("median_participation", float("nan"))
    )
    out["sweep_p95_event_adv_participation"] = float(adv.get("p95_participation", float("nan")))
    out["sweep_inference_index"] = str(
        sweep_evidence.get("inference_index") or "calendar_including_idle_zeros"
    )
    out["sweep_two_way_inference_index"] = "event_rows_not_calendar_zeros"
    out["sweep_overnight_gap_method"] = "event_close_to_next_open"
    coverage = sweep_evidence.get("coverage") or {}
    out["sweep_n_ohlc_quarantined"] = int(coverage.get("n_ohlc_quarantined", 0))
    return out
