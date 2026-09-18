"""Northset research bench: candlesticks + L2 books.

Identities, date-level IC, OHLC vol estimators, Kyle/Roll/Corwin–Schultz,
Amihud, OFI, VPIN, session RV/jumps. No Sharpe. Research-only.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.metrics.cross_section import DateICResult, date_ic_series
from quant_fund.metrics.scoring import qlike
from quant_fund.microstructure import book_metrics as book_metrics_mod
from quant_fund.microstructure.candle_book_features import attach_candle_book_features
from quant_fund.microstructure.synthetic_lob import (
    aggregate_session_book_to_daily,
    ensure_book_panel_shape_columns,
    synthesize_l2_from_bars,
    synthesize_session_l2,
)
from quant_fund.northset.candles import geometry_rates
from quant_fund.northset.data_view import canonical_northset_bars
from quant_fund.northset.estimators import (
    abdi_ranaldo_spread,
    amihud_illiquidity,
    corwin_schultz_spread,
    dm_range_vs_park,
    dm_split_vs_park,
    garman_klass_vs_close_to_close,
    kyle_lambda,
    lag1_corr,
    ohlc_variance_frame,
    order_flow_imbalance,
    overnight_plus_oc_vs_close_to_close,
    overnight_share,
    parkinson_vs_close_to_close,
    queue_imbalance,
    realized_semivariance,
    rogers_satchell_vs_close_to_close,
    roll_spread,
    session_bipower_jump,
    session_realized_variance,
    session_vpin,
    true_range_frame,
    volume_over_range,
    vpin_proxy,
    yang_zhang_variance,
    yang_zhang_vs_close_to_close,
)
from quant_fund.northset.identities import (
    book_uncrossed_rate,
    gap_finite_rate,
    ohlc_identity_rate,
    session_candles_from_daily,
    session_chain_rate,
    session_reconstructs_daily_rate,
    session_volume_conservation_rate,
    validate_session_book_counts,
)
from quant_fund.northset.sweep_research import PRIMARY_EXECUTABLE_TEST, sweep_evidence_battery
from quant_fund.northset.sweeps import (
    liquidity_sweep_frame,
    sweep_output_columns,
    sweep_rates,
)
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent


def enforce_session_l2_identity_floors(
    *,
    ohlc_identity_rate: float,
    session_ohlc_identity_rate: float,
    session_reconstructs_daily_rate: float,
    session_volume_conservation_rate: float,
    session_chain_rate: float,
    book_uncrossed_rate: float,
    floor: float = 0.99,
) -> dict[str, float]:
    """Fail-closed identity gates for the session-L2 path (ADR-021).

    SYNTHETIC session candles + books must preserve OHLC / volume / chain /
    uncrossed-book contracts. Rates below ``floor`` are data-contract bugs,
    not alpha. Floor default 0.99 (allow tiny float noise; honest SYNTHETIC
    paths hit 1.0).
    """
    if not (0.0 <= float(floor) <= 1.0) or floor != floor:
        raise ValueError(f"identity floor must be in [0, 1], got {floor!r}")
    rates = {
        "ohlc_identity_rate": float(ohlc_identity_rate),
        "session_ohlc_identity_rate": float(session_ohlc_identity_rate),
        "session_reconstructs_daily_rate": float(session_reconstructs_daily_rate),
        "session_volume_conservation_rate": float(session_volume_conservation_rate),
        "session_chain_rate": float(session_chain_rate),
        "book_uncrossed_rate": float(book_uncrossed_rate),
    }
    failures: list[str] = []
    for name, value in rates.items():
        if value != value:  # NaN
            failures.append(f"{name}=nan")
        elif value + 1e-12 < float(floor):
            failures.append(f"{name}={value:.6g}<{floor}")
    if failures:
        raise ValueError("session-L2 identity gate failed (fail-closed): " + ", ".join(failures))
    return rates


_IC_FEATURES = (
    "imbalance_top",
    "imbalance_depth",
    "microprice_minus_mid_bps",
    "ofi",
    "ofi_lag",
    "wick_skew",
    "candle_body_ret",
    "candle_gap",
    "spread_bps",
    "amihud",
    "candle_dir_x_imbalance",
    "bid_log_size_slope",
    "queue_imbalance",
    "vpin",
    "session_ofi_sum",
    "session_imbalance_mean",
    "session_book_vpin",
    "session_close_imbalance",
    "session_close_spread_bps",
    "session_close_micro_bps",
    "session_close_mid",
    "session_close_bid_depth",
    "session_close_ask_depth",
    "session_spread_bps_mean",
    "close_location_value",
    "volume_over_range",
    "sweep_reject_signed",
    "sweep_follow_signed",
    "sweep_depth_signed",
)

# Receipt keys for session-L2 path / last-snap means (soft-verify companions).
SESSION_RECEIPT_KEYS: frozenset[str] = frozenset(
    {
        "session_ofi_sum_mean",
        "mean_session_ofi_abs_sum",
        "session_book_vpin_mean",
        "mean_session_imbalance_mean",
        "mean_session_imbalance_std",
        "mean_session_spread_bps_mean",
        "mean_session_close_spread_bps",
        "mean_session_close_imbalance",
        "mean_session_close_micro_bps",
        "mean_session_close_mid",
        "mean_session_close_bid_depth",
        "mean_session_close_ask_depth",
        "mean_session_book_snaps",
    }
)

# CLI northset + research family echo every SESSION_RECEIPT_KEYS entry (no blob-only exceptions).
SESSION_RECEIPT_KEYS_CLI_ECHO: frozenset[str] = SESSION_RECEIPT_KEYS

# Scoped northset CLI echo honesty (Sergeant): every stamped non-discovery /
# non-kyle receipt key must be either CLI-echoed or explicitly blob-only.
# Discovery IC companions (*_mean_ic / *_p_ic / …) and kyle_* stay out of scope.
# CLI short aliases: mean_book_age_seconds → mean_book_age_s, max_book_age_seconds → max_book_age_s.

NORTHSET_CLI_ECHO_KEY_ALIASES: dict[str, str] = {
    "mean_book_age_seconds": "mean_book_age_s",
    "max_book_age_seconds": "max_book_age_s",
}

# Must appear in `dipcatcher northset` CLI source (key= or get('key')).
# Core rates / structure / session / spread / shape — not the full mean_* surface
# (CoS owns mean_* echo completeness separately).
NORTHSET_CLI_ECHO_REQUIRED: frozenset[str] = frozenset(
    {
        "join_coverage",
        "book_source",
        "ohlc_identity_rate",
        "session_ohlc_identity_rate",
        "book_uncrossed_rate",
        "session_chain_rate",
        "session_reconstructs_daily_rate",
        "session_volume_conservation_rate",
        "gap_finite_rate",
        "structure_finite_rate",
        "depth_shape_finite_rate",
        "concentration_top_finite_rate",
        "queue_priority_finite_rate",
        "side_notional_finite_rate",
        "tob_size_share_finite_rate",
        "mean_book_age_seconds",
        "max_book_age_seconds",
        "mean_quoted_spread",
        "mean_effective_spread",
        "mean_half_spread",
        "mean_half_spread_bps",
        "mean_spread_bps",
        "mean_close_mid_abs_rel",
        "mean_microprice_weight_balance",
        "mean_microprice_minus_mid",
        "mean_tob_size_share",
        "mean_tob_notional_share",
        "mean_imbalance_top",
        "mean_depth_imbalance",
        "mean_queue_priority_proxy",
        "mean_n_bid_levels",
        "mean_n_ask_levels",
        "mean_bid_log_size_slope",
        "mean_ask_log_size_slope",
        "mean_bid_log_price_slope",
        "mean_ask_log_price_slope",
        "mean_bid_mean_log_tick_spacing",
        "mean_ask_mean_log_tick_spacing",
        *SESSION_RECEIPT_KEYS,
    }
)

# Explicitly allowed to stay receipt/blob-only (not CLI key= lines).
NORTHSET_RECEIPT_BLOB_ONLY: frozenset[str] = frozenset(
    {
        "book_dgp",
        "book_join_coverage_floor",
        "book_panel_path",
        "component_sources",
        "dgp",
        "dm_gk_vs_park_p",
        "dm_gk_vs_park_preferred",
        "dm_gk_vs_park_stat",
        "dm_rs_vs_park_p",
        "dm_rs_vs_park_preferred",
        "dm_rs_vs_park_stat",
        "dm_split_vs_park_p",
        "dm_split_vs_park_preferred",
        "dm_split_vs_park_stat",
        "doji_rate",
        "engulfing_rate",
        "family",
        "hammer_rate",
        "impact_estimator_scope",
        "impact_proxy_warning",
        "label",
        "marubozu_rate",
        "mean_sweep_depth_high",
        "mean_sweep_depth_low",
        "mid_lag1_n_securities",
        "n_bars",
        "n_fused",
        "n_scored",
        "n_session_candles",
        "ofi_lag1_n_securities",
        "price_basis",
        "product",
        "queue_priority_finite_floor",
        "research_only",
        "return_basis",
        "roll_n_securities",
        "session_bulk_vpin",
        "session_vpin_method",
        "session_l2_identity_floor",
        "session_mean_jump_ratio",
        "shooting_star_rate",
        "side_notional_finite_floor",
        "spinning_top_rate",
        "sweep_any_rate",
        "sweep_both_rate",
        "sweep_eligible_rate",
        "sweep_evidence",
        "sweep_evidence_scope",
        "sweep_follow_control_diff_p",
        "sweep_follow_control_diff_t",
        "sweep_follow_control_sample_adequate",
        "sweep_follow_liq_control_diff_p",
        "sweep_follow_liq_control_diff_t",
        "sweep_follow_liq_control_sample_adequate",
        "sweep_follow_name_cluster_p",
        "sweep_follow_name_cluster_t",
        "sweep_follow_two_way_cluster_p",
        "sweep_follow_two_way_cluster_t",
        "sweep_follow_two_way_wild_p",
        "sweep_follow_overnight_gap_mean_bps",
        "sweep_follow_overnight_gap_p",
        "sweep_follow_overnight_gap_t",
        "sweep_follow_oot_insample_mean_bps",
        "sweep_follow_oot_same_sign",
        "sweep_p95_event_adv_participation",
        "sweep_reject_liq_control_diff_p",
        "sweep_reject_liq_control_diff_t",
        "sweep_reject_liq_control_sample_adequate",
        "sweep_reject_name_cluster_p",
        "sweep_reject_name_cluster_t",
        "sweep_reject_two_way_cluster_p",
        "sweep_reject_two_way_cluster_t",
        "sweep_reject_two_way_wild_p",
        "sweep_reject_overnight_gap_mean_bps",
        "sweep_reject_overnight_gap_p",
        "sweep_reject_overnight_gap_t",
        "sweep_follow_cost_adjusted_mean_bps",
        "sweep_follow_cost_adjusted_p",
        "sweep_follow_event_p",
        "sweep_follow_event_t",
        "sweep_follow_placebo_observed_ic",
        "sweep_follow_placebo_p",
        "sweep_high_follow_share",
        "sweep_high_rate",
        "sweep_high_reclaim_share",
        "sweep_low_follow_share",
        "sweep_low_rate",
        "sweep_low_reclaim_share",
        "sweep_min_fold_positive_fraction",
        "sweep_reject_control_diff_p",
        "sweep_reject_control_diff_t",
        "sweep_reject_control_sample_adequate",
        "sweep_reject_cost_adjusted_p",
        "sweep_reject_event_p",
        "sweep_reject_event_t",
        "sweep_reject_placebo_observed_ic",
        "sweep_reject_placebo_p",
        "tob_size_share_finite_floor",
        "use_session_l2",
        "vpin_method",
        "yang_zhang_qlike_scope",
        "sweep_inference_index",
        "sweep_n_ohlc_quarantined",
        "sweep_overnight_gap_method",
        "sweep_two_way_inference_index",
        "corwin_schultz_pair_scope",
    }
)

NORTHSET_CLI_ECHO_EXTRA: frozenset[str] = frozenset(
    {
        "abdi_ranaldo_spread",
        "amihud_mean",
        "book_hypothesis_eligible",
        "claim",
        "concentration_top_finite_floor",
        "corwin_schultz_spread",
        "data_source",
        "depth",
        "depth_shape_finite_floor",
        "garman_klass_qlike_vs_cc",
        "mean_ask_depth",
        "mean_ask_queue_priority_proxy",
        "mean_ask_size_concentration_top",
        "mean_bid_depth",
        "mean_bid_size_concentration_top",
        "mean_depth_imbalance_abs",
        "mean_fwd_ret_after_high_follow",
        "mean_fwd_ret_after_high_reclaim",
        "mean_fwd_ret_after_low_follow",
        "mean_fwd_ret_after_low_reclaim",
        "mean_microprice_minus_mid_bps",
        "mean_notional_imbalance",
        "mean_side_notional_proxy_ask",
        "mean_side_notional_proxy_bid",
        "mean_spread_over_mid",
        "mean_top_ask_size",
        "mean_top_bid_size",
        "mean_top_of_book_notional_proxy",
        "mean_true_range",
        "metrics_required_finite_ok",
        "mid_lag1_corr",
        "n_session_book_rows",
        "n_sweep_high",
        "n_sweep_low",
        "ofi_lag1_corr",
        "overnight_plus_oc_qlike_vs_cc",
        "overnight_share",
        "parkinson_qlike_vs_cc",
        "queue_imbalance_mean",
        "rogers_satchell_qlike_vs_cc",
        "roll_spread",
        "semi_down",
        "semi_up",
        "session_book_hypothesis_eligible",
        "session_l2_identity_gate",
        "session_mean_bv",
        "session_mean_rv",
        "session_rv_qlike_vs_cc",
        "shape_columns_ensured",
        "sweep_follow_control_diff_mean_bps",
        "sweep_follow_event_mean_bps",
        "sweep_follow_fold_positive_fraction",
        "sweep_follow_liq_control_diff_mean_bps",
        "sweep_follow_oot_holdout_mean_bps",
        "sweep_median_event_adv_participation",
        "sweep_n_counted_trials",
        "sweep_primary_test_id",
        "sweep_reject_control_diff_mean_bps",
        "sweep_reject_liq_control_diff_mean_bps",
        "sweep_reject_cost_adjusted_mean_bps",
        "sweep_reject_event_mean_bps",
        "sweep_reject_fold_positive_fraction",
        "vpin_mean",
        "yang_zhang_qlike_vs_cc",
        "yang_zhang_variance",
    }
)

NORTHSET_RECEIPT_CLASSIFIED: frozenset[str] = (
    NORTHSET_CLI_ECHO_REQUIRED | NORTHSET_CLI_ECHO_EXTRA | NORTHSET_RECEIPT_BLOB_ONLY
)


def _is_discovery_or_kyle_receipt_key(key: str) -> bool:
    if key.startswith("ic_") or key.startswith("kyle_") or "kyle_ofi" in key:
        return True
    for suf in ("_mean_ic", "_mean_rank_ic", "_p_ic", "_t_ic", "_n_dates", "_pearson"):
        if key.endswith(suf):
            return True
    return False


# Scorecard metadata flags may ride along on family blobs (scorecard-validated
# separately); they are not northset receipt stamps and never need CLI echo.
_SCORECARD_METADATA_KEYS: frozenset[str] = frozenset(
    {"executed", "nonempty", "finite_observation", "forbidden_metrics_absent"}
)


def northset_receipt_key_classification_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: no silent new stamps outside REQUIRED ∪ EXTRA ∪ BLOB_ONLY.

    Discovery IC companions and kyle_* keys are out of scope. Research diagnostic
    only; never live Sharpe. New stamps must be classified (CLI echo or blob-only).
    """
    if not isinstance(blob, dict):
        return []
    unknown = sorted(
        k
        for k in blob
        if isinstance(k, str)
        and not _is_discovery_or_kyle_receipt_key(k)
        and k not in NORTHSET_RECEIPT_CLASSIFIED
        and k not in _SCORECARD_METADATA_KEYS
    )
    if not unknown:
        return []
    shown = unknown[:12]
    more = len(unknown) - len(shown)
    token = "northset_receipt_unclassified_keys:" + ",".join(shown)
    if more > 0:
        token += f",+{more}"
    return [token]


assert not (NORTHSET_CLI_ECHO_REQUIRED & NORTHSET_RECEIPT_BLOB_ONLY)
assert not (NORTHSET_CLI_ECHO_EXTRA & NORTHSET_RECEIPT_BLOB_ONLY)
assert not (NORTHSET_CLI_ECHO_REQUIRED & NORTHSET_CLI_ECHO_EXTRA)


def _pack_ic(name: str, result: DateICResult) -> dict[str, Any]:
    return {
        f"{name}_mean_ic": float(result.mean_pearson),
        f"{name}_mean_rank_ic": float(result.mean_spearman),
        f"{name}_t_ic": float(result.t_pearson),
        f"{name}_p_ic": float(result.p_pearson),
        f"{name}_n_dates": int(result.n_dates),
    }


def _nan_ic(name: str) -> dict[str, Any]:
    nan = float("nan")
    return {
        f"{name}_mean_ic": nan,
        f"{name}_mean_rank_ic": nan,
        f"{name}_t_ic": nan,
        f"{name}_p_ic": nan,
        f"{name}_n_dates": 0,
    }


def _ic_col(
    frame: pl.DataFrame,
    score: str,
    y: str,
    *,
    min_names: int,
) -> dict[str, Any]:
    if score not in frame.columns or y not in frame.columns or frame.height == 0:
        return _nan_ic(score)
    sub = frame.select(["event_time", score, y]).drop_nulls()
    if sub.height == 0:
        return _nan_ic(score)
    return _pack_ic(
        score,
        date_ic_series(
            sub[score].to_numpy().astype(float),
            sub[y].to_numpy().astype(float),
            sub["event_time"].to_numpy(),
            min_names=min_names,
        ),
    )


def _nanmean(arr: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """NaN-safe mean; accepts ndarray or small Python sequences (e.g. rate packs)."""
    finite = np.asarray(arr, dtype=float)
    if finite.size == 0:
        return float("nan")
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _panel_kyle(frame: pl.DataFrame, q_col: str) -> tuple[float, float, int]:
    slopes: list[float] = []
    fits: list[float] = []
    for _sid, group in frame.group_by("security_id", maintain_order=True):
        slope, fit = kyle_lambda(
            group["delta_mid"].to_numpy().astype(float),
            group[q_col].to_numpy().astype(float),
        )
        if np.isfinite(slope):
            slopes.append(float(slope))
        if np.isfinite(fit):
            fits.append(float(fit))
    return (
        _nanmean(np.asarray(slopes)),
        _nanmean(np.asarray(fits)),
        len(slopes),
    )


def _panel_roll(frame: pl.DataFrame) -> tuple[float, int]:
    values = [
        roll_spread(group["mid"].to_numpy().astype(float))
        for _sid, group in frame.group_by("security_id", maintain_order=True)
    ]
    finite = np.asarray(values, dtype=float)
    return _nanmean(finite), int(np.isfinite(finite).sum())


def _panel_lag1(frame: pl.DataFrame, column: str) -> tuple[float, int]:
    values = [
        lag1_corr(group[column].to_numpy().astype(float))
        for _sid, group in frame.group_by("security_id", maintain_order=True)
    ]
    finite = np.asarray(values, dtype=float)
    return _nanmean(finite), int(np.isfinite(finite).sum())


def bench_northset(bars: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Fuse daily candles with L2 and score the Northset battery."""
    if bars.height == 0:
        raise ValueError("bars must be non-empty")
    ns = config.northset
    market_view = canonical_northset_bars(
        bars,
        # Synthetic fixtures intentionally lack corporate-action columns.
        require_adjusted=bool(ns.require_adjusted_ohlc and config.data.source != "synthetic"),
    )
    bars = market_view.frame
    depth = int(ns.n_book_levels)
    seed = int(config.data.synthetic_seed)
    min_names = int(ns.min_names)
    book_path = getattr(ns, "book_panel_path", None)
    book_panel_path_str: str | None = None
    if book_path:
        from quant_fund.microstructure.book_panel import load_book_panel

        book_panel_path_str = str(book_path)
        book = load_book_panel(book_path)
        book_source = str(book["source"][0]) if book.height else "parquet"
        book_dgp = f"vendor_panel:{book_source}"
    else:
        book = synthesize_l2_from_bars(
            bars,
            depth=depth,
            seed=seed,
            base_spread_bps=float(ns.base_spread_bps),
        )
        book_source = "synthetic_lob"
        book_dgp = "synthetic_lob"
    # Shape/structure honesty: SYNTHETIC panels are ensured fail-closed before
    # rates run; external panels are never repaired (NaN rates stay honest).
    shape_columns_ensured = book_panel_path_str is None
    if shape_columns_ensured:
        ensure_book_panel_shape_columns(book)
    shape_rows = book_metrics_mod.metric_rows_from_frame(book)

    def _rate_or_nan(
        required: tuple[str, ...],
        rate_fn,
    ) -> float:
        # A missing column class is NaN (never a fake 0.0): external panels are
        # not repaired, and absent structure must read as unmeasured.
        if not set(required).issubset(book.columns):
            return float("nan")
        return rate_fn(shape_rows)

    depth_shape_rate = _rate_or_nan(
        (*book_metrics_mod.DEPTH_SHAPE_FIELDS, "n_bid_levels", "n_ask_levels"),
        book_metrics_mod.depth_shape_finite_rate,
    )
    concentration_rate = _rate_or_nan(
        book_metrics_mod.SIDE_STRUCTURE_FIELDS,
        book_metrics_mod.concentration_top_finite_rate,
    )
    queue_rate = _rate_or_nan(
        book_metrics_mod.QUEUE_STRUCTURE_FIELDS,
        book_metrics_mod.queue_priority_finite_rate,
    )
    side_notional_rate = _rate_or_nan(
        book_metrics_mod.SIDE_NOTIONAL_FIELDS,
        book_metrics_mod.side_notional_finite_rate,
    )
    tob_size_share_rate = _rate_or_nan(
        book_metrics_mod.TOB_SHARE_FIELDS,
        book_metrics_mod.tob_size_share_finite_rate,
    )
    metrics_required_ok = False
    if shape_rows:
        try:
            book_metrics_mod.assert_metrics_required_finite(shape_rows[0])
            metrics_required_ok = True
        except (TypeError, ValueError):
            metrics_required_ok = False
    floor_specs = (
        ("depth_shape_finite_rate", depth_shape_rate, ns.depth_shape_finite_floor),
        ("concentration_top_finite_rate", concentration_rate, ns.concentration_top_finite_floor),
        ("queue_priority_finite_rate", queue_rate, ns.queue_priority_finite_floor),
        ("side_notional_finite_rate", side_notional_rate, ns.side_notional_finite_floor),
        ("tob_size_share_finite_rate", tob_size_share_rate, ns.tob_size_share_finite_floor),
    )
    for rate_name, rate_value, floor_value in floor_specs:
        if floor_value is None:
            continue
        if not np.isfinite(rate_value) or float(rate_value) + 1e-12 < float(floor_value):
            raise ValueError(
                f"{rate_name}={rate_value} below floor {floor_value} (fail-closed data contract)"
            )
    book = order_flow_imbalance(book)
    book = queue_imbalance(book)
    book = vpin_proxy(book, window=max(10, min_names * 4))
    join_floor = float(getattr(ns, "book_join_coverage_floor", 0.5))
    fused = attach_candle_book_features(
        bars,
        book=book,
        depth=depth,
        seed=seed,
        max_book_age_seconds=int(ns.book_max_age_seconds),
        min_join_coverage=join_floor if book_panel_path_str else None,
    )
    # Stamp join/age honesty from fuse (attach writes scalar/literal columns).
    if "join_coverage" in fused.columns and fused.height:
        join_coverage = float(fused["join_coverage"][0])
    else:
        join_coverage = float("nan")
    if "book_age_seconds" in fused.columns and fused.height:
        _ages = fused["book_age_seconds"].to_numpy().astype(float)
        mean_book_age_seconds = float(np.nanmean(_ages)) if _ages.size else float("nan")
        max_book_age_seconds = float(np.nanmax(_ages)) if _ages.size else float("nan")
    else:
        mean_book_age_seconds = float("nan")
        max_book_age_seconds = float("nan")
    if book_panel_path_str is not None and (
        join_coverage != join_coverage or join_coverage + 1e-12 < join_floor
    ):
        raise ValueError(
            f"external book join_coverage {join_coverage:.4f} < "
            f"book_join_coverage_floor {join_floor:.4f} (fail-closed)"
        )
    amihud = amihud_illiquidity(bars).select(["security_id", "event_time", "amihud"])
    vor = volume_over_range(bars).select(["security_id", "event_time", "volume_over_range"])
    tr = true_range_frame(bars).select(["security_id", "event_time", "true_range"])
    fused = fused.join(amihud, on=["security_id", "event_time"], how="left")
    fused = fused.join(vor, on=["security_id", "event_time"], how="left")
    fused = fused.join(tr, on=["security_id", "event_time"], how="left")
    sweep_frame = liquidity_sweep_frame(bars, lookback=int(getattr(ns, "sweep_lookback", 20)))
    sweeps = sweep_frame.select(
        ["security_id", "event_time", "sweep_eligible", *sweep_output_columns()]
    )
    fused = fused.join(sweeps, on=["security_id", "event_time"], how="left")
    session = session_candles_from_daily(bars, n_candles=int(ns.n_session_candles), seed=seed)
    session_book_rows = 0
    if bool(getattr(ns, "use_session_l2", True)) and bars.height > 0 and session.height == 0:
        raise ValueError(
            "session-L2 enabled but session_candles_from_daily returned empty "
            "(fail-closed data contract)"
        )
    if bool(getattr(ns, "use_session_l2", True)) and session.height > 0:
        session_book = synthesize_session_l2(
            session,
            depth=depth,
            seed=seed,
            base_spread_bps=float(ns.base_spread_bps),
        )
        ensure_book_panel_shape_columns(session_book)
        validate_session_book_counts(session_book, n_session_candles=int(ns.n_session_candles))
        session_book_rows = int(session_book.height)
        daily_session_book = aggregate_session_book_to_daily(session_book)
        fused = fused.join(daily_session_book, on=["security_id", "event_time"], how="left")
    fused = fused.sort(["security_id", "event_time"]).with_columns(
        (
            pl.col("candle_return_close").shift(-1).over("security_id")
            / pl.col("candle_return_close")
            - 1.0
        ).alias("fwd_ret_1"),
        (pl.col("mid").shift(-1).over("security_id") - pl.col("mid")).alias("delta_mid"),
        (pl.col("bid_depth") - pl.col("ask_depth")).alias("signed_volume"),
        pl.col("ofi").shift(1).over("security_id").alias("ofi_lag"),
        # Candle close–mid diagnostic is close_mid_abs_rel; the book
        # effective_spread alias (ask−bid) must never be overwritten by it.
        (2.0 * (pl.col("candle_close") - pl.col("mid")).abs() / pl.col("mid")).alias(
            "close_mid_abs_rel"
        ),
    )
    fused = fused.with_columns(pl.col("fwd_ret_1").abs().alias("abs_fwd_ret_1"))
    scored = fused.drop_nulls(["fwd_ret_1"])
    ics: dict[str, Any] = {}
    for col in _IC_FEATURES:
        ics.update(_ic_col(scored, col, "fwd_ret_1", min_names=min_names))
    abs_ic = _ic_col(scored, "amihud", "abs_fwd_ret_1", min_names=min_names)
    ics["amihud_abs_mean_ic"] = abs_ic["amihud_mean_ic"]
    ics["amihud_abs_p_ic"] = abs_ic["amihud_p_ic"]
    ics["amihud_abs_t_ic"] = abs_ic["amihud_t_ic"]
    ics["amihud_abs_n_dates"] = abs_ic["amihud_n_dates"]
    vor_ic = _ic_col(scored, "volume_over_range", "abs_fwd_ret_1", min_names=min_names)
    ics["volume_over_range_abs_mean_ic"] = vor_ic["volume_over_range_mean_ic"]
    ics["volume_over_range_abs_p_ic"] = vor_ic["volume_over_range_p_ic"]

    lam, r2, n_kyle = _panel_kyle(fused, "signed_volume")
    ofi_arr = fused["ofi"].to_numpy().astype(float) if "ofi" in fused.columns else np.array([])
    lam_ofi, r2_ofi, n_kyle_ofi = (
        _panel_kyle(fused, "ofi") if ofi_arr.size else (float("nan"), float("nan"), 0)
    )
    session_id_rate = ohlc_identity_rate(session) if session.height else float("nan")
    reconstruct = session_reconstructs_daily_rate(bars, session)
    vol_cons = session_volume_conservation_rate(bars, session)
    chain = session_chain_rate(session)
    identity_floor = float(getattr(ns, "session_l2_identity_floor", 0.99))
    session_l2_on = bool(getattr(ns, "use_session_l2", True)) and session.height > 0
    book_uncrossed = book_uncrossed_rate(fused)
    if session_l2_on:
        enforce_session_l2_identity_floors(
            ohlc_identity_rate=ohlc_identity_rate(bars),
            session_ohlc_identity_rate=session_id_rate,
            session_reconstructs_daily_rate=reconstruct,
            session_volume_conservation_rate=vol_cons,
            session_chain_rate=chain,
            book_uncrossed_rate=book_uncrossed,
            floor=identity_floor,
        )
    nan_jump = {
        "mean_jump_ratio": float("nan"),
        "mean_rv": float("nan"),
        "mean_bv": float("nan"),
    }
    jumps = session_bipower_jump(session) if session.height else nan_jump
    bulk_vpin = session_vpin(session) if session.height else float("nan")
    sess_rv = session_realized_variance(session) if session.height else pl.DataFrame()
    session_rv_qlike = float("nan")
    if sess_rv.height:
        var_cc = ohlc_variance_frame(bars).select(
            "security_id",
            pl.col("event_time").alias("parent_event_time"),
            "var_cc",
        )
        joined_rv = sess_rv.join(var_cc, on=["security_id", "parent_event_time"], how="inner")
        y = joined_rv["var_cc"].to_numpy().astype(float)
        yhat = joined_rv["session_rv"].to_numpy().astype(float)
        mask = np.isfinite(y) & np.isfinite(yhat)
        if int(mask.sum()) >= 8:
            session_rv_qlike = qlike(y[mask], yhat[mask])
    dm_gk = dm_range_vs_park(bars, which="gk")
    dm_rs = dm_range_vs_park(bars, which="rs")
    dm_split = dm_split_vs_park(bars)
    geo = geometry_rates(fused)
    sweep = sweep_rates(fused)
    sweep_evidence = sweep_evidence_battery(sweep_frame, config)

    def _cond_fwd_mean(flag: str) -> float:
        if flag not in scored.columns:
            return float("nan")
        sub = scored.filter(pl.col(flag) == 1.0)
        if sub.height == 0:
            return float("nan")
        mean = sub["fwd_ret_1"].mean()
        return float(cast(float, mean)) if mean is not None else float("nan")

    n_sweep_high = int((fused["sweep_high"] == 1.0).sum()) if "sweep_high" in fused.columns else 0
    n_sweep_low = int((fused["sweep_low"] == 1.0).sum()) if "sweep_low" in fused.columns else 0
    bar_source = str(config.data.source).strip()
    bars_synthetic = bar_source.lower() == "synthetic"
    book_synthetic = book_source.lower() in {
        "synthetic",
        "synthetic_lob",
        "synthetic_reconstruction",
    }
    session_synthetic = bool(getattr(ns, "use_session_l2", True))
    if bars_synthetic and book_synthetic:
        evidence_label = "SYNTHETIC"
        family_dgp = "synthetic_lob"
    elif not book_synthetic:
        # The candle label remains SYNTHETIC, but the book provenance must be
        # the authoritative vendor-panel DGP rather than a vague mixture tag.
        evidence_label = "SYNTHETIC" if bars_synthetic else bar_source
        family_dgp = book_dgp
    elif bars_synthetic or book_synthetic or session_synthetic:
        evidence_label = "MIXED_SYNTHETIC_DERIVED"
        family_dgp = "mixed_sources"
    else:
        evidence_label = bar_source
        family_dgp = "empirical"
    quoted = fused["spread"].to_numpy().astype(float) if "spread" in fused.columns else np.array([])
    slope_bid = (
        fused["bid_log_size_slope"].to_numpy().astype(float)
        if "bid_log_size_slope" in fused.columns
        else np.array([])
    )
    slope_ask = (
        fused["ask_log_size_slope"].to_numpy().astype(float)
        if "ask_log_size_slope" in fused.columns
        else np.array([])
    )
    eff = (
        fused["effective_spread"].to_numpy().astype(float)
        if "effective_spread" in fused.columns
        else np.array([])
    )
    tr_arr = (
        fused["true_range"].to_numpy().astype(float)
        if "true_range" in fused.columns
        else np.array([])
    )
    vpin_arr = fused["vpin"].to_numpy().astype(float) if "vpin" in fused.columns else np.array([])
    qi_arr = (
        fused["queue_imbalance"].to_numpy().astype(float)
        if "queue_imbalance" in fused.columns
        else np.array([])
    )
    semi_up, semi_down = realized_semivariance(bars)
    ar_spread = abdi_ranaldo_spread(bars)
    cs_spread = corwin_schultz_spread(bars)
    panel_roll, n_roll = _panel_roll(fused)
    panel_mid_lag, n_mid_lag = _panel_lag1(fused, "mid")
    panel_ofi_lag, n_ofi_lag = _panel_lag1(fused, "ofi")
    out: dict[str, Any] = {
        "family": "northset",
        "product": "Northset",
        "dgp": family_dgp,
        "label": evidence_label,
        "data_source": evidence_label,
        "price_basis": market_view.price_basis,
        "return_basis": market_view.return_basis,
        "book_dgp": book_dgp,
        "component_sources": {
            "bars": bar_source,
            "book": book_source,
            "session_candles": "synthetic_reconstruction",
            "session_book": ("synthetic_reconstruction" if session_synthetic else "disabled"),
        },
        "book_hypothesis_eligible": bool(bars_synthetic or not book_synthetic),
        "session_book_hypothesis_eligible": bool(bars_synthetic),
        "sweep_evidence_scope": (
            "synthetic"
            if bars_synthetic
            else (
                "empirical_adjusted"
                if market_view.price_basis == "split_adjusted"
                else "fixture_raw_unadjusted"
            )
        ),
        "book_panel_path": book_panel_path_str,
        "join_coverage": float(join_coverage),
        "mean_book_age_seconds": float(mean_book_age_seconds),
        "max_book_age_seconds": float(max_book_age_seconds),
        "book_join_coverage_floor": float(join_floor),
        "depth_shape_finite_rate": float(depth_shape_rate),
        "depth_shape_finite_floor": ns.depth_shape_finite_floor,
        "concentration_top_finite_rate": float(concentration_rate),
        "concentration_top_finite_floor": ns.concentration_top_finite_floor,
        "queue_priority_finite_rate": float(queue_rate),
        "queue_priority_finite_floor": ns.queue_priority_finite_floor,
        "side_notional_finite_rate": float(side_notional_rate),
        "side_notional_finite_floor": ns.side_notional_finite_floor,
        "tob_size_share_finite_rate": float(tob_size_share_rate),
        "tob_size_share_finite_floor": ns.tob_size_share_finite_floor,
        # Aggregate of LOB structure finite-rate companions (≠ candle finite_rate_*).
        "structure_finite_rate": float(
            _nanmean(
                np.asarray(
                    [
                        float(concentration_rate),
                        float(queue_rate),
                        float(side_notional_rate),
                        float(tob_size_share_rate),
                    ],
                    dtype=float,
                )
            )
        ),
        "mean_tob_size_share": (
            _nanmean(book["tob_size_share"].to_numpy().astype(float))
            if "tob_size_share" in book.columns
            else float("nan")
        ),
        # TOB notional share ∈ (0,1] when finite — ≠ mean_tob_size_share (size vs notional).
        "mean_tob_notional_share": (
            _nanmean(book["tob_notional_share"].to_numpy().astype(float))
            if "tob_notional_share" in book.columns
            else float("nan")
        ),
        # Notional imbalance ∈ [-1,1] — ≠ mean_depth_imbalance / imbalance_top.
        "mean_notional_imbalance": (
            _nanmean(book["notional_imbalance"].to_numpy().astype(float))
            if "notional_imbalance" in book.columns
            else float("nan")
        ),
        # Size concentration tops ∈ (0,1] when finite — ≠ queue_priority_proxy.
        "mean_bid_size_concentration_top": (
            _nanmean(book["bid_size_concentration_top"].to_numpy().astype(float))
            if "bid_size_concentration_top" in book.columns
            else float("nan")
        ),
        "mean_ask_size_concentration_top": (
            _nanmean(book["ask_size_concentration_top"].to_numpy().astype(float))
            if "ask_size_concentration_top" in book.columns
            else float("nan")
        ),
        # Side depths ≥0 — companion to imbalance means; not IC features.
        "mean_bid_depth": (
            _nanmean(book["bid_depth"].to_numpy().astype(float))
            if "bid_depth" in book.columns
            else float("nan")
        ),
        "mean_ask_depth": (
            _nanmean(book["ask_depth"].to_numpy().astype(float))
            if "ask_depth" in book.columns
            else float("nan")
        ),
        # Side/TOB notional proxies ≥0 when finite.
        "mean_side_notional_proxy_bid": (
            _nanmean(book["side_notional_proxy_bid"].to_numpy().astype(float))
            if "side_notional_proxy_bid" in book.columns
            else float("nan")
        ),
        "mean_side_notional_proxy_ask": (
            _nanmean(book["side_notional_proxy_ask"].to_numpy().astype(float))
            if "side_notional_proxy_ask" in book.columns
            else float("nan")
        ),
        "mean_top_of_book_notional_proxy": (
            _nanmean(book["top_of_book_notional_proxy"].to_numpy().astype(float))
            if "top_of_book_notional_proxy" in book.columns
            else float("nan")
        ),
        # spread/mid — ≠ mean_spread_bps (bps scale); ≥0 when finite.
        "mean_spread_over_mid": (
            _nanmean(book["spread_over_mid"].to_numpy().astype(float))
            if "spread_over_mid" in book.columns
            else float("nan")
        ),
        # Price slopes (signed OK) — ≠ mean_bid/ask_log_size_slope.
        "mean_bid_log_price_slope": (
            _nanmean(book["bid_log_price_slope"].to_numpy().astype(float))
            if "bid_log_price_slope" in book.columns
            else float("nan")
        ),
        "mean_ask_log_price_slope": (
            _nanmean(book["ask_log_price_slope"].to_numpy().astype(float))
            if "ask_log_price_slope" in book.columns
            else float("nan")
        ),
        # Mean log tick spacings ≥0 when finite.
        "mean_bid_mean_log_tick_spacing": (
            _nanmean(book["bid_mean_log_tick_spacing"].to_numpy().astype(float))
            if "bid_mean_log_tick_spacing" in book.columns
            else float("nan")
        ),
        "mean_ask_mean_log_tick_spacing": (
            _nanmean(book["ask_mean_log_tick_spacing"].to_numpy().astype(float))
            if "ask_mean_log_tick_spacing" in book.columns
            else float("nan")
        ),
        # Top sizes and level counts ≥0 when finite.
        "mean_top_bid_size": (
            _nanmean(book["top_bid_size"].to_numpy().astype(float))
            if "top_bid_size" in book.columns
            else float("nan")
        ),
        "mean_top_ask_size": (
            _nanmean(book["top_ask_size"].to_numpy().astype(float))
            if "top_ask_size" in book.columns
            else float("nan")
        ),
        "mean_n_bid_levels": (
            _nanmean(book["n_bid_levels"].to_numpy().astype(float))
            if "n_bid_levels" in book.columns
            else float("nan")
        ),
        "mean_n_ask_levels": (
            _nanmean(book["n_ask_levels"].to_numpy().astype(float))
            if "n_ask_levels" in book.columns
            else float("nan")
        ),
        # Bid/ask queue priority proxies ∈ [0,1] when finite (≠ size_concentration_top).
        "mean_queue_priority_proxy": (
            _nanmean(book["queue_priority_proxy"].to_numpy().astype(float))
            if "queue_priority_proxy" in book.columns
            else float("nan")
        ),
        "mean_ask_queue_priority_proxy": (
            _nanmean(book["ask_queue_priority_proxy"].to_numpy().astype(float))
            if "ask_queue_priority_proxy" in book.columns
            else float("nan")
        ),
        "shape_columns_ensured": bool(shape_columns_ensured),
        "metrics_required_finite_ok": bool(metrics_required_ok),
        "n_bars": int(bars.height),
        "n_fused": int(fused.height),
        "n_scored": int(scored.height),
        "n_session_candles": int(session.height),
        "n_session_book_rows": int(session_book_rows),
        "use_session_l2": bool(getattr(ns, "use_session_l2", True)),
        "session_l2_identity_floor": float(getattr(ns, "session_l2_identity_floor", 0.99)),
        "session_l2_identity_gate": "enforced" if session_l2_on else "skipped",
        "depth": depth,
        "ohlc_identity_rate": ohlc_identity_rate(bars),
        "session_ohlc_identity_rate": session_id_rate,
        "session_reconstructs_daily_rate": reconstruct,
        "session_volume_conservation_rate": vol_cons,
        "session_chain_rate": chain,
        "book_uncrossed_rate": float(book_uncrossed),
        "gap_finite_rate": gap_finite_rate(bars),
        **geo,
        **sweep,
        **ics,
        "kyle_lambda": float(lam),
        "kyle_r2": float(r2),
        "kyle_n_securities": int(n_kyle),
        "kyle_ofi_lambda": float(lam_ofi),
        "kyle_ofi_r2": float(r2_ofi),
        "kyle_ofi_n_securities": int(n_kyle_ofi),
        "impact_estimator_scope": "per_security_equal_weight",
        "impact_proxy_warning": "depth_or_ofi_proxy_not_signed_trade_flow",
        "roll_spread": float(panel_roll),
        "roll_n_securities": int(n_roll),
        "corwin_schultz_spread": float(cs_spread),
        "corwin_schultz_pair_scope": "prior_and_current_bar",
        "abdi_ranaldo_spread": float(ar_spread),
        "mean_quoted_spread": _nanmean(quoted),
        "mean_effective_spread": _nanmean(eff),
        "mean_spread_bps": (
            _nanmean(fused["spread_bps"].to_numpy().astype(float))
            if "spread_bps" in fused.columns
            else float("nan")
        ),
        "mean_half_spread": (
            _nanmean(fused["half_spread"].to_numpy().astype(float))
            if "half_spread" in fused.columns
            else float("nan")
        ),
        "mean_half_spread_bps": (
            _nanmean(fused["half_spread_bps"].to_numpy().astype(float))
            if "half_spread_bps" in fused.columns
            else float("nan")
        ),
        # Candle close–mid diagnostic (2·|C−mid|/mid) — never the book spread.
        "mean_close_mid_abs_rel": (
            _nanmean(fused["close_mid_abs_rel"].to_numpy().astype(float))
            if "close_mid_abs_rel" in fused.columns
            else float("nan")
        ),
        "mean_microprice_weight_balance": _nanmean(
            fused["microprice_weight_balance"].to_numpy().astype(float)
        )
        if "microprice_weight_balance" in fused.columns
        else float("nan"),
        # Absolute mid gap (price units) — ≠ _bps; candle_order_book stamps same key.
        "mean_microprice_minus_mid": (
            _nanmean(fused["microprice_minus_mid"].to_numpy().astype(float))
            if "microprice_minus_mid" in fused.columns
            else float("nan")
        ),
        # 1e4*(mp-mid)/mid — IC feature companion; ≠ mean_microprice_minus_mid.
        "mean_microprice_minus_mid_bps": (
            _nanmean(fused["microprice_minus_mid_bps"].to_numpy().astype(float))
            if "microprice_minus_mid_bps" in fused.columns
            else float("nan")
        ),
        "mean_bid_log_size_slope": _nanmean(slope_bid),
        "mean_ask_log_size_slope": _nanmean(slope_ask),
        "mean_true_range": _nanmean(tr_arr),
        "mid_lag1_corr": float(panel_mid_lag),
        "mid_lag1_n_securities": int(n_mid_lag),
        "ofi_lag1_corr": float(panel_ofi_lag),
        "ofi_lag1_n_securities": int(n_ofi_lag),
        "parkinson_qlike_vs_cc": float(parkinson_vs_close_to_close(bars)),
        "garman_klass_qlike_vs_cc": float(garman_klass_vs_close_to_close(bars)),
        "rogers_satchell_qlike_vs_cc": float(rogers_satchell_vs_close_to_close(bars)),
        "yang_zhang_variance": float(yang_zhang_variance(bars)),
        "yang_zhang_qlike_vs_cc": float(yang_zhang_vs_close_to_close(bars)),
        "yang_zhang_qlike_scope": "per_security_expanding_oos",
        "vpin_method": "count_window_bulk_ofi_proxy",
        "overnight_plus_oc_qlike_vs_cc": float(overnight_plus_oc_vs_close_to_close(bars)),
        "overnight_share": float(overnight_share(bars)),
        "session_rv_qlike_vs_cc": float(session_rv_qlike),
        "session_mean_jump_ratio": float(jumps["mean_jump_ratio"]),
        "session_mean_rv": float(jumps["mean_rv"]),
        "session_mean_bv": float(jumps["mean_bv"]),
        "session_bulk_vpin": float(bulk_vpin),
        "session_vpin_method": "daily_bulk_imbalance_ratio",
        "semi_up": float(semi_up),
        "semi_down": float(semi_down),
        "dm_gk_vs_park_stat": float(dm_gk["statistic"]),
        "dm_gk_vs_park_p": float(dm_gk["p_value"]),
        "dm_gk_vs_park_preferred": str(dm_gk["preferred"]),
        "dm_rs_vs_park_stat": float(dm_rs["statistic"]),
        "dm_rs_vs_park_p": float(dm_rs["p_value"]),
        "dm_rs_vs_park_preferred": str(dm_rs["preferred"]),
        "dm_split_vs_park_stat": float(dm_split["statistic"]),
        "dm_split_vs_park_p": float(dm_split["p_value"]),
        "dm_split_vs_park_preferred": str(dm_split["preferred"]),
        "amihud_mean": (
            _nanmean(fused["amihud"].to_numpy().astype(float))
            if "amihud" in fused.columns
            else float("nan")
        ),
        "vpin_mean": _nanmean(vpin_arr),
        "queue_imbalance_mean": _nanmean(qi_arr),
        # Depth imbalance mean ∈ [-1,1] when finite — ≠ queue_imbalance_mean / imbalance_top.
        "mean_depth_imbalance": (
            _nanmean(fused["imbalance_depth"].to_numpy().astype(float))
            if "imbalance_depth" in fused.columns
            else float("nan")
        ),
        # |imbalance_depth| mean ∈ [0,1] — ≠ mean_depth_imbalance (signed).
        "mean_depth_imbalance_abs": (
            _nanmean(fused["depth_imbalance_abs"].to_numpy().astype(float))
            if "depth_imbalance_abs" in fused.columns
            else float("nan")
        ),
        # Top-of-book size imbalance ∈ [-1,1] — ≠ mean_depth_imbalance / notional.
        # touch_size_imbalance is an alias of imbalance_top; do not dual-stamp.
        "mean_imbalance_top": (
            _nanmean(fused["imbalance_top"].to_numpy().astype(float))
            if "imbalance_top" in fused.columns
            else (
                _nanmean(book["imbalance_top"].to_numpy().astype(float))
                if "imbalance_top" in book.columns
                else float("nan")
            )
        ),
        "book_source": book_source,
        "session_ofi_sum_mean": _nanmean(fused["session_ofi_sum"].to_numpy().astype(float))
        if "session_ofi_sum" in fused.columns
        else float("nan"),
        # Path |OFI| sum mean — VPIN denominator companion; ≠ |session_ofi_sum_mean|; ≥0 when finite.
        "mean_session_ofi_abs_sum": _nanmean(fused["session_ofi_abs_sum"].to_numpy().astype(float))
        if "session_ofi_abs_sum" in fused.columns
        else float("nan"),
        "session_book_vpin_mean": _nanmean(fused["session_book_vpin"].to_numpy().astype(float))
        if "session_book_vpin" in fused.columns
        else float("nan"),
        # Path mean of session L2 imbalance — NOT daily imbalance_top / queue_imbalance.
        "mean_session_imbalance_mean": _nanmean(
            fused["session_imbalance_mean"].to_numpy().astype(float)
        )
        if "session_imbalance_mean" in fused.columns
        else float("nan"),
        # Path dispersion — ≠ path mean ≠ last-snap close imbalance; ≥0 when finite.
        # Receipt-only (not an IC feature).
        "mean_session_imbalance_std": _nanmean(
            fused["session_imbalance_std"].to_numpy().astype(float)
        )
        if "session_imbalance_std" in fused.columns
        else float("nan"),
        # Session-path mean of intra-day spread_bps — NOT daily mean_spread_bps.
        "mean_session_spread_bps_mean": _nanmean(
            fused["session_spread_bps_mean"].to_numpy().astype(float)
        )
        if "session_spread_bps_mean" in fused.columns
        else float("nan"),
        # Last-snap close spread — ≠ path mean_session_spread_bps_mean, ≠ daily mean_spread_bps.
        "mean_session_close_spread_bps": _nanmean(
            fused["session_close_spread_bps"].to_numpy().astype(float)
        )
        if "session_close_spread_bps" in fused.columns
        else float("nan"),
        # Last-snap close imbalance — ≠ path mean_session_imbalance_mean, ≠ daily imbalance_top.
        "mean_session_close_imbalance": _nanmean(
            fused["session_close_imbalance"].to_numpy().astype(float)
        )
        if "session_close_imbalance" in fused.columns
        else float("nan"),
        # Last-snap micro — ≠ daily microprice_minus_mid_bps mean (fuse/session companion).
        "mean_session_close_micro_bps": _nanmean(
            fused["session_close_micro_bps"].to_numpy().astype(float)
        )
        if "session_close_micro_bps" in fused.columns
        else float("nan"),
        # Last-snap mid — ≠ daily mid/close; companion of close_micro (not a Sharpe claim).
        "mean_session_close_mid": _nanmean(fused["session_close_mid"].to_numpy().astype(float))
        if "session_close_mid" in fused.columns
        else float("nan"),
        # Last-snap depths — ≠ daily bid_depth/ask_depth means; ≥0 when finite.
        "mean_session_close_bid_depth": _nanmean(
            fused["session_close_bid_depth"].to_numpy().astype(float)
        )
        if "session_close_bid_depth" in fused.columns
        else float("nan"),
        "mean_session_close_ask_depth": _nanmean(
            fused["session_close_ask_depth"].to_numpy().astype(float)
        )
        if "session_close_ask_depth" in fused.columns
        else float("nan"),
        "mean_session_book_snaps": _nanmean(fused["n_session_book_snaps"].to_numpy().astype(float))
        if "n_session_book_snaps" in fused.columns
        else float("nan"),
        "n_sweep_high": n_sweep_high,
        "n_sweep_low": n_sweep_low,
        "sweep_min_fold_positive_fraction": float(ns.sweep_min_fold_positive_fraction),
        "mean_fwd_ret_after_high_reclaim": _cond_fwd_mean("sweep_high_reclaim"),
        "mean_fwd_ret_after_low_reclaim": _cond_fwd_mean("sweep_low_reclaim"),
        "mean_fwd_ret_after_high_follow": _cond_fwd_mean("sweep_high_follow"),
        "mean_fwd_ret_after_low_follow": _cond_fwd_mean("sweep_low_follow"),
        "sweep_evidence": sweep_evidence,
        "research_only": True,
        "claim": "research_diagnostic_only",
    }
    out["microprice_p_ic"] = out.get("microprice_minus_mid_bps_p_ic", float("nan"))
    out["microprice_t_ic"] = out.get("microprice_minus_mid_bps_t_ic", float("nan"))
    out["clv_p_ic"] = out.get("close_location_value_p_ic", float("nan"))
    out["clv_t_ic"] = out.get("close_location_value_t_ic", float("nan"))
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
    # Matched-control evidence (H44/H45): direction-matched event-minus-control
    # excess difference vs same-date eligible non-swept names.
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
    out["sweep_primary_test_id"] = str(primary.get("id") or PRIMARY_EXECUTABLE_TEST["id"])
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
    if bool(getattr(ns, "include_kyle_ofi", False)):
        from quant_fund.northset.kyle_ofi import bench_kyle_ofi_fused

        kyle_blob = bench_kyle_ofi_fused(
            bars,
            book,
            min_names=min_names,
            min_join_coverage=join_floor if book_panel_path_str else None,
            book_panel_path=book_panel_path_str,
            label=(
                "SYNTHETIC"
                if str(config.data.source).strip().lower() == "synthetic"
                else str(config.data.source)
            ),
        )
        if not isinstance(kyle_blob, dict) or not kyle_blob.get("research_only"):
            raise AssertionError("kyle_ofi nest missing research_only honesty")
        out["kyle_ofi"] = kyle_blob
        out["include_kyle_ofi"] = True
    else:
        out["include_kyle_ofi"] = False

    if not family_blob_forbidden_metrics_absent(out):
        raise AssertionError("northset bench leaked forbidden research keys")
    return out
